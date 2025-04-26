#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace py = pybind11;

// Hàm kích hoạt SiLU tối ưu
__device__ __forceinline__ float silu(float x) {
    float sigmoid = 1.0f / (1.0f + __expf(-x));
    return x * sigmoid;
}

// Kernel cho inter_dim nhỏ (<= 1024)
__global__ void moe_kernel_small_inter_dim(
    const float* __restrict__ hidden_states,
    const float* __restrict__ w1,
    const float* __restrict__ w2,
    const int* __restrict__ topk_ids,
    float* __restrict__ out,
    int token_num,
    int topk,
    int expert,
    int model_dim,
    int inter_dim) {
    
    constexpr int VEC_SIZE = 4;
    const int chunk_size = min(256, inter_dim);
    const int t = blockIdx.x / topk;
    const int k = blockIdx.x % topk;
    
    if (t >= token_num) return;
    
    const int E_id = topk_ids[t * topk + k];
    extern __shared__ float s_act_out[];
    
    // Xử lý từng chunk
    for (int c_start = 0; c_start < inter_dim; c_start += chunk_size) {
        const int c_end = min(c_start + chunk_size, inter_dim);
        const int chunk_len = c_end - c_start;
        
        // Tính toán song song cho từng phần tử trong chunk
        for (int j = c_start + threadIdx.x; j < c_end; j += blockDim.x) {
            float gate = 0.0f, up = 0.0f;
            
            // Vectorized computation
            for (int m = 0; m < model_dim; m += VEC_SIZE) {
                if (m + VEC_SIZE <= model_dim) {
                    float4 h_vec = *reinterpret_cast<const float4*>(&hidden_states[t * model_dim + m]);
                    float4 w1_gate_vec = *reinterpret_cast<const float4*>(&w1[E_id * 2 * inter_dim * model_dim + j * model_dim + m]);
                    float4 w1_up_vec = *reinterpret_cast<const float4*>(&w1[E_id * 2 * inter_dim * model_dim + (inter_dim + j) * model_dim + m]);
                    
                    gate += h_vec.x * w1_gate_vec.x + h_vec.y * w1_gate_vec.y + h_vec.z * w1_gate_vec.z + h_vec.w * w1_gate_vec.w;
                    up += h_vec.x * w1_up_vec.x + h_vec.y * w1_up_vec.y + h_vec.z * w1_up_vec.z + h_vec.w * w1_up_vec.w;
                } else {
                    // Xử lý phần tử cuối không đủ vector
                    for (int m_remain = m; m_remain < model_dim; m_remain++) {
                        float h = hidden_states[t * model_dim + m_remain];
                        gate += h * w1[E_id * 2 * inter_dim * model_dim + j * model_dim + m_remain];
                        up += h * w1[E_id * 2 * inter_dim * model_dim + (inter_dim + j) * model_dim + m_remain];
                    }
                }
            }
            s_act_out[j - c_start] = silu(gate) * up;
        }
        __syncthreads();
        
        // Reduction trong shared memory trước khi atomicAdd
        for (int d = threadIdx.x; d < model_dim; d += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < chunk_len; j++) {
                sum += s_act_out[j] * w2[E_id * model_dim * inter_dim + d * inter_dim + c_start + j];
            }
            atomicAdd(&out[t * topk * model_dim + k * model_dim + d], sum);
        }
        __syncthreads();
    }
}

// Kernel cho inter_dim lớn (> 1024)
__global__ void moe_kernel_large_inter_dim(
    const float* __restrict__ hidden_states,
    const float* __restrict__ w1,
    const float* __restrict__ w2,
    const int* __restrict__ topk_ids,
    float* __restrict__ out,
    int token_num,
    int topk,
    int expert,
    int model_dim,
    int inter_dim) {
    
    constexpr int VEC_SIZE = 4;
    const int t = blockIdx.x / topk;
    const int k = blockIdx.x % topk;
    
    if (t >= token_num) return;
    
    const int E_id = topk_ids[t * topk + k];
    
    // Mỗi thread xử lý một phần của model_dim
    const int d_start = threadIdx.x * (model_dim / blockDim.x);
    const int d_end = (threadIdx.x == blockDim.x - 1) ? model_dim : (threadIdx.x + 1) * (model_dim / blockDim.x);
    
    // Tích lũy kết quả tạm thời
    float temp_result[32] = {0}; // Giả sử tối đa 32 phần tử mỗi thread
    
    for (int j = 0; j < inter_dim; j++) {
        float gate = 0.0f, up = 0.0f;
        
        // Tính gate và up
        for (int m = 0; m < model_dim; m += VEC_SIZE) {
            if (m + VEC_SIZE <= model_dim) {
                float4 h_vec = *reinterpret_cast<const float4*>(&hidden_states[t * model_dim + m]);
                float4 w1_gate_vec = *reinterpret_cast<const float4*>(&w1[E_id * 2 * inter_dim * model_dim + j * model_dim + m]);
                float4 w1_up_vec = *reinterpret_cast<const float4*>(&w1[E_id * 2 * inter_dim * model_dim + (inter_dim + j) * model_dim + m]);
                
                gate += h_vec.x * w1_gate_vec.x + h_vec.y * w1_gate_vec.y + h_vec.z * w1_gate_vec.z + h_vec.w * w1_gate_vec.w;
                up += h_vec.x * w1_up_vec.x + h_vec.y * w1_up_vec.y + h_vec.z * w1_up_vec.z + h_vec.w * w1_up_vec.w;
            } else {
                // Xử lý phần tử cuối
                for (int m_remain = m; m_remain < model_dim; m_remain++) {
                    float h = hidden_states[t * model_dim + m_remain];
                    gate += h * w1[E_id * 2 * inter_dim * model_dim + j * model_dim + m_remain];
                    up += h * w1[E_id * 2 * inter_dim * model_dim + (inter_dim + j) * model_dim + m_remain];
                }
            }
        }
        
        float activated = silu(gate) * up;
        
        // Nhân với w2 và tích lũy
        for (int d = d_start; d < d_end; d++) {
            temp_result[d - d_start] += activated * w2[E_id * model_dim * inter_dim + d * inter_dim + j];
        }
    }
    
    // Ghi kết quả
    for (int d = d_start; d < d_end; d++) {
        atomicAdd(&out[t * topk * model_dim + k * model_dim + d], temp_result[d - d_start]);
    }
}

torch::Tensor launch_custom_moe(
    torch::Tensor hidden_states, torch::Tensor w1, torch::Tensor w2,
    torch::Tensor topk_weight, torch::Tensor topk_ids) {
    
    int token_num = hidden_states.size(0);
    int model_dim = hidden_states.size(1);
    int topk = topk_weight.size(1);
    int expert = w2.size(0);
    int inter_dim = w2.size(2);
    
    auto options = torch::TensorOptions()
        .dtype(hidden_states.dtype())
        .device(hidden_states.device())
        .requires_grad(false);
        
    auto out = torch::zeros({token_num, topk, model_dim}, options);
    
    int num_blocks = token_num * topk;
    int threads_per_block = 256;
    
    // Auto-tuning: Chọn kernel dựa trên inter_dim
    bool use_small_kernel = inter_dim <= 1024;
    size_t shared_mem_size = use_small_kernel ? min(256, inter_dim) * sizeof(float) : 0;
    
    auto [hidden_states_ptr, w1_ptr, w2_ptr, topk_ids_ptr, out_ptr] = 
        std::make_tuple(
            hidden_states.data_ptr<float>(),
            w1.data_ptr<float>(),
            w2.data_ptr<float>(),
            topk_ids.data_ptr<int>(),
            out.data_ptr<float>()
        );
    
    // Chọn kernel phù hợp
    if (use_small_kernel) {
        hipLaunchKernelGGL(
            moe_kernel_small_inter_dim,
            dim3(num_blocks),
            dim3(threads_per_block),
            shared_mem_size,
            0,
            hidden_states_ptr, w1_ptr, w2_ptr, topk_ids_ptr, out_ptr,
            token_num, topk, expert, model_dim, inter_dim
        );
    } else {
        // Điều chỉnh số thread cho kernel lớn
        int adjusted_threads = min(1024, model_dim);
        hipLaunchKernelGGL(
            moe_kernel_large_inter_dim,
            dim3(num_blocks),
            dim3(adjusted_threads),
            0,
            0,
            hidden_states_ptr, w1_ptr, w2_ptr, topk_ids_ptr, out_ptr,
            token_num, topk, expert, model_dim, inter_dim
        );
    }
    
    auto weighted_out = out * topk_weight.view({token_num, topk, 1});
    return weighted_out.sum(1);
}

PYBIND11_MODULE(custom_moe, m) {
    m.def("launch_custom_moe", &launch_custom_moe, "custom MoE kernel with auto-tuning");
}