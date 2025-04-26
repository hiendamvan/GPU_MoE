#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace py = pybind11;

#define CHUNK_SIZE 256
#define VEC_SIZE 4

// SiLU optimized device function
__device__ float silu_optimized(float x) {
    float exp_val = __expf(-x);
    return x * __frcp_rn(1.0f + exp_val);
}

__global__ void moe_kernel(
    const float* __restrict__ hidden_states,
    const float* __restrict__ w1,
    const float* __restrict__ w2,
    const int* __restrict__ topk_ids,
    float* __restrict__ out,
    int token_num,
    int topk,
    int expert,
    int model_dim,
    int inter_dim) 
{
    int block_idx = blockIdx.x;
    int t = block_idx / topk;
    int k = block_idx % topk;
    int tid = threadIdx.x;

    if (t >= token_num) return;

    int E_id = topk_ids[t * topk + k];
    extern __shared__ float shared_mem[]; // size = CHUNK_SIZE + MODEL_DIM partial sum

    float* s_act_out_chunk = shared_mem;                       // CHUNK_SIZE floats
    float* s_partial_sum = (float*)&shared_mem[CHUNK_SIZE];     // MODEL_DIM floats

    // Zero initialize partial sums
    for (int d = tid; d < model_dim; d += blockDim.x) {
        s_partial_sum[d] = 0.0f;
    }
    __syncthreads();

    int num_chunks = (inter_dim + CHUNK_SIZE - 1) / CHUNK_SIZE;

    for (int c = 0; c < num_chunks; c++) {
        int j_start = c * CHUNK_SIZE;
        int j_end = min(j_start + CHUNK_SIZE, inter_dim);
        int chunk_len = j_end - j_start;

        // Step 1: tính gate * up cho chunk hiện tại
        for (int j = j_start + tid; j < j_end; j += blockDim.x) {
            float gate = 0.0f, up = 0.0f;
            int m = 0;

            for (; m <= model_dim - VEC_SIZE; m += VEC_SIZE) {
                float4 h_vec = *reinterpret_cast<const float4*>(&hidden_states[t * model_dim + m]);
                float4 w1_gate_vec = *reinterpret_cast<const float4*>(&w1[E_id * (2 * inter_dim) * model_dim + j * model_dim + m]);
                float4 w1_up_vec = *reinterpret_cast<const float4*>(&w1[E_id * (2 * inter_dim) * model_dim + (inter_dim + j) * model_dim + m]);

                gate += h_vec.x * w1_gate_vec.x + h_vec.y * w1_gate_vec.y + h_vec.z * w1_gate_vec.z + h_vec.w * w1_gate_vec.w;
                up += h_vec.x * w1_up_vec.x + h_vec.y * w1_up_vec.y + h_vec.z * w1_up_vec.z + h_vec.w * w1_up_vec.w;
            }

            for (; m < model_dim; m++) {
                float h = hidden_states[t * model_dim + m];
                gate += h * w1[E_id * (2 * inter_dim) * model_dim + j * model_dim + m];
                up += h * w1[E_id * (2 * inter_dim) * model_dim + (inter_dim + j) * model_dim + m];
            }

            s_act_out_chunk[j - j_start] = silu_optimized(gate) * up;
        }
        __syncthreads();

        // Step 2: multiply act_out_chunk với w2 và accumulate vào s_partial_sum
        for (int d = tid; d < model_dim; d += blockDim.x) {
            float partial = 0.0f;

            int j = 0;
            for (; j <= chunk_len - VEC_SIZE; j += VEC_SIZE) {
                float4 act_vec = *reinterpret_cast<float4*>(&s_act_out_chunk[j]);
                float4 w2_vec = *reinterpret_cast<const float4*>(&w2[E_id * model_dim * inter_dim + d * inter_dim + j_start + j]);

                partial += act_vec.x * w2_vec.x + act_vec.y * w2_vec.y + act_vec.z * w2_vec.z + act_vec.w * w2_vec.w;
            }

            for (; j < chunk_len; j++) {
                partial += s_act_out_chunk[j] * w2[E_id * model_dim * inter_dim + d * inter_dim + j_start + j];
            }

            s_partial_sum[d] += partial;
        }
        __syncthreads();
    }

    // Step 3: ghi kết quả từ shared memory ra global memory
    for (int d = tid; d < model_dim; d += blockDim.x) {
        out[t * topk * model_dim + k * model_dim + d] = s_partial_sum[d];
    }
}

torch::Tensor launch_custom_moe(
    torch::Tensor hidden_states, torch::Tensor w1, torch::Tensor w2,
    torch::Tensor topk_weight, torch::Tensor topk_ids) 
{
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
    size_t shared_mem_size = (CHUNK_SIZE + model_dim) * sizeof(float);

    auto [hidden_states_ptr, w1_ptr, w2_ptr, topk_ids_ptr, out_ptr] = 
        std::make_tuple(
            hidden_states.data_ptr<float>(),
            w1.data_ptr<float>(),
            w2.data_ptr<float>(),
            topk_ids.data_ptr<int>(),
            out.data_ptr<float>()
        );

    hipLaunchKernelGGL(
        moe_kernel,
        dim3(num_blocks),
        dim3(threads_per_block),
        shared_mem_size,
        0,
        hidden_states_ptr, w1_ptr, w2_ptr, topk_ids_ptr, out_ptr,
        token_num, topk, expert, model_dim, inter_dim
    );

    auto weighted_out = out * topk_weight.view({token_num, topk, 1});
    return weighted_out.sum(1);
}

PYBIND11_MODULE(custom_moe, m) {
    m.def("launch_custom_moe", &launch_custom_moe, "Optimized MoE kernel with HIP vectorization and fast SiLU");
}