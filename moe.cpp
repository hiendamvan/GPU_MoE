#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include <hip/hip_runtime.h>

namespace py = pybind11;

__device__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void moe_kernel_hip(
    const float* __restrict__ hidden_states,
    const float* __restrict__ w1,
    const float* __restrict__ w2,
    const int* __restrict__ topk_ids,
    float* __restrict__ out,
    int token_num,
    int topk,
    int expert,
    int model_dim,
    int inter_dim
) {
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int token_id = global_tid / topk;
    int k = global_tid % topk;

    if (token_id >= token_num) return;

    int expert_id = topk_ids[token_id * topk + k];

    // Pointers to hidden state and output slice
    const float* h = &hidden_states[token_id * model_dim];
    float* o = &out[(token_id * topk + k) * model_dim];

    // Allocate shared memory for intermediate values
    extern __shared__ float sdata[];
    float* act_out = sdata;

    for (int i = threadIdx.x; i < inter_dim; i += blockDim.x) {
        float gate = 0.0f;
        float up = 0.0f;

        for (int m = 0; m < model_dim; ++m) {
            float h_val = h[m];
            float w1_gate = w1[expert_id * 2 * inter_dim * model_dim + i * model_dim + m];
            float w1_up   = w1[expert_id * 2 * inter_dim * model_dim + (inter_dim + i) * model_dim + m];
            gate += h_val * w1_gate;
            up   += h_val * w1_up;
        }

        act_out[i] = silu(gate) * up;
    }

    __syncthreads();

    for (int d = threadIdx.x; d < model_dim; d += blockDim.x) {
        float result = 0.0f;
        for (int i = 0; i < inter_dim; ++i) {
            float w = w2[expert_id * model_dim * inter_dim + d * inter_dim + i];
            result += act_out[i] * w;
        }
        o[d] = result;
    }
}
torch::Tensor launch_custom_moe(
    torch::Tensor hidden_states, torch::Tensor w1, torch::Tensor w2,
    torch::Tensor topk_weight, torch::Tensor topk_ids) {

    int token_num = hidden_states.size(0);
    int topk = topk_ids.size(1);
    int model_dim = hidden_states.size(1);
    int inter_dim = w2.size(2);  // w2: [expert, model_dim, inter_dim]

    auto out = torch::zeros({token_num, topk, model_dim}, hidden_states.options());

    for (int t = 0; t < token_num; ++t) {
        for (int k = 0; k < topk; ++k) {
            int E_id = topk_ids[t][k].item<int>();

            auto h_t = hidden_states[t];                      // [model_dim]
            auto w1_E = w1[E_id];                             // [2*inter_dim, model_dim]
            auto w2_E = w2[E_id];                             // [model_dim, inter_dim]

            auto gate_up = torch::matmul(w1_E, h_t);          // [2*inter_dim]
            auto gate = gate_up.slice(0, 0, inter_dim);
            auto up   = gate_up.slice(0, inter_dim, 2 * inter_dim);

            auto act = torch::silu(gate) * up;                // [inter_dim]
            auto final_out = torch::matmul(w2_E, act);        // [model_dim]

            out[t][k] = final_out;
        }
    }

    auto weighted_out = out * topk_weight.unsqueeze(-1);     // [token, topk, model_dim]
    return weighted_out.sum(1);                              // [token, model_dim]
}

PYBIND11_MODULE(custom_moe, m) {
    m.def("launch_custom_moe", &launch_custom_moe, "Optimized MoE kernel with HIP (no PyTorch ops)");
}
