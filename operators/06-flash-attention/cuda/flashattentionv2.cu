#include <cuda_runtime.h>
#include <torch/extension.h>
#include <float.h>
#include <cmath>

// 宏定义保护，由 Python 端的 get_macros 注入
#ifndef BR
#define BR 32
#endif
#ifndef BC
#define BC 32
#endif

// 特征维度上限，用于寄存器数组分配
#ifndef MAX_D
#define MAX_D 128
#endif

// FlashAttention V2
// 关键变化：外层 Q 由 grid 并行（每个 block 独占 Br 行 Q）；
// kernel 内只剩 KV 内循环；O / l / m 全程驻寄存器；
// 不再每轮做 1/l 归一化，只在最后写回前做一次（延迟归一化）。
template<int Br, int Bc, int max_d>
__global__ void __launch_bounds__(Br * 32) flash_attn_v2_kernel(
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
    float* __restrict__ O,
    int M, int N, int d, float attention_scale)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.x * Br + ty;
    bool row_valid = (row < M);

    extern __shared__ float s_mem[];
    float* s_K = s_mem;            // [Bc, d]
    float* s_V = s_mem + Bc * d;   // [Bc, d]

    constexpr int cols_per_thread = max_d / 32;
    float r_q[cols_per_thread];
    float r_o[cols_per_thread];

    // 1. Q 行 + O 累加器一起驻寄存器
    #pragma unroll
    for (int k = 0; k < cols_per_thread; k++) {
        int col = tx + k * 32;
        r_q[k] = (row_valid && col < d) ? Q[row * d + col] : 0.0f;
        r_o[k] = 0.0f;
    }

    float m_i = -1e38f;
    float l_i = 0.0f;

    // 2. 内循环：KV 分块
    for (int j = 0; j < N; j += Bc) {
        int tid = ty * 32 + tx;
        int n_threads = Br * 32;

        // 协作加载 K_j, V_j
        for (int i = tid; i < Bc * d; i += n_threads) {
            int kv_row = j + (i / d);
            int kv_col = i % d;
            if (kv_row < N && kv_col < d) {
                s_K[i] = K[kv_row * d + kv_col];
                s_V[i] = V[kv_row * d + kv_col];
            } else {
                s_K[i] = 0.0f;
                s_V[i] = 0.0f;
            }
        }
        __syncthreads();

        if (row_valid) {
            for (int t = 0; t < Bc; t++) {
                if (j + t >= N) break;

                // Q_i · K_j[t]，warp 内规约
                float dot = 0.0f;
                #pragma unroll
                for (int k = 0; k < cols_per_thread; k++) {
                    dot += r_q[k] * s_K[t * d + (tx + k * 32)];
                }
                for (int mask = 16; mask > 0; mask >>= 1)
                    dot += __shfl_xor_sync(0xffffffff, dot, mask);

                // V2 在线 softmax 更新（延迟归一化：只乘 alpha，不除 l）
                float score = dot * attention_scale;
                float m_new = fmaxf(m_i, score);
                float alpha = expf(m_i - m_new);
                float p     = expf(score - m_new);

                l_i = l_i * alpha + p;
                #pragma unroll
                for (int k = 0; k < cols_per_thread; k++) {
                    r_o[k] = r_o[k] * alpha + p * s_V[t * d + (tx + k * 32)];
                }
                m_i = m_new;
            }
        }
        __syncthreads();
    }

    // 3. 最终归一化只做一次
    if (row_valid) {
        float inv_l = 1.0f / l_i;
        #pragma unroll
        for (int k = 0; k < cols_per_thread; k++) {
            int col = tx + k * 32;
            if (col < d) O[row * d + col] = r_o[k] * inv_l;
        }
    }
}

void solve(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
           int M, int N, int d)
{
    const float* d_Q = Q.data_ptr<float>();
    const float* d_K = K.data_ptr<float>();
    const float* d_V = V.data_ptr<float>();
    float* d_O = O.data_ptr<float>();

    dim3 grid((M + BR - 1) / BR);
    dim3 block(32, BR);

    size_t smem_size = (BC * d * 2) * sizeof(float);
    float scale = 1.0f / sqrtf((float)d);

    flash_attn_v2_kernel<BR, BC, MAX_D><<<grid, block, smem_size>>>(
        d_Q, d_K, d_V, d_O, M, N, d, scale);
}

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve", &solve, "Flash Attention V2 (Q-outer via grid, in-register accumulator)",
        py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("O"),
        py::arg("M"), py::arg("N"), py::arg("d"));
}
