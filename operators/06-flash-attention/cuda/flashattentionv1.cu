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

// FlashAttention V1
// 论文 Algorithm 1：外层 KV，内层 Q；O / l / m 每轮 KV 都从 HBM 读出、计算、写回。
// 体现 V1 的关键特征：HBM 流量较大，对照 V2 的「寄存器内累加」清晰可见。
template<int Br, int Bc, int max_d>
__global__ void __launch_bounds__(Br * 32) flash_attn_v1_kernel(
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
    float* __restrict__ O, float* __restrict__ L_data, float* __restrict__ M_data,
    int M, int N, int d, float attention_scale)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.x * Br + ty;
    bool row_valid = (row < M);

    // K, V 分块共享内存
    extern __shared__ float s_mem[];
    float* s_K = s_mem;            // [Bc, d]
    float* s_V = s_mem + Bc * d;   // [Bc, d]

    constexpr int cols_per_thread = max_d / 32;
    float r_q[cols_per_thread];

    // 1. Q 行驻寄存器（KV 外循环内不变）
    #pragma unroll
    for (int k = 0; k < cols_per_thread; k++) {
        int col = tx + k * 32;
        r_q[k] = (row_valid && col < d) ? Q[row * d + col] : 0.0f;
    }

    // 2. 外循环：KV 分块
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
            // V1 关键步骤：从 HBM 读回上一轮的 (l_i, m_i, O_i)
            float m_i = M_data[row];
            float l_i = L_data[row];

            float r_o[cols_per_thread];
            #pragma unroll
            for (int k = 0; k < cols_per_thread; k++) {
                int col = tx + k * 32;
                r_o[k] = (col < d) ? O[row * d + col] : 0.0f;
            }

            // Step 1: 当前 (i, j) tile 的 S_ij = Q_i K_j^T，找 m_tilde
            float s_row[Bc];
            float m_tilde = -1e38f;
            for (int t = 0; t < Bc; t++) {
                if (j + t >= N) { s_row[t] = -1e38f; continue; }
                float dot = 0.0f;
                #pragma unroll
                for (int k = 0; k < cols_per_thread; k++) {
                    dot += r_q[k] * s_K[t * d + (tx + k * 32)];
                }
                // warp 内规约 d 维点积
                for (int mask = 16; mask > 0; mask >>= 1)
                    dot += __shfl_xor_sync(0xffffffff, dot, mask);
                float score = dot * attention_scale;
                s_row[t] = score;
                m_tilde = fmaxf(m_tilde, score);
            }

            // Step 2: P_tilde = exp(S - m_tilde), l_tilde = rowsum
            float l_tilde = 0.0f;
            for (int t = 0; t < Bc; t++) {
                if (j + t >= N) { s_row[t] = 0.0f; continue; }
                s_row[t] = expf(s_row[t] - m_tilde);
                l_tilde += s_row[t];
            }

            // Step 3: 在线 softmax 合并新旧统计量
            float m_new = fmaxf(m_i, m_tilde);
            float alpha = expf(m_i - m_new);
            float beta  = expf(m_tilde - m_new);
            float l_new = alpha * l_i + beta * l_tilde;
            float inv_l_new = 1.0f / l_new;

            // Step 4: V1 归一化公式
            //   O_i ← (l_i * alpha * O_i_prev + beta * (P_tilde @ V_j)) / l_new
            for (int k = 0; k < cols_per_thread; k++) {
                int col = tx + k * 32;
                if (col >= d) continue;
                float pv = 0.0f;
                for (int t = 0; t < Bc; t++) {
                    pv += s_row[t] * s_V[t * d + col];
                }
                O[row * d + col] = (l_i * alpha * r_o[k] + beta * pv) * inv_l_new;
            }

            // Step 5: 写回 l, m
            L_data[row] = l_new;
            M_data[row] = m_new;
        }
        __syncthreads();
    }
}

void solve(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
           int M, int N, int d)
{
    const float* d_Q = Q.data_ptr<float>();
    const float* d_K = K.data_ptr<float>();
    const float* d_V = V.data_ptr<float>();
    float* d_O = O.data_ptr<float>();

    // V1 要求 O 初始化为 0：每轮 KV 都会读 O_prev 做加权
    O.zero_();
    auto L = torch::zeros({M}, Q.options());
    auto Max = torch::full({M}, -1e38f, Q.options());
    float* d_L = L.data_ptr<float>();
    float* d_M = Max.data_ptr<float>();

    dim3 grid((M + BR - 1) / BR);
    dim3 block(32, BR);

    size_t smem_size = (BC * d * 2) * sizeof(float);
    float scale = 1.0f / sqrtf((float)d);

    flash_attn_v1_kernel<BR, BC, MAX_D><<<grid, block, smem_size>>>(
        d_Q, d_K, d_V, d_O, d_L, d_M, M, N, d, scale);
}

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve", &solve, "Flash Attention V1 (KV-outer, HBM round-trip)",
        py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("O"),
        py::arg("M"), py::arg("N"), py::arg("d"));
}
