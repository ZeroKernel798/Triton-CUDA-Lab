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

// 精度/特征维度上限，用于寄存器数组分配
#ifndef MAX_D
#define MAX_D 128
#endif

template<int Br, int Bc, int max_d>
__global__ void flash_attn_v1_kernel(
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V, 
    float* __restrict__ O, float* __restrict__ L_data, float* __restrict__ M_data, 
    int M, int N, int d, float attention_scale) 
{
    // 逻辑索引计算
    int q_start = blockIdx.x * Br;
    int tx = threadIdx.x; 
    int ty = threadIdx.y; 

    // 静态计算 Shared Memory 布局
    extern __shared__ float s_mem[];
    float* s_K = s_mem;            // 大小: Bc * d
    float* s_V = s_mem + Bc * d;   // 大小: Bc * d

    int row = q_start + ty;
    bool row_valid = (row < M);

    // 在线 Softmax 的中间变量 (Online Softmax)
    float m_prev = -1e38f;
    float l_prev = 0.0f;
    
    // 寄存器分配：每个线程负责处理 d 维度的一块 (通常是 d/32)
    constexpr int cols_per_thread = max_d / 32;
    float r_q[cols_per_thread];
    float r_acc[cols_per_thread];

    // 1. 加载 Q 到寄存器
    #pragma unroll
    for (int k = 0; k < cols_per_thread; k++) {
        int col = tx + k * 32;
        r_q[k] = (row_valid && col < d) ? Q[row * d + col] : 0.0f;
        r_acc[k] = 0.0f; 
    }

    // 2. 外层循环：遍历 KV 的 Blocks (BC)
    for (int j = 0; j < N; j += Bc) {
        // 协作搬运 K, V 到 Shared Memory
        #pragma unroll
        for (int i = ty * 32 + tx; i < Bc * d; i += blockDim.x * blockDim.y) {
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

        // 3. 计算 QK^T 并更新 Online Softmax 状态
        if (row_valid) {
            for (int t = 0; t < Bc; t++) {
                if (j + t >= N) break;

                // 计算点积 Score
                float dot = 0.0f;
                #pragma unroll
                for (int k = 0; k < cols_per_thread; k++) {
                    dot += r_q[k] * s_K[t * d + (tx + k * 32)];
                }
                // Warp Shuffle 规约点积结果
                for (int mask = 16; mask > 0; mask >>= 1)
                    dot += __shfl_xor_sync(0xffffffff, dot, mask);
                
                dot = __shfl_sync(0xffffffff, dot, 0); 

                float score = dot * attention_scale;
                
                // --- Online Softmax 核心逻辑 ---
                float m_new = fmaxf(m_prev, score);
                float alpha = expf(m_prev - m_new);
                float p = expf(score - m_new);

                l_prev = l_prev * alpha + p;

                // 更新结果累加器 (融合了 V 的乘法)
                #pragma unroll
                for (int k = 0; k < cols_per_thread; k++) {
                    r_acc[k] = r_acc[k] * alpha + p * s_V[t * d + (tx + k * 32)];
                }
                m_prev = m_new;
            }
        }
        __syncthreads();
    }

    // 4. 写回最终结果 O = r_acc / l_prev
    if (row_valid) {
        M_data[row] = m_prev;
        L_data[row] = l_prev;
        #pragma unroll
        for (int k = 0; k < cols_per_thread; k++) {
            int col = tx + k * 32;
            if (col < d) {
                O[row * d + col] = r_acc[k] / l_prev;
            }
        }
    }
}

// 移除 Br, Bc 参数
void solve(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
           int M, int N, int d) 
{
    const float* d_Q = Q.data_ptr<float>();
    const float* d_K = K.data_ptr<float>();
    const float* d_V = V.data_ptr<float>();
    float* d_O = O.data_ptr<float>();

    // 中间变量空间申请
    auto L = torch::zeros({M}, Q.options());
    auto Max = torch::full({M}, -1e38, Q.options());
    float* d_L = L.data_ptr<float>();
    float* d_M = Max.data_ptr<float>();

    // 编译时确定的线程块和网格布局
    dim3 grid((M + BR - 1) / BR);
    dim3 block(32, BR); 

    size_t smem_size = (BC * d * 2) * sizeof(float);
    float scale = 1.0f / sqrtf((float)d);

    // 直接实例化模板，消除 switch-case
    flash_attn_v1_kernel<BR, BC, MAX_D><<<grid, block, smem_size>>>(
        d_Q, d_K, d_V, d_O, d_L, d_M, M, N, d, scale);
}

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve", &solve, "Flash Attention Kernel (Macro Configured)",
        py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("O"),
        py::arg("M"), py::arg("N"), py::arg("d")); 
}