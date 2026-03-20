#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <torch/extension.h>
#include <float.h>
#include <math.h>

// 宏定义保护，由 Python 端的 get_macros 注入
#ifndef BR
#define BR 32
#endif
#ifndef BC
#define BC 32
#endif
#ifndef MAX_D
#define MAX_D 128
#endif

template<int Br, int Bc, int max_d>
__global__ void flash_attn_ultra_kernel(
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V, 
    float* __restrict__ O,
    int M, int N, int d, float attention_scale) 
{
    // 逻辑索引计算
    int tx = threadIdx.x; 
    int ty = threadIdx.y;
    int row = blockIdx.x * Br + ty;

    // 静态计算 Shared Memory 布局
    extern __shared__ float s_mem[];
    float* s_K = s_mem;               
    float* s_V = s_mem + Bc * d;      

    // 寄存器数组静态分配
    constexpr int cols_per_thread = max_d / 32;
    float r_acc[cols_per_thread];
    float r_q[cols_per_thread];
    
    #pragma unroll
    for (int i = 0; i < cols_per_thread; i++) {
        r_acc[i] = 0.0f;
        r_q[i] = 0.0f;
    }
    
    float m_prev = -FLT_MAX;
    float l_prev = 0.0f;    

    // 1. 加载 Q 到寄存器
    if (row < M) {
        #pragma unroll
        for (int i = 0; i < cols_per_thread; i++) {
            int col = tx + i * 32;
            if (col < d) r_q[i] = Q[row * d + col];
        }
    }

    // 2. 外层循环：遍历 KV 的 Blocks
    for (int j = 0; j < N; j += Bc) {
        int tid_in_block = ty * 32 + tx;
        int threads_per_block = Br * 32;
        
        // 协作搬运 K, V 到 Shared Memory
        #pragma unroll
        for (int i = tid_in_block; i < Bc * d; i += threads_per_block) {
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

        // 3. 计算 Block 内的 Attention
        if (row < M) {
            for (int t = 0; t < Bc; t++) {
                if (j + t >= N) break;

                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < cols_per_thread; i++) {
                    dot += r_q[i] * s_K[t * d + (tx + i * 32)];
                }
                
                // Warp 内规约点积结果
                for (int mask = 16; mask > 0; mask >>= 1) {
                    dot += __shfl_xor_sync(0xffffffff, dot, mask);
                }

                // --- Flash Attention V2 核心更新逻辑 ---
                float score = dot * attention_scale; 
                float m_new = fmaxf(m_prev, score);
                
                // 使用 exp2f 加速，注意 attention_scale 在 solve 中已包含 log2(e)
                float alpha = exp2f(m_prev - m_new);
                float p = exp2f(score - m_new);
                
                // 使用硬件级 FMA 指令
                l_prev = __fmaf_rn(l_prev, alpha, p);

                #pragma unroll
                for (int i = 0; i < cols_per_thread; i++) {
                    r_acc[i] = __fmaf_rn(r_acc[i], alpha, p * s_V[t * d + (tx + i * 32)]);
                }
                m_prev = m_new;
            }
        }
        __syncthreads();
    }

    // 4. 最终归一化并写回 O
    if (row < M) {
        // 使用快速倒数指令
        float inv_l = __frcp_rn(l_prev);
        #pragma unroll
        for (int i = 0; i < cols_per_thread; i++) {
            int col = tx + i * 32;
            if (col < d) O[row * d + col] = r_acc[i] * inv_l;
        }
    }
}

// 移除运行时 Br, Bc 参数
void solve(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O, 
           int M, int N, int d) 
{
    const float* d_Q = Q.data_ptr<float>();
    const float* d_K = K.data_ptr<float>();
    const float* d_V = V.data_ptr<float>();
    float* d_O = O.data_ptr<float>();

    // 编译时确定的布局
    dim3 grid((M + BR - 1) / BR);
    dim3 block(32, BR); 

    size_t smem_size = (BC * d * 2) * sizeof(float);

    // 1.4426... 是 log2(e)，预乘以 scale 可以让内核直接调用快速的 exp2f
    float attention_scale = (1.0f / sqrtf((float)d)) * 1.4426950408889634f;

    // 直接启动模板，消除 switch-case 分发
    flash_attn_ultra_kernel<BR, BC, MAX_D><<<grid, block, smem_size>>>(
        d_Q, d_K, d_V, d_O, M, N, d, attention_scale);
}

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve", &solve, "Flash Attention Ultra Kernel (V2 Opt)",
        py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("O"),
        py::arg("M"), py::arg("N"), py::arg("d"));
}