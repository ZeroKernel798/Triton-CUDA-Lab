#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <torch/extension.h>
#include <float.h>
#include <math.h>

// 确保参数名与 solve 函数内部定义的变量名一致
#define LAUNCH_FLASH_ATTN(BR, BC) \
    flash_attn_ultra_kernel<BR, BC, 128><<<grid, block, smem_size>>>( \
        d_Q, d_K, d_V, d_O, M, N, d, attention_scale);


template<int Br, int Bc, int max_d>
__global__ void flash_attn_ultra_kernel(
    const float* Q, const float* K, const float* V, float* O,
    int M, int N, int d, float attention_scale) 
{
    int tx = threadIdx.x; 
    int ty = threadIdx.y;
    int row = blockIdx.x * Br + ty;

    extern __shared__ float s_mem[];
    float* s_K = s_mem;               
    float* s_V = s_mem + Bc * d;      

    constexpr int cols_per_thread = max_d / 32;
    float r_acc[cols_per_thread] = {0.0f};
    float r_q[cols_per_thread] = {0.0f};
    
    float m_prev = -FLT_MAX;
    float l_prev = 0.0f;    

    if (row < M) {
        #pragma unroll
        for (int i = 0; i < cols_per_thread; i++) {
            int col = tx + i * 32;
            if (col < d) r_q[i] = Q[row * d + col];
        }
    }

    for (int j = 0; j < N; j += Bc) {
        int tid_in_block = ty * 32 + tx;
        int threads_per_block = Br * 32;
        for (int i = tid_in_block; i < Bc * d; i += threads_per_block) {
            if (j + (i / d) < N) {
                s_K[i] = K[(j + (i / d)) * d + (i % d)];
                s_V[i] = V[(j + (i / d)) * d + (i % d)];
            } else {
                s_K[i] = 0.0f; s_V[i] = 0.0f;
            }
        }
        __syncthreads();

        if (row < M) {
            for (int t = 0; t < Bc; t++) {
                if (j + t >= N) break;

                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < cols_per_thread; i++) {
                    dot += r_q[i] * s_K[t * d + (tx + i * 32)];
                }
                
                for (int mask = 16; mask > 0; mask >>= 1) {
                    dot += __shfl_xor_sync(0xffffffff, dot, mask);
                }

                float score = dot * attention_scale; 
                float m_new = fmaxf(m_prev, score);
                float alpha = exp2f(m_prev - m_new);
                float p = exp2f(score - m_new);
                
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

    if (row < M) {
        float inv_l = __frcp_rn(l_prev);
        #pragma unroll
        for (int i = 0; i < cols_per_thread; i++) {
            int col = tx + i * 32;
            if (col < d) O[row * d + col] = r_acc[i] * inv_l;
        }
    }
}

// -----------------------------------------------------------------------
// Pybind11 接口函数
// -----------------------------------------------------------------------
void solve(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O, 
           int M, int N, int d, int Br, int Bc) 
{
    // 显式提取指针，供宏 LAUNCH_FLASH_ATTN 使用
    const float* d_Q = Q.data_ptr<float>();
    const float* d_K = K.data_ptr<float>();
    const float* d_V = V.data_ptr<float>();
    float* d_O = O.data_ptr<float>();

    dim3 grid((M + Br - 1) / Br);
    dim3 block(32, Br); 

    size_t smem_size = (Bc * d * 2) * sizeof(float);

    // 预缩放常数，用于 exp2f
    float attention_scale = (1.0f / sqrtf((float)d)) * 1.4426950408889634f;

    // 模板分发
    if (Br == 16 && Bc == 16) {
        LAUNCH_FLASH_ATTN(16, 16);
    } else if (Br == 16 && Bc == 32) {
        LAUNCH_FLASH_ATTN(16, 32);
    } else if (Br == 32 && Bc == 16) {
        LAUNCH_FLASH_ATTN(32, 16);
    } else if (Br == 32 && Bc == 32) {
        LAUNCH_FLASH_ATTN(32, 32);
    } else {
        // 如果没有匹配的模板，这里可以加一个默认启动或报错
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve", &solve, "Flash Attention Kernel V2 opt",
        py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("O"),
        py::arg("M"), py::arg("N"), py::arg("d"), 
        py::arg("Br"), py::arg("Bc"));
}