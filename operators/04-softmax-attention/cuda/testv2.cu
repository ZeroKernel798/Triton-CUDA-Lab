#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <torch/extension.h>
#include <float.h>
#include <math.h>

// -----------------------------------------------------------------------
// Flash Attention Ultra Kernel
// 保持你极致优化的计算逻辑不变
// -----------------------------------------------------------------------
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
    float r_acc[cols_per_thread];
    float r_q[cols_per_thread];
    
    #pragma unroll
    for (int i = 0; i < cols_per_thread; i++) {
        r_acc[i] = 0.0f;
        r_q[i] = 0.0f;
    }

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
                
                // 使用硬件指令 exp2f
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
void solve(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor output, 
           int M, int N, int d) 
{
    const int Br = 16; 
    const int Bc = 32;

    dim3 grid((M + Br - 1) / Br);
    dim3 block(32, Br); 

    // 动态计算共享内存
    size_t smem_size = (Bc * d + Bc * d) * sizeof(float);

    // 预缩放 scale 以便使用硬件 exp2f
    float attention_scale = (1.0f / sqrtf((float)d)) * 1.4426950408889634074f;

    // 启动内核，假设 d <= 128
    flash_attn_ultra_kernel<16, 32, 128><<<grid, block, smem_size>>>(
        Q.data_ptr<float>(), 
        K.data_ptr<float>(), 
        V.data_ptr<float>(), 
        output.data_ptr<float>(), 
        M, N, d, attention_scale
    );
}

// 模块定义
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve", &solve, "Flash Attention Ultra (CUDAized)");
}