#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <math.h>


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

    // m 和 l 现在都在 log2 空间或受其缩放影响
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

                // --- 极致优化区 ---
                // 直接计算基于 log2 的 score
                float score = dot * attention_scale; 

                float m_new = fmaxf(m_prev, score);
                
                // 使用硬件指令 __exp2f，输入已经是 log2 比例
                float alpha = exp2f(m_prev - m_new);
                float p = exp2f(score - m_new);

                // 使用 FMA (Fused Multiply-Add) 指令加速累加器更新
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
        // 使用 __frcp_rn (快速倒数指令) 进一步压榨性能
        float inv_l = __frcp_rn(l_prev);
        #pragma unroll
        for (int i = 0; i < cols_per_thread; i++) {
            int col = tx + i * 32;
            if (col < d) O[row * d + col] = r_acc[i] * inv_l;
        }
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, 
                      int M, int N, int d) 
{
    const int Br = 16; 
    const int Bc = 32;

    dim3 grid((M + Br - 1) / Br);
    dim3 block(32, Br); 

    size_t smem_size = (Bc * d + Bc * d) * sizeof(float);

    // 关键：预先将 log2(e) 乘入 scale，减少内核中每一步的计算量
    float attention_scale = (1.0f / sqrtf((float)d)) * 1.4426950408889634074f;

    flash_attn_ultra_kernel<16, 32, 128><<<grid, block, smem_size>>>(
        Q, K, V, output, M, N, d, attention_scale
    );
}