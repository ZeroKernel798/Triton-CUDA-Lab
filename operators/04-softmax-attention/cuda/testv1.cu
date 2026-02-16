#include <cuda_runtime.h>
#include <torch/extension.h>
#include <float.h>
#include <math.h>

// -----------------------------------------------------------------------
// Flash Attention Kernel
// 保持你的逻辑不变，仅将 max_d 设为模板参数以优化寄存器分配
// -----------------------------------------------------------------------
template<int Br, int Bc, int max_d>
__global__ void flash_attn_kernel(
    const float* Q, const float* K, const float* V, float* O,
    int M, int N, int d, float scale) 
{
    extern __shared__ float s_mem[];
    float* s_Q = s_mem;                  
    float* s_K = s_mem + Br * d;         
    float* s_V = s_mem + (Br + Bc) * d;  

    int tid = threadIdx.x; 
    int bid = blockIdx.x;
    int q_row_start = bid * Br;
    int row = q_row_start + tid;

    float m_prev = -FLT_MAX;
    float l_prev = 0.0f;
    float acc[max_d]; 
    #pragma unroll
    for (int k = 0; k < max_d; k++) acc[k] = 0.0f;

    // 协作加载 Q
    for (int i = tid; i < Br * d; i += Br) {
        int r = i / d;
        int c = i % d;
        if (q_row_start + r < M && c < d)
            s_Q[r * d + c] = Q[(q_row_start + r) * d + c];
        else
            s_Q[r * d + c] = 0.0f;
    }
    __syncthreads();

    for (int j = 0; j < N; j += Bc) {
        // 协作加载 K 和 V
        for (int i = tid; i < Bc * d; i += Br) {
            int r = i / d;
            int c = i % d;
            if (j + r < N && c < d) {
                s_K[r * d + c] = K[(j + r) * d + c];
                s_V[r * d + c] = V[(j + r) * d + c];
            } else {
                s_K[r * d + c] = 0.0f;
                s_V[r * d + c] = 0.0f;
            }
        }
        __syncthreads();

        if (row < M) {
            for (int t = 0; t < Bc; t++) {
                if (j + t >= N) break; 

                float score = 0.0f;
                for (int k = 0; k < d; k++) {
                    score += s_Q[tid * d + k] * s_K[t * d + k];
                }
                score *= scale;

                // Online Softmax 逻辑
                float m_new = fmaxf(m_prev, score);
                float p = __expf(score - m_new);
                float alpha = __expf(m_prev - m_new);

                l_prev = l_prev * alpha + p;

                #pragma unroll
                for (int k = 0; k < max_d; k++) {
                    if (k < d) acc[k] = acc[k] * alpha + p * s_V[t * d + k];
                }
                m_prev = m_new;
            }
        }
        __syncthreads();
    }

    if (row < M) {
        for (int k = 0; k < d; k++) {
            O[row * d + k] = acc[k] / l_prev;
        }
    }
}

// -----------------------------------------------------------------------
// Pybind11 接口函数
// -----------------------------------------------------------------------
void solve(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
           int M, int N, int d) 
{
    // 获取设备指针
    const float* d_Q = Q.data_ptr<float>();
    const float* d_K = K.data_ptr<float>();
    const float* d_V = V.data_ptr<float>();
    float* d_O = O.data_ptr<float>();

    const int Br = 32; 
    const int Bc = 32; 

    dim3 grid((M + Br - 1) / Br);
    dim3 block(Br); 

    // 动态计算共享内存：(Br*d + Bc*d + Bc*d) * sizeof(float)
    size_t smem_size = (Br * d + Bc * d + Bc * d) * sizeof(float);
    float scale = 1.0f / sqrtf((float)d);

    // 针对 d <= 128 的情况启动内核
    flash_attn_kernel<Br, Bc, 128><<<grid, block, smem_size>>>(
        d_Q, d_K, d_V, d_O, M, N, d, scale
    );
}

// 模块定义
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve", &solve, "Flash Attention Kernel (CUDA)");
}