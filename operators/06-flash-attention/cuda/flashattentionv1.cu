#include <cuda_runtime.h>
#include <torch/extension.h>
#include <float.h>
#include <cmath>

// 修正 1: 确保宏内部引用的变量名与 solve 函数的参数名一致
#define LAUNCH_FLASH_ATTN(BR, BC) \
    flash_attn_v1_kernel<BR, BC, 128><<<grid, block, smem_size>>>( \
        d_Q, d_K, d_V, d_O, d_L, d_M, M, N, d, scale);

template<int Br, int Bc, int max_d>
__global__ void flash_attn_v1_kernel(
    const float* Q, const float* K, const float* V, 
    float* O, float* L_data, float* M_data, 
    int M, int N, int d, float attention_scale) 
{
    int q_start = blockIdx.x * Br;
    int tx = threadIdx.x; 
    int ty = threadIdx.y; 
    int tid = ty * 32 + tx;

    extern __shared__ float s_mem[];
    float* s_K = s_mem;
    float* s_V = s_mem + Bc * d;

    int row = q_start + ty;
    bool row_valid = (row < M);

    float m_prev = -1e38f;
    float l_prev = 0.0f;
    
    constexpr int cols_per_thread = max_d / 32;
    float r_q[cols_per_thread];
    float r_acc[cols_per_thread];

    #pragma unroll
    for (int k = 0; k < cols_per_thread; k++) {
        int col = tx + k * 32;
        r_q[k] = (row_valid && col < d) ? Q[row * d + col] : 0.0f;
        r_acc[k] = 0.0f; 
    }

    for (int j = 0; j < N; j += Bc) {
        for (int i = tid; i < Bc * d; i += blockDim.x * blockDim.y) {
            int kv_row = j + (i / d);
            int kv_col = i % d;
            if (kv_row < N && kv_col < d) {
                s_K[i] = K[kv_row * d + kv_col];
                s_V[i] = V[kv_row * d + kv_col];
            } else {
                s_K[i] = 0.0f; s_V[i] = 0.0f;
            }
        }
        __syncthreads();

        if (row_valid) {
            for (int t = 0; t < Bc; t++) {
                if (j + t >= N) break;

                float dot = 0.0f;
                #pragma unroll
                for (int k = 0; k < cols_per_thread; k++) {
                    dot += r_q[k] * s_K[t * d + (tx + k * 32)];
                }
                for (int mask = 16; mask > 0; mask >>= 1)
                    dot += __shfl_xor_sync(0xffffffff, dot, mask);
                
                dot = __shfl_sync(0xffffffff, dot, 0); 

                float score = dot * attention_scale;
                float m_new = fmaxf(m_prev, score);
                
                // 使用 expf 对应标准 softmax
                float alpha = expf(m_prev - m_new);
                float p = expf(score - m_new);

                l_prev = l_prev * alpha + p;
                #pragma unroll
                for (int k = 0; k < cols_per_thread; k++) {
                    r_acc[k] = r_acc[k] * alpha + p * s_V[t * d + (tx + k * 32)];
                }
                m_prev = m_new;
            }
        }
        __syncthreads();
    }

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

void solve(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
           int M, int N, int d, int Br, int Bc) 
{
    const float* d_Q = Q.data_ptr<float>();
    const float* d_K = K.data_ptr<float>();
    const float* d_V = V.data_ptr<float>();
    float* d_O = O.data_ptr<float>();

    auto L = torch::zeros({M}, Q.options());
    auto Max = torch::full({M}, -1e38, Q.options());
    float* d_L = L.data_ptr<float>();
    float* d_M = Max.data_ptr<float>();

    dim3 grid((M + Br - 1) / Br);
    dim3 block(32, Br); 

    size_t smem_size = (Bc * d * 2) * sizeof(float);
    float scale = 1.0f / sqrtf((float)d);

    if (Br == 16 && Bc == 16) { LAUNCH_FLASH_ATTN(16, 16); }
    else if (Br == 16 && Bc == 32) { LAUNCH_FLASH_ATTN(16, 32); }
    else if (Br == 32 && Bc == 16) { LAUNCH_FLASH_ATTN(32, 16); }
    else if (Br == 32 && Bc == 32) { LAUNCH_FLASH_ATTN(32, 32); }
}

// 修正 2: 显式声明 py 别名
namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve", &solve, "Flash Attention Kernel",
        py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("O"),
        py::arg("M"), py::arg("N"), py::arg("d"), 
        py::arg("Br"), py::arg("Bc")); 
}