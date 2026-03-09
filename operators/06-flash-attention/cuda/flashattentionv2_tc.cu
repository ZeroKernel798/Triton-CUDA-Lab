#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <mma.h>
#include <torch/extension.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define LOG2E 1.4426950408889634f

#define LAUNCH_FLASH_HALF(BR, BC) \
    flash_attn_wmma_kernel<BR, BC, 128><<<grid, block, smem_size>>>( \
        d_Q, d_K, d_V, d_O, M, N, d, attention_scale);

template<int Br, int Bc, int max_d>
__global__ void flash_attn_wmma_kernel(
    const __half* Q, const __half* K, const __half* V, __half* O,
    int M, int N, int d, float attention_scale) 
{
    int warp_id = threadIdx.y; 
    int tid = threadIdx.x; // warp 内的线程 ID (0-31)
    int row_start = blockIdx.x * Br + warp_id * 16;

    // --- 修正：统一共享内存管理 ---
    extern __shared__ char shared_buf[];
    __half* s_K = (__half*)shared_buf;               
    __half* s_V = (__half*)(shared_buf + Bc * d * sizeof(__half));      
    
    // 为每个 Warp 分配 16x16 的 float 空间，用于安全地缩放 out_frag
    float* s_out = (float*)(shared_buf + 2 * Bc * d * sizeof(__half));
    float* s_out_warp = s_out + warp_id * 16 * 16;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> q_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> k_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> v_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> out_frag;

    float m_prev[16];
    float l_prev[16];
    #pragma unroll
    for(int i=0; i<16; ++i) { m_prev[i] = -1e38f; l_prev[i] = 0.0f; }

    wmma::fill_fragment(out_frag, 0.0f);

    if (row_start < M) {
        wmma::load_matrix_sync(q_frag, Q + row_start * d, d);
    }

    for (int j = 0; j < N; j += Bc) {
        // 协作加载 K, V
        int linear_tid = threadIdx.y * 32 + threadIdx.x;
        int num_threads = blockDim.y * 32;
        for (int i = linear_tid; i < Bc * d; i += num_threads) {
            int kv_row = j + (i / d);
            if (kv_row < N) {
                s_K[i] = K[kv_row * d + (i % d)];
                s_V[i] = V[kv_row * d + (i % d)];
            } else {
                s_K[i] = (__half)0.0f; 
                s_V[i] = (__half)0.0f;
            }
        }
        __syncthreads();

        for (int step = 0; step < Bc; step += 16) {
            wmma::fill_fragment(acc_frag, 0.0f);
            wmma::load_matrix_sync(k_frag, s_K + step * d, d);
            wmma::mma_sync(acc_frag, q_frag, k_frag, acc_frag);

            // --- 强制内存对齐，防止 load_matrix 崩溃 ---
            __align__(16) float temp_scores[16][16];
            wmma::store_matrix_sync(&temp_scores[0][0], acc_frag, 16, wmma::mem_row_major);

            __align__(16) __half p_half[16][16];
            float alpha[16]; 

            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                float m_block = -1e38f;
                #pragma unroll
                for (int k = 0; k < 16; ++k) {
                    temp_scores[i][k] *= attention_scale;
                    m_block = fmaxf(m_block, temp_scores[i][k]);
                }

                float m_new = fmaxf(m_prev[i], m_block);
                alpha[i] = exp2f((m_prev[i] - m_new) * LOG2E);
                
                float row_sum = 0.0f;
                #pragma unroll
                for (int k = 0; k < 16; ++k) {
                    float p = exp2f((temp_scores[i][k] - m_new) * LOG2E);
                    p_half[i][k] = __float2half(p);
                    row_sum += p;
                }

                l_prev[i] = l_prev[i] * alpha[i] + row_sum;
                m_prev[i] = m_new;
            }

            // --- 核心修复：基于 Shared Memory 按行缩放 out_frag ---
            wmma::store_matrix_sync(s_out_warp, out_frag, 16, wmma::mem_row_major);
            
            // Warp 内 32 个线程协同缩放，每个线程缩放 8 个元素
            int row = tid / 2;             
            int col_start = (tid % 2) * 8; 
            
            #pragma unroll
            for (int c = 0; c < 8; ++c) {
                s_out_warp[row * 16 + col_start + c] *= alpha[row];
            }
            __syncwarp(); 
            
            wmma::load_matrix_sync(out_frag, s_out_warp, 16, wmma::mem_row_major);
            // --------------------------------------------------------

            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> p_frag;
            wmma::load_matrix_sync(p_frag, &p_half[0][0], 16);
            wmma::load_matrix_sync(v_frag, s_V + step * d, d);
            wmma::mma_sync(out_frag, p_frag, v_frag, out_frag);
        }
        __syncthreads();
    }

    if (row_start < M) {
        __align__(16) float res[16][16];
        wmma::store_matrix_sync(&res[0][0], out_frag, 16, wmma::mem_row_major);
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            float inv_l = 1.0f / l_prev[i];
            for (int k = 0; k < 16; ++k) {
                if (row_start + i < M && k < d)
                    O[(row_start + i) * d + k] = __float2half(res[i][k] * inv_l);
            }
        }
    }
}

void solve(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O, 
           int M, int N, int d, int Br, int Bc) 
{
    auto Q_half = Q.to(at::kHalf).contiguous();
    auto K_half = K.to(at::kHalf).contiguous();
    auto V_half = V.to(at::kHalf).contiguous();
    auto O_half = torch::empty_like(Q_half);

    const __half* d_Q = (const __half*)Q_half.data_ptr<at::Half>();
    const __half* d_K = (const __half*)K_half.data_ptr<at::Half>();
    const __half* d_V = (const __half*)V_half.data_ptr<at::Half>();
    __half* d_O = (__half*)O_half.data_ptr<at::Half>();

    dim3 grid((M + Br - 1) / Br);
    dim3 block(32, Br / 16); 

    // --- 修复：精确计算所需的 smem 大小 ---
    // 包含: s_K 的大小 + s_V 的大小 + 用于缩放的 s_out 大小 (每个 Warp 16x16 个 float)
    int num_warps = Br / 16;
    size_t smem_size = (Bc * d * 2 * sizeof(__half)) + (num_warps * 16 * 16 * sizeof(float)); 
    
    float attention_scale = 1.0f / sqrtf((float)d);

    if (Br == 16 && Bc == 16) { LAUNCH_FLASH_HALF(16, 16); }
    else if (Br == 32 && Bc == 32) { LAUNCH_FLASH_HALF(32, 32); }
    else { LAUNCH_FLASH_HALF(16, 16); }

    O.copy_(O_half); 
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve", &solve, "Flash Attention FP16 Tensor Core",
        py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("O"),
        py::arg("M"), py::arg("N"), py::arg("d"), 
        py::arg("Br"), py::arg("Bc"));
}