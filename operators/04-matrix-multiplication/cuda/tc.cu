#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>
#include <torch/extension.h>

using namespace nvcuda;

// =====================================================================
// 🚀 核心 Kernel：Ampere Asynchronous Copy + Tensor Core WMMA
// =====================================================================
template<int BM, int BN, int BK, int WPT_M, int WPT_N>
__global__ void wmma_gemm_cp_async_kernel(const __half* __restrict__ A, 
                                          const __half* __restrict__ B, 
                                          float* __restrict__ C, 
                                          int N, int M, int K) 
{
    // Shared Memory 布局：双缓冲 (Double Buffering)
    extern __shared__ __half s_mem[];
    const int buffer_size = BM * BK + BK * BN;
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;
    int num_warps = blockDim.y * blockDim.x / 32;

    // Warp 切分逻辑：决定当前 Warp 负责的 64x32 区域
    int warp_row = warp_id / (BN / WPT_N); 
    int warp_col = warp_id % (BN / WPT_N);

    int row_start = blockIdx.y * BM;
    int col_start = blockIdx.x * BN;

    // WMMA 参数计算：每个 Warp 负责几个 16x16 的 Tensor Core 矩阵块
    constexpr int WMMA_M = WPT_M / 16;
    constexpr int WMMA_N = WPT_N / 16;
    constexpr int WMMA_K = BK / 16; 

    // 🚀 核心：定义 Tensor Core 片段 (FP16 输入，FP32 累加输出！)
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag[WMMA_M];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag[WMMA_N];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[WMMA_M][WMMA_N];

    // 初始化 C 矩阵累加器
    #pragma unroll
    for (int i = 0; i < WMMA_M; i++) {
        #pragma unroll
        for (int j = 0; j < WMMA_N; j++) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    int write_idx = 0;
    int read_idx = 0;
    int num_threads = blockDim.x * blockDim.y;

    // ===================================================================
    // 🚚 Prologue: 第一波异步直通车 (A 和 B 都直接从 Global 飞入 Shared)
    // ===================================================================
    __half* sA_write = s_mem + write_idx * buffer_size;
    __half* sB_write = s_mem + write_idx * buffer_size + BM * BK;

    // 加载 A (每次读取 16 bytes = 8 个 half)
    for (int i = tid * 8; i < BM * BK; i += num_threads * 8) {
        int r = i / BK; int c = i % BK;
        const __half* g_ptr = &A[(row_start + r) * K + c];
        __half* s_ptr = &sA_write[r * BK + c];
        if (row_start + r < M && c < K) {
            __pipeline_memcpy_async(s_ptr, g_ptr, 16);
        } else {
            for(int k=0; k<8; k++) s_ptr[k] = __float2half(0.0f); // 越界补零
        }
    }

    // 加载 B
    for (int i = tid * 8; i < BK * BN; i += num_threads * 8) {
        int r = i / BN; int c = i % BN;
        const __half* g_ptr = &B[r * N + (col_start + c)];
        __half* s_ptr = &sB_write[r * BN + c];
        if (r < K && col_start + c < N) {
            __pipeline_memcpy_async(s_ptr, g_ptr, 16);
        } else {
            for(int k=0; k<8; k++) s_ptr[k] = __float2half(0.0f);
        }
    }
    __pipeline_commit();
    __pipeline_wait_prior(0); // 阻塞等待第一波数据抵达
    __syncthreads();

    // ===================================================================
    // ⚙️ Main Pipeline: 边算边等
    // ===================================================================
    for (int k_step = BK; k_step < K; k_step += BK) {
        write_idx ^= 1; // 切换缓冲
        sA_write = s_mem + write_idx * buffer_size;
        sB_write = s_mem + write_idx * buffer_size + BM * BK;

        // 1. 发射下一波直通车
        for (int i = tid * 8; i < BM * BK; i += num_threads * 8) {
            int r = i / BK; int c = i % BK;
            const __half* g_ptr = &A[(row_start + r) * K + (k_step + c)];
            __half* s_ptr = &sA_write[r * BK + c];
            if (row_start + r < M && k_step + c < K) {
                __pipeline_memcpy_async(s_ptr, g_ptr, 16);
            } else {
                for(int k=0; k<8; k++) s_ptr[k] = __float2half(0.0f);
            }
        }
        for (int i = tid * 8; i < BK * BN; i += num_threads * 8) {
            int r = i / BN; int c = i % BN;
            const __half* g_ptr = &B[(k_step + r) * N + (col_start + c)];
            __half* s_ptr = &sB_write[r * BN + c];
            if (k_step + r < K && col_start + c < N) {
                __pipeline_memcpy_async(s_ptr, g_ptr, 16);
            } else {
                for(int k=0; k<8; k++) s_ptr[k] = __float2half(0.0f);
            }
        }
        __pipeline_commit();

        // 2. 用上一波抵达的数据执行 Tensor Core 计算
        __half* sA_read = s_mem + read_idx * buffer_size;
        __half* sB_read = s_mem + read_idx * buffer_size + BM * BK;

        for (int k = 0; k < WMMA_K; k++) {
            #pragma unroll
            for (int i = 0; i < WMMA_M; i++) {
                int a_row = warp_row * WPT_M + i * 16;
                // Tensor Core 直接从 Row-Major 共享内存取数！
                wmma::load_matrix_sync(a_frag[i], &sA_read[a_row * BK + k * 16], BK);
            }
            #pragma unroll
            for (int j = 0; j < WMMA_N; j++) {
                int b_col = warp_col * WPT_N + j * 16;
                wmma::load_matrix_sync(b_frag[j], &sB_read[(k * 16) * BN + b_col], BN);
            }

            // 🚀 Tensor Core 矩阵乘加 MMA
            #pragma unroll
            for (int i = 0; i < WMMA_M; i++) {
                #pragma unroll
                for (int j = 0; j < WMMA_N; j++) {
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }
        }

        __pipeline_wait_prior(0); // 必须等下一波数据抵达
        __syncthreads();
        read_idx ^= 1;
    }

    // ===================================================================
    // 🏁 Epilogue: 消耗最后一波数据并安全写回
    // ===================================================================
    __half* sA_read = s_mem + read_idx * buffer_size;
    __half* sB_read = s_mem + read_idx * buffer_size + BM * BK;

    for (int k = 0; k < WMMA_K; k++) {
        #pragma unroll
        for (int i = 0; i < WMMA_M; i++) {
            int a_row = warp_row * WPT_M + i * 16;
            wmma::load_matrix_sync(a_frag[i], &sA_read[a_row * BK + k * 16], BK);
        }
        #pragma unroll
        for (int j = 0; j < WMMA_N; j++) {
            int b_col = warp_col * WPT_N + j * 16;
            wmma::load_matrix_sync(b_frag[j], &sB_read[(k * 16) * BN + b_col], BN);
        }
        #pragma unroll
        for (int i = 0; i < WMMA_M; i++) {
            #pragma unroll
            for (int j = 0; j < WMMA_N; j++) {
                wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
            }
        }
    }

    // 将碎片数据写回 Global Memory (带边界保护，写回的是 Float32)
    #pragma unroll
    for (int i = 0; i < WMMA_M; i++) {
        #pragma unroll
        for (int j = 0; j < WMMA_N; j++) {
            float temp_c[16][16];
            wmma::store_matrix_sync(&temp_c[0][0], c_frag[i][j], 16, wmma::mem_row_major);
            
            int base_r = row_start + warp_row * WPT_M + i * 16;
            int base_c = col_start + warp_col * WPT_N + j * 16;
            
            for(int tr = 0; tr < 16; tr++) {
                for(int tc = 0; tc < 16; tc++) {
                    if (base_r + tr < M && base_c + tc < N) {
                        C[(base_r + tr) * N + (base_c + tc)] = temp_c[tr][tc];
                    }
                }
            }
        }
    }
}

// =====================================================================
// 📦 外部封装：不改变 Python 接口
// =====================================================================
void solve(torch::Tensor A, torch::Tensor B, torch::Tensor C, 
           int N, int M, int K, int bx, int by, int bk) {
    
    // 💡 自动转型：向 Tensor Core 喂入 FP16，但保留 C 的 FP32 精确度
    auto A_half = A.to(at::kHalf).contiguous();
    auto B_half = B.to(at::kHalf).contiguous();
    
    auto d_A = (const __half*)A_half.data_ptr<at::Half>();
    auto d_B = (const __half*)B_half.data_ptr<at::Half>();
    auto d_C = C.data_ptr<float>(); // 直接修改传入的 C (Float) Tensor
    
    // WMMA 的 BK 必须是 16 的倍数，如果传入过小，强行修正
    int valid_bk = bk < 16 ? 16 : bk; 
    
    // 使用 8 个 Warp (256 线程) 
    dim3 threads(32, 8); 
    dim3 blocks((N + bx - 1) / bx, (M + by - 1) / by);
    
    // 动态显存大小 = 2 缓冲 * (A 和 B 的碎片) * FP16 字节
    size_t shared_mem_size = 2 * (by * valid_bk + valid_bk * bx) * sizeof(__half);

    // 根据输入的 Block 大小启动对应的模板
    if (bx == 128 && by == 128) {
        // BM=128, BN=128, BK=32, WPT_M=64, WPT_N=32
        wmma_gemm_cp_async_kernel<128, 128, 32, 64, 32><<<blocks, threads, shared_mem_size>>>(d_A, d_B, d_C, N, M, K);
    } else if (bx == 128 && by == 64) {
        // BM=64, BN=128, BK=32, WPT_M=32, WPT_N=32
        wmma_gemm_cp_async_kernel<64, 128, 32, 32, 32><<<blocks, threads, shared_mem_size>>>(d_A, d_B, d_C, N, M, K);
    } else {
        // 默认保底策略：64x64 Block
        wmma_gemm_cp_async_kernel<64, 64, 16, 32, 16><<<blocks, threads, shared_mem_size>>>(d_A, d_B, d_C, N, M, K);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve", &solve, "CP_ASYNC + Tensor Core Hybrid GEMM",
          py::arg("A"), py::arg("B"), py::arg("C"), 
          py::arg("N"), py::arg("M"), py::arg("K"), 
          py::arg("bx"), py::arg("by"), py::arg("bk")); 
}