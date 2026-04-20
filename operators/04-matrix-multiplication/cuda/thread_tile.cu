#include <cuda_runtime.h>
#include <torch/extension.h>
#include <string>

#ifndef BLOCK_X
#define BLOCK_X 128
#endif
#ifndef BLOCK_Y
#define BLOCK_Y 128
#endif
#ifndef BLOCK_K
#define BLOCK_K 8
#endif

// 纯外积版本 
template<int BM, int BN, int BK>
__global__ void matrix_mul_kernel_outer_product_pure(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, 
                                                     int N, int M, int K) 
{
    // 共享内存：sA 存储转置后的结果 [BK, BM]，sB 存储原始布局 [BK, BN]
    __shared__ float sA[BK * BM];   
    __shared__ float sB[BK * BN];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 线程在 Block 内的扁平索引
    int idx = ty * blockDim.x + tx;
    int thread_nums = blockDim.x * blockDim.y;

    int row_start = blockIdx.y * BM;
    int col_start = blockIdx.x * BN;

    // 每个线程负责 8x8 的计算结果，存储在寄存器中
    float accum[8][8];
    #pragma unroll
    for(int i = 0; i < 8; ++i) {
        for(int j = 0; j < 8; ++j) {
            accum[i][j] = 0.0f;
        }
    }

    // 外层 K 循环
    for(int k_offset = 0; k_offset < K; k_offset += BK){
        
        // 1. 协作搬运 A 并转置写入 sA
        #pragma unroll
        for(int i = idx; i < BK * BM; i += thread_nums){
            int a_tile_r = i / BK;
            int a_tile_c = i % BK;
            float val = 0.0f;
            if((row_start + a_tile_r) < M && (k_offset + a_tile_c < K))
                val = A[(row_start + a_tile_r) * K + (k_offset + a_tile_c)];
            
            // 重点：转置写入，方便后续按行读取 K 维数据
            sA[a_tile_c * BM + a_tile_r] = val;
        }

        // 2. 协作搬运 B 写入 sB
        #pragma unroll
        for(int i = idx; i < BK * BN; i += thread_nums){
            int b_tile_r = i / BN;
            int b_tile_c = i % BN;
            float val = 0.0f;
            if((k_offset + b_tile_r) < K && (col_start + b_tile_c) < N)
                val = B[(k_offset + b_tile_r) * N + (col_start + b_tile_c)];

            sB[b_tile_r * BN + b_tile_c] = val;
        }

        __syncthreads();

        // 3. 计算阶段：基于寄存器的外积累加
        #pragma unroll
        for(int kk = 0; kk < BK; kk++){
            float reg_a[8];
            float reg_b[8];

            // 从 sA 读取负责的 8 个 A 元素（由于转置，现在是连续读取）
            #pragma unroll
            for(int i = 0; i < 8; ++i) reg_a[i] = sA[kk * BM + ty * 8 + i]; 
            
            // 从 sB 读取负责的 8 个 B 元素
            #pragma unroll
            for(int i = 0; i < 8; ++i) reg_b[i] = sB[kk * BN + tx * 8 + i]; 

            // 核心外积：8x8 累加
            #pragma unroll
            for(int i = 0; i < 8; ++i){
                #pragma unroll
                for(int j = 0; j < 8; ++j){
                    accum[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }
        __syncthreads();
    }

    // 4. 写回阶段
    #pragma unroll
    for(int i = 0; i < 8; ++i){
        #pragma unroll
        for(int j = 0; j < 8; ++j){
            int g_r = row_start + ty * 8 + i;
            int g_c = col_start + tx * 8 + j;
            if (g_r < M && g_c < N) {
                C[g_r * N + g_c] = accum[i][j];
            }
        }
    }
}

void solve(torch::Tensor A, torch::Tensor B, torch::Tensor C, int N, int M, int K) {
    auto d_A = A.data_ptr<float>(); 
    auto d_B = B.data_ptr<float>(); 
    auto d_C = C.data_ptr<float>();
    
    // 配置 Block 线程数：(BLOCK_X/8, BLOCK_Y/8)
    dim3 threads(BLOCK_X / 8, BLOCK_Y / 8); 
    dim3 blocks((N + BLOCK_X - 1) / BLOCK_X, (M + BLOCK_Y - 1) / BLOCK_Y);
    
    // 直接实例化模板，消除 if-else 分发
    matrix_mul_kernel_outer_product_pure<BLOCK_Y, BLOCK_X, BLOCK_K><<<blocks, threads>>>(d_A, d_B, d_C, N, M, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve", &solve, "Pure Outer Product GEMM (Macro Configured)",
          py::arg("A"), py::arg("B"), py::arg("C"), 
          py::arg("N"), py::arg("M"), py::arg("K")); 
}