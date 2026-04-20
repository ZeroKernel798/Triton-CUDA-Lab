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

// 判断是否对齐 帮助实现float4向量化读取
#define IS_ALIGNED_16(ptr) ((reinterpret_cast<size_t>(ptr) & 15) == 0)

// Swizzle 索引计算 用于规避银行冲突 (通过对列索引进行异或操作)
#define GET_S_INDEX(row, col, width) (((row) * (width)) + (((col) / 4 ^ (row)) * 4) + ((col) % 4))

template<int BM, int BN, int BK>
__global__ void matrix_mul_kernel_outer_product(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, 
                                               int N, int M, int K) 
{
    // 编译器现在可以静态计算共享内存大小
    __shared__ float sA[BK * BM];
    __shared__ float sB[BK * BN];

    int tx = threadIdx.x; 
    int ty = threadIdx.y; 
    int tid = ty * blockDim.x + tx;
    int num_threads = blockDim.x * blockDim.y; 

    int row_start = blockIdx.y * BM;
    int col_start = blockIdx.x * BN;

    // 寄存器累加器：每个线程负责 8x8 的输出块
    float4 accum[8][2]; 
    #pragma unroll
    for(int i = 0; i < 8; i++) {
        accum[i][0] = {0.0f, 0.0f, 0.0f, 0.0f};
        accum[i][1] = {0.0f, 0.0f, 0.0f, 0.0f};
    }

    // 外层 K 循环 (以 BK 为步长)
    for (int k_ptr = 0; k_ptr < K; k_ptr += BK) {
        
        // 1. 协作搬运 A 到 sA (转置存储以优化后续外积读取)
        #pragma unroll
        for (int p = 0; p < (BM * BK + num_threads * 4 - 1) / (num_threads * 4); p++) {
            int idx = (tid + p * num_threads) * 4;
            if (idx < BM * BK) {
                int a_tile_row = idx / BK; 
                int a_tile_col = idx % BK; 
                const float* g_ptr = &A[(row_start + a_tile_row) * K + (k_ptr + a_tile_col)];
                
                if (row_start + a_tile_row < M && (k_ptr + a_tile_col + 3) < K && IS_ALIGNED_16(g_ptr)) {
                    float4 tmp = reinterpret_cast<const float4*>(g_ptr)[0];
                    sA[GET_S_INDEX(a_tile_col + 0, a_tile_row, BM)] = tmp.x;
                    sA[GET_S_INDEX(a_tile_col + 1, a_tile_row, BM)] = tmp.y;
                    sA[GET_S_INDEX(a_tile_col + 2, a_tile_row, BM)] = tmp.z;
                    sA[GET_S_INDEX(a_tile_col + 3, a_tile_row, BM)] = tmp.w;
                } else {
                    for(int i = 0; i < 4; i++) {
                        float val = (row_start + a_tile_row < M && k_ptr + a_tile_col + i < K) ? A[(row_start + a_tile_row) * K + (k_ptr + a_tile_col + i)] : 0.0f;
                        if ((idx + i) < BM * BK) sA[GET_S_INDEX(a_tile_col + i, a_tile_row, BM)] = val;
                    }
                }
            }
        }

        // 2. 协作搬运 B 到 sB
        #pragma unroll
        for (int p = 0; p < (BK * BN + num_threads * 4 - 1) / (num_threads * 4); p++) {
            int idx = (tid + p * num_threads) * 4;
            if (idx < BK * BN) {
                int b_tile_row = idx / BN; 
                int b_tile_col = idx % BN; 
                const float* g_ptr = &B[(k_ptr + b_tile_row) * N + (col_start + b_tile_col)];

                if (k_ptr + b_tile_row < K && (col_start + b_tile_col + 3) < N && IS_ALIGNED_16(g_ptr) && (b_tile_col % 4 == 0)) {
                    reinterpret_cast<float4*>(&sB[GET_S_INDEX(b_tile_row, b_tile_col, BN)])[0] = reinterpret_cast<const float4*>(g_ptr)[0];
                } else {
                    for(int i = 0; i < 4; i++) {
                        float val = (k_ptr + b_tile_row < K && col_start + b_tile_col + i < N) ? B[(k_ptr + b_tile_row) * N + (col_start + b_tile_col + i)] : 0.0f;
                        if ((idx + i) < BK * BN) sB[GET_S_INDEX(b_tile_row, b_tile_col + i, BN)] = val;
                    }
                }
            }
        }

        __syncthreads();

        // 3. 计算：K 维度的外积累加
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            // 向量化读取 Shared Memory
            float4 rA0 = reinterpret_cast<float4*>(&sA[GET_S_INDEX(kk, ty * 8 + 0, BM)])[0];
            float4 rA1 = reinterpret_cast<float4*>(&sA[GET_S_INDEX(kk, ty * 8 + 4, BM)])[0];
            float4 rB0 = reinterpret_cast<float4*>(&sB[GET_S_INDEX(kk, tx * 8 + 0, BN)])[0];
            float4 rB1 = reinterpret_cast<float4*>(&sB[GET_S_INDEX(kk, tx * 8 + 4, BN)])[0];

            #define OP(idx, a_val) \
                accum[idx][0].x += a_val * rB0.x; accum[idx][0].y += a_val * rB0.y; \
                accum[idx][0].z += a_val * rB0.z; accum[idx][0].w += a_val * rB0.w; \
                accum[idx][1].x += a_val * rB1.x; accum[idx][1].y += a_val * rB1.y; \
                accum[idx][1].z += a_val * rB1.z; accum[idx][1].w += a_val * rB1.w;

            OP(0, rA0.x); OP(1, rA0.y); OP(2, rA0.z); OP(3, rA0.w);
            OP(4, rA1.x); OP(5, rA1.y); OP(6, rA1.z); OP(7, rA1.w);
            #undef OP
        }
        __syncthreads();
    }

    // 4. 写回阶段：将寄存器中的 8x8 结果写回 Global Memory
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int g_r = row_start + ty * 8 + i;
        if (g_r < M) {
            float* row_ptr = &C[g_r * N + col_start + tx * 8];
            float* regs_0 = reinterpret_cast<float*>(&accum[i][0]);
            float* regs_1 = reinterpret_cast<float*>(&accum[i][1]);
            
            #pragma unroll
            for(int j = 0; j < 4; j++) {
                if (col_start + tx * 8 + j < N) row_ptr[j] = regs_0[j];
                if (col_start + tx * 8 + 4 + j < N) row_ptr[4 + j] = regs_1[j];
            }
        }
    }
}


void solve(torch::Tensor A, torch::Tensor B, torch::Tensor C, int N, int M, int K) {
    auto d_A = A.data_ptr<float>(); 
    auto d_B = B.data_ptr<float>(); 
    auto d_C = C.data_ptr<float>();
    
    // 每个线程处理 8x8 的块，因此线程数是 Block 大小的 1/8
    dim3 threads(BLOCK_X / 8, BLOCK_Y / 8); 
    dim3 blocks((N + BLOCK_X - 1) / BLOCK_X, (M + BLOCK_Y - 1) / BLOCK_Y);

    // 实例化模板
    matrix_mul_kernel_outer_product<BLOCK_Y, BLOCK_X, BLOCK_K><<<blocks, threads>>>(d_A, d_B, d_C, N, M, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve", &solve, "Swizzled Optimized GEMM (Macro Configured)",
          py::arg("A"), py::arg("B"), py::arg("C"), 
          py::arg("N"), py::arg("M"), py::arg("K")); 
}