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

// 判断 16 字节对齐（float4 必备）
#define IS_ALIGNED_16(ptr) ((reinterpret_cast<size_t>(ptr) & 15) == 0)

// Swizzle 索引计算：规避 Shared Memory Bank Conflicts
#define GET_S_INDEX(row, col, width) (((row) * (width)) + (((col) / 4 ^ (row)) * 4) + ((col) % 4))

template<int BM, int BN, int BK>
__global__ void matrix_mul_kernel_warp_tile(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, 
                                           int N, int M, int K) 
{
    // 编译器根据宏静态分配共享内存
    extern __shared__ float s_mem[];
    float* sA = s_mem;                
    float* sB = s_mem + BK * BM;      

    // 线程索引获取
    constexpr int NUM_THREADS = (BM * BN) / 64;
    int tid = threadIdx.x; // 1D 布局

    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // 动态支持不同的 BN/BM 分块，基于编译时常量计算布局
    int warp_col = warp_id % (BN / 32); 
    int warp_row = warp_id / (BN / 32); 

    // Lane 布局保持 8x4 最优访存比
    int lane_row = lane_id / 4; // 0-7
    int lane_col = lane_id % 4; // 0-3

    int row_start = blockIdx.y * BM;
    int col_start = blockIdx.x * BN;

    // 寄存器累加器：每个线程负责 8x8 块
    float4 accum[8][2];
    #pragma unroll
    for(int i = 0; i < 8; i++) {
        accum[i][0] = {0.0f, 0.0f, 0.0f, 0.0f};
        accum[i][1] = {0.0f, 0.0f, 0.0f, 0.0f};
    }

    for (int k_ptr = 0; k_ptr < K; k_ptr += BK) {
        
        // 1. 搬运 A (Global -> Shared) 
        #pragma unroll
        for (int p = 0; p < (BM * BK + NUM_THREADS * 4 - 1) / (NUM_THREADS * 4); p++) {
            int idx = (tid + p * NUM_THREADS) * 4;
            if (idx < BM * BK) {
                int a_r = idx / BK; int a_c = idx % BK;
                const float* g_ptr = &A[(row_start + a_r) * K + (k_ptr + a_c)];
                
                if (row_start + a_r < M && (k_ptr + a_c + 3) < K && IS_ALIGNED_16(g_ptr)) {
                    float4 tmp = reinterpret_cast<const float4*>(g_ptr)[0];
                    sA[GET_S_INDEX(a_c+0, a_r, BM)] = tmp.x;
                    sA[GET_S_INDEX(a_c+1, a_r, BM)] = tmp.y;
                    sA[GET_S_INDEX(a_c+2, a_r, BM)] = tmp.z;
                    sA[GET_S_INDEX(a_c+3, a_r, BM)] = tmp.w;
                } else {
                    for(int i=0; i<4; i++) {
                        float val = (row_start + a_r < M && k_ptr + a_c + i < K) ? A[(row_start + a_r)*K + (k_ptr + a_c + i)] : 0.0f;
                        if ((idx + i) < BM * BK) sA[GET_S_INDEX(a_c + i, a_r, BM)] = val;
                    }
                }
            }
        }

        // 2. 搬运 B (Global -> Shared) 
        #pragma unroll
        for (int p = 0; p < (BK * BN + NUM_THREADS * 4 - 1) / (NUM_THREADS * 4); p++) {
            int idx = (tid + p * NUM_THREADS) * 4;
            if (idx < BK * BN) {
                int b_r = idx / BN; int b_c = idx % BN;
                const float* g_ptr = &B[(k_ptr + b_r) * N + (col_start + b_c)];
                
                if (k_ptr + b_r < K && (col_start + b_c + 3) < N && IS_ALIGNED_16(g_ptr) && (b_c % 4 == 0)) {
                    reinterpret_cast<float4*>(&sB[GET_S_INDEX(b_r, b_c, BN)])[0] = reinterpret_cast<const float4*>(g_ptr)[0];
                } else {
                    // --- 修复这里 ---
                    for(int i=0; i<4; i++) {
                        // 使用 b_r 和 b_c + i 代替未定义的 r 和 c
                        float val = (k_ptr + b_r < K && col_start + b_c + i < N) ? B[(k_ptr + b_r) * N + (col_start + b_c + i)] : 0.0f;
                        if ((idx + i) < BK * BN) {
                            sB[GET_S_INDEX(b_r, b_c + i, BN)] = val;
                        }
                    }
                }
            }
        }
        __syncthreads();

        // 3. 计算阶段：Warp 内分块计算
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            // 计算当前 Warp 负责的 Shared Memory 基地址
            int sA_base = warp_row * 64 + lane_row * 8; 
            int sB_base = warp_col * 32 + lane_col * 8;

            float4 rA0 = reinterpret_cast<float4*>(&sA[GET_S_INDEX(kk, sA_base + 0, BM)])[0];
            float4 rA1 = reinterpret_cast<float4*>(&sA[GET_S_INDEX(kk, sA_base + 4, BM)])[0];
            float4 rB0 = reinterpret_cast<float4*>(&sB[GET_S_INDEX(kk, sB_base + 0, BN)])[0];
            float4 rB1 = reinterpret_cast<float4*>(&sB[GET_S_INDEX(kk, sB_base + 4, BN)])[0];

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

    // 4. 写回阶段 
    int final_r = row_start + warp_row * 64 + lane_row * 8;
    int final_c = col_start + warp_col * 32 + lane_col * 8;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int g_r = final_r + i;
        if (g_r < M) {
            float* row_ptr = &C[g_r * N + final_c];
            float* regs_0 = reinterpret_cast<float*>(&accum[i][0]);
            float* regs_1 = reinterpret_cast<float*>(&accum[i][1]);
            #pragma unroll
            for(int j = 0; j < 4; j++) {
                if (final_c + j < N) row_ptr[j] = regs_0[j];
                if (final_c + 4 + j < N) row_ptr[4 + j] = regs_1[j];
            }
        }
    }
}


void solve(torch::Tensor A, torch::Tensor B, torch::Tensor C, int N, int M, int K) {
    auto d_A = A.data_ptr<float>(); 
    auto d_B = B.data_ptr<float>(); 
    auto d_C = C.data_ptr<float>();
    
    // 每个线程算 8x8=64 个元素，动态计算线程数
    constexpr int total_threads = (BLOCK_X * BLOCK_Y) / 64;
    dim3 threads(total_threads); 
    dim3 blocks((N + BLOCK_X - 1) / BLOCK_X, (M + BLOCK_Y - 1) / BLOCK_Y);
    
    // 申请 Shared Memory 大小
    size_t shared_mem_size = (BLOCK_K * BLOCK_Y + BLOCK_K * BLOCK_X) * sizeof(float);

    // 直接实例化模板
    matrix_mul_kernel_warp_tile<BLOCK_Y, BLOCK_X, BLOCK_K><<<blocks, threads, shared_mem_size>>>(d_A, d_B, d_C, N, M, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve", &solve, "Warp Tile Optimized GEMM (Macro Configured)",
          py::arg("A"), py::arg("B"), py::arg("C"), 
          py::arg("N"), py::arg("M"), py::arg("K")); 
}