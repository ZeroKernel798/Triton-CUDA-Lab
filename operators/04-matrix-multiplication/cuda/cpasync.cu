#include <cuda_runtime.h>
#include <torch/extension.h>
#include <string>
#include <cuda_pipeline_primitives.h> 

#ifndef BLOCK_X
#define BLOCK_X 128
#endif
#ifndef BLOCK_Y
#define BLOCK_Y 128
#endif
#ifndef BLOCK_K
#define BLOCK_K 8
#endif

#define IS_ALIGNED_16(ptr) ((reinterpret_cast<size_t>(ptr) & 15) == 0)
// 使用 XOR 索引解决 Shared Memory Bank Conflict
#define GET_S_INDEX(row, col, width) (((row) * (width)) + (((col) / 4 ^ (row)) * 4) + ((col) % 4))

template<int BM, int BN, int BK>
__global__ void matrix_mul_kernel_warp_tile_cp_async(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, 
                                                     int N, int M, int K) 
{
    // 静态分配或通过 extern 传入，逻辑保持不变
    extern __shared__ float s_mem[];
    float* sA = s_mem;                                
    float* sB = s_mem + 2 * BK * BM;                  

    constexpr int NUM_THREADS = (BM * BN) / 64;
    int tid = threadIdx.x; // 注意：由于是 1D threads，直接取 threadIdx.x

    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_col = warp_id % (BN / 32); 
    int warp_row = warp_id / (BN / 32); 
    int lane_row = lane_id / 4; 
    int lane_col = lane_id % 4; 

    int row_start = blockIdx.y * BM;
    int col_start = blockIdx.x * BN;

    // 寄存器堆初始化
    float4 accum[8][2];
    #pragma unroll
    for(int i = 0; i < 8; i++) {
        accum[i][0] = {0.0f, 0.0f, 0.0f, 0.0f};
        accum[i][1] = {0.0f, 0.0f, 0.0f, 0.0f};
    }

    int write_idx = 0;
    int read_idx  = 0;

    constexpr int FETCH_A = (BM * BK + NUM_THREADS * 4 - 1) / (NUM_THREADS * 4);
    constexpr int FETCH_B = (BK * BN + NUM_THREADS * 4 - 1) / (NUM_THREADS * 4);
    float4 ldg_A[FETCH_A];

    int k_ptr = 0;
    
    // --- Prologue: 第一块数据搬运 ---
    if (k_ptr < K) {
        #pragma unroll
        for (int p = 0; p < FETCH_A; p++) {
            int idx = (tid + p * NUM_THREADS) * 4;
            if (idx < BM * BK) {
                int a_r = idx / BK; int a_c = idx % BK;
                float4 val4 = {0.0f, 0.0f, 0.0f, 0.0f};
                if (row_start + a_r < M && k_ptr + a_c + 3 < K && IS_ALIGNED_16(&A[(row_start + a_r) * K + (k_ptr + a_c)])) {
                    val4 = reinterpret_cast<const float4*>(&A[(row_start + a_r) * K + (k_ptr + a_c)])[0];
                } else {
                    for(int i=0; i<4; i++) {
                        if (row_start + a_r < M && k_ptr + a_c + i < K)
                            reinterpret_cast<float*>(&val4)[i] = A[(row_start + a_r) * K + (k_ptr + a_c + i)];
                    }
                }
                sA[write_idx * (BK * BM) + GET_S_INDEX(a_c+0, a_r, BM)] = val4.x;
                sA[write_idx * (BK * BM) + GET_S_INDEX(a_c+1, a_r, BM)] = val4.y;
                sA[write_idx * (BK * BM) + GET_S_INDEX(a_c+2, a_r, BM)] = val4.z;
                sA[write_idx * (BK * BM) + GET_S_INDEX(a_c+3, a_r, BM)] = val4.w;
            }
        }
        
        #pragma unroll
        for (int p = 0; p < FETCH_B; p++) {
            int idx = (tid + p * NUM_THREADS) * 4;
            if (idx < BK * BN) {
                int b_r = idx / BN; int b_c = idx % BN;
                const float* g_ptr = &B[(k_ptr + b_r) * N + (col_start + b_c)];
                float* s_ptr = &sB[write_idx * (BK * BN) + GET_S_INDEX(b_r, b_c, BN)];

                if (k_ptr + b_r < K && col_start + b_c + 3 < N && IS_ALIGNED_16(g_ptr)) {
                    __pipeline_memcpy_async(s_ptr, g_ptr, 16);
                } else {
                    for(int i=0; i<4; i++) {
                        float val = (k_ptr + b_r < K && col_start + b_c + i < N) ? B[(k_ptr + b_r) * N + (col_start + b_c + i)] : 0.0f;
                        sB[write_idx * (BK * BN) + GET_S_INDEX(b_r, b_c + i, BN)] = val;
                    }
                }
            }
        }
        __pipeline_commit();       
        __pipeline_wait_prior(0);  
        __syncthreads();
    }

    // --- Main Loop: Double Buffering ---
    float4 rA[2][2], rB[2][2]; 
    int sA_base = warp_row * 64 + lane_row * 8; 
    int sB_base = warp_col * 32 + lane_col * 8;

    rA[0][0] = reinterpret_cast<float4*>(&sA[read_idx * (BK * BM) + GET_S_INDEX(0, sA_base + 0, BM)])[0];
    rA[0][1] = reinterpret_cast<float4*>(&sA[read_idx * (BK * BM) + GET_S_INDEX(0, sA_base + 4, BM)])[0];
    rB[0][0] = reinterpret_cast<float4*>(&sB[read_idx * (BK * BN) + GET_S_INDEX(0, sB_base + 0, BN)])[0];
    rB[0][1] = reinterpret_cast<float4*>(&sB[read_idx * (BK * BN) + GET_S_INDEX(0, sB_base + 4, BN)])[0];

    for (k_ptr = BK; k_ptr < K; k_ptr += BK) {
        write_idx ^= 1; 

        // 异步发射下一块数据
        #pragma unroll
        for (int p = 0; p < FETCH_B; p++) {
            int idx = (tid + p * NUM_THREADS) * 4;
            if (idx < BK * BN) {
                int b_r = idx / BN; int b_c = idx % BN;
                const float* g_ptr = &B[(k_ptr + b_r) * N + (col_start + b_c)];
                float* s_ptr = &sB[write_idx * (BK * BN) + GET_S_INDEX(b_r, b_c, BN)];
                if (k_ptr + b_r < K && col_start + b_c + 3 < N && IS_ALIGNED_16(g_ptr)) {
                    __pipeline_memcpy_async(s_ptr, g_ptr, 16);
                } else {
                    for(int i=0; i<4; i++) {
                        float val = (k_ptr + b_r < K && col_start + b_c + i < N) ? B[(k_ptr + b_r) * N + (col_start + b_c + i)] : 0.0f;
                        sB[write_idx * (BK * BN) + GET_S_INDEX(b_r, b_c + i, BN)] = val;
                    }
                }
            }
        }
        #pragma unroll
        for (int p = 0; p < FETCH_A; p++) {
            int idx = (tid + p * NUM_THREADS) * 4;
            if (idx < BM * BK) {
                int a_r = idx / BK; int a_c = idx % BK;
                ldg_A[p] = {0.0f, 0.0f, 0.0f, 0.0f};
                if (row_start + a_r < M && k_ptr + a_c + 3 < K && IS_ALIGNED_16(&A[(row_start + a_r) * K + (k_ptr + a_c)])) {
                    ldg_A[p] = reinterpret_cast<const float4*>(&A[(row_start + a_r) * K + (k_ptr + a_c)])[0];
                } else {
                    for(int i=0; i<4; i++) {
                        if (row_start + a_r < M && k_ptr + a_c + i < K)
                            reinterpret_cast<float*>(&ldg_A[p])[i] = A[(row_start + a_r) * K + (k_ptr + a_c + i)];
                    }
                }
            }
        }
        __pipeline_commit(); 

        // 计算当前块，掩盖访存延迟
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            int load_reg_idx = (kk + 1) % 2; 
            int comp_reg_idx = kk % 2;
            if (kk < BK - 1) {
                rA[load_reg_idx][0] = reinterpret_cast<float4*>(&sA[read_idx * (BK * BM) + GET_S_INDEX(kk+1, sA_base + 0, BM)])[0];
                rA[load_reg_idx][1] = reinterpret_cast<float4*>(&sA[read_idx * (BK * BM) + GET_S_INDEX(kk+1, sA_base + 4, BM)])[0];
                rB[load_reg_idx][0] = reinterpret_cast<float4*>(&sB[read_idx * (BK * BN) + GET_S_INDEX(kk+1, sB_base + 0, BN)])[0];
                rB[load_reg_idx][1] = reinterpret_cast<float4*>(&sB[read_idx * (BK * BN) + GET_S_INDEX(kk+1, sB_base + 4, BN)])[0];
            }
            #define OP(idx, a_val) \
                accum[idx][0].x += a_val * rB[comp_reg_idx][0].x; accum[idx][0].y += a_val * rB[comp_reg_idx][0].y; \
                accum[idx][0].z += a_val * rB[comp_reg_idx][0].z; accum[idx][0].w += a_val * rB[comp_reg_idx][0].w; \
                accum[idx][1].x += a_val * rB[comp_reg_idx][1].x; accum[idx][1].y += a_val * rB[comp_reg_idx][1].y; \
                accum[idx][1].z += a_val * rB[comp_reg_idx][1].z; accum[idx][1].w += a_val * rB[comp_reg_idx][1].w;
            OP(0, rA[comp_reg_idx][0].x); OP(1, rA[comp_reg_idx][0].y); OP(2, rA[comp_reg_idx][0].z); OP(3, rA[comp_reg_idx][0].w);
            OP(4, rA[comp_reg_idx][1].x); OP(5, rA[comp_reg_idx][1].y); OP(6, rA[comp_reg_idx][1].z); OP(7, rA[comp_reg_idx][1].w);
            #undef OP
        }

        // 写回 A 到 Shared
        #pragma unroll
        for (int p = 0; p < FETCH_A; p++) {
            int idx = (tid + p * NUM_THREADS) * 4;
            if (idx < BM * BK) {
                int a_r = idx / BK; int a_c = idx % BK;
                sA[write_idx * (BK * BM) + GET_S_INDEX(a_c+0, a_r, BM)] = ldg_A[p].x;
                sA[write_idx * (BK * BM) + GET_S_INDEX(a_c+1, a_r, BM)] = ldg_A[p].y;
                sA[write_idx * (BK * BM) + GET_S_INDEX(a_c+2, a_r, BM)] = ldg_A[p].z;
                sA[write_idx * (BK * BM) + GET_S_INDEX(a_c+3, a_r, BM)] = ldg_A[p].w;
            }
        }
        __pipeline_wait_prior(0); 
        __syncthreads(); 
        read_idx ^= 1;   

        rA[0][0] = reinterpret_cast<float4*>(&sA[read_idx * (BK * BM) + GET_S_INDEX(0, sA_base + 0, BM)])[0];
        rA[0][1] = reinterpret_cast<float4*>(&sA[read_idx * (BK * BM) + GET_S_INDEX(0, sA_base + 4, BM)])[0];
        rB[0][0] = reinterpret_cast<float4*>(&sB[read_idx * (BK * BN) + GET_S_INDEX(0, sB_base + 0, BN)])[0];
        rB[0][1] = reinterpret_cast<float4*>(&sB[read_idx * (BK * BN) + GET_S_INDEX(0, sB_base + 4, BN)])[0];
    }

    // --- Epilogue: 最后一块计算 ---
    #pragma unroll
    for (int kk = 0; kk < BK; kk++) {
        int load_reg_idx = (kk + 1) % 2; 
        int comp_reg_idx = kk % 2;
        if (kk < BK - 1) {
            rA[load_reg_idx][0] = reinterpret_cast<float4*>(&sA[read_idx * (BK * BM) + GET_S_INDEX(kk+1, sA_base + 0, BM)])[0];
            rA[load_reg_idx][1] = reinterpret_cast<float4*>(&sA[read_idx * (BK * BM) + GET_S_INDEX(kk+1, sA_base + 4, BM)])[0];
            rB[load_reg_idx][0] = reinterpret_cast<float4*>(&sB[read_idx * (BK * BN) + GET_S_INDEX(kk+1, sB_base + 0, BN)])[0];
            rB[load_reg_idx][1] = reinterpret_cast<float4*>(&sB[read_idx * (BK * BN) + GET_S_INDEX(kk+1, sB_base + 4, BN)])[0];
        }
        #define OP(idx, a_val) \
            accum[idx][0].x += a_val * rB[comp_reg_idx][0].x; accum[idx][0].y += a_val * rB[comp_reg_idx][0].y; \
            accum[idx][0].z += a_val * rB[comp_reg_idx][0].z; accum[idx][0].w += a_val * rB[comp_reg_idx][0].w; \
            accum[idx][1].x += a_val * rB[comp_reg_idx][1].x; accum[idx][1].y += a_val * rB[comp_reg_idx][1].y; \
            accum[idx][1].z += a_val * rB[comp_reg_idx][1].z; accum[idx][1].w += a_val * rB[comp_reg_idx][1].w;
        OP(0, rA[comp_reg_idx][0].x); OP(1, rA[comp_reg_idx][0].y); OP(2, rA[comp_reg_idx][0].z); OP(3, rA[comp_reg_idx][0].w);
        OP(4, rA[comp_reg_idx][1].x); OP(5, rA[comp_reg_idx][1].y); OP(6, rA[comp_reg_idx][1].z); OP(7, rA[comp_reg_idx][1].w);
        #undef OP
    }

    // --- Write back to Global Memory ---
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
    
    // 线程数和 Block 大小现在在编译时由宏确定
    constexpr int total_threads = (BLOCK_Y * BLOCK_X) / 64; 
    dim3 threads(total_threads); 
    dim3 blocks((N + BLOCK_X - 1) / BLOCK_X, (M + BLOCK_Y - 1) / BLOCK_Y);
    
    // 共享内存大小计算
    size_t shared_mem_size = 2 * (BLOCK_K * BLOCK_Y + BLOCK_K * BLOCK_X) * sizeof(float);

    // 直接实例化模板，省去繁琐的 if-else 分支
    matrix_mul_kernel_warp_tile_cp_async<BLOCK_Y, BLOCK_X, BLOCK_K><<<blocks, threads, shared_mem_size>>>(d_A, d_B, d_C, N, M, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve", &solve, "CP_ASYNC Hybrid GEMM (Macro Configured)",
          py::arg("A"), py::arg("B"), py::arg("C"), 
          py::arg("N"), py::arg("M"), py::arg("K")); 
}