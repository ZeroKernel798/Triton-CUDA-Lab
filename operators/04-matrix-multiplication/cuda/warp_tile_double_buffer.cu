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

#define IS_ALIGNED_16(ptr) ((reinterpret_cast<size_t>(ptr) & 15) == 0)
#define GET_S_INDEX(row, col, width) (((row) * (width)) + (((col) / 4 ^ (row)) * 4) + ((col) % 4))

template<int BM, int BN, int BK>
__global__ void matrix_mul_kernel_warp_tile_pipelined(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, 
                                                     int N, int M, int K) 
{
    // Shared Memory 翻倍，用于实现双缓冲 (Buffer 0 和 Buffer 1)
    extern __shared__ float s_mem[];
    float* sA = s_mem;                                
    float* sB = s_mem + 2 * BK * BM;                  

    constexpr int NUM_THREADS = (BM * BN) / 64;
    int tid = threadIdx.x; // 1D 布局下直接取 x 即可

    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_col = warp_id % (BN / 32); 
    int warp_row = warp_id / (BN / 32); 
    int lane_row = lane_id / 4; 
    int lane_col = lane_id % 4; 

    int row_start = blockIdx.y * BM;
    int col_start = blockIdx.x * BN;

    // 寄存器累加器
    float4 accum[8][2];
    #pragma unroll
    for(int i = 0; i < 8; i++) {
        accum[i][0] = {0.0f, 0.0f, 0.0f, 0.0f};
        accum[i][1] = {0.0f, 0.0f, 0.0f, 0.0f};
    }

    int write_idx = 0;
    int read_idx  = 0;

    // Global -> Register 的暂存区
    constexpr int FETCH_A = (BM * BK + NUM_THREADS * 4 - 1) / (NUM_THREADS * 4);
    constexpr int FETCH_B = (BK * BN + NUM_THREADS * 4 - 1) / (NUM_THREADS * 4);
    float4 ldg_A[FETCH_A];
    float4 ldg_B[FETCH_B];

    int k_ptr = 0;
    
    // --- Prologue: 预取第一个 Tile 到 Shared Memory ---
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
                float4 val4 = {0.0f, 0.0f, 0.0f, 0.0f};
                if (k_ptr + b_r < K && col_start + b_c + 3 < N && IS_ALIGNED_16(&B[(k_ptr + b_r) * N + (col_start + b_c)])) {
                    val4 = reinterpret_cast<const float4*>(&B[(k_ptr + b_r) * N + (col_start + b_c)])[0];
                } else {
                    for(int i=0; i<4; i++) {
                        if (k_ptr + b_r < K && col_start + b_c + i < N)
                            reinterpret_cast<float*>(&val4)[i] = B[(k_ptr + b_r) * N + (col_start + b_c + i)];
                    }
                }
                reinterpret_cast<float4*>(&sB[write_idx * (BK * BN) + GET_S_INDEX(b_r, b_c, BN)])[0] = val4;
            }
        }
        __syncthreads();
    }

    // 寄存器级别双缓冲：用于隐藏 Shared -> Register 延迟
    float4 rA[2][2], rB[2][2]; 
    int sA_base = warp_row * 64 + lane_row * 8; 
    int sB_base = warp_col * 32 + lane_col * 8;

    // 预取第一个计算步的数据
    rA[0][0] = reinterpret_cast<float4*>(&sA[read_idx * (BK * BM) + GET_S_INDEX(0, sA_base + 0, BM)])[0];
    rA[0][1] = reinterpret_cast<float4*>(&sA[read_idx * (BK * BM) + GET_S_INDEX(0, sA_base + 4, BM)])[0];
    rB[0][0] = reinterpret_cast<float4*>(&sB[read_idx * (BK * BN) + GET_S_INDEX(0, sB_base + 0, BN)])[0];
    rB[0][1] = reinterpret_cast<float4*>(&sB[read_idx * (BK * BN) + GET_S_INDEX(0, sB_base + 4, BN)])[0];


    // --- Main Loop: 深度流水线 ---
    for (k_ptr = BK; k_ptr < K; k_ptr += BK) {
        write_idx ^= 1; 

        // 1. 异步读取下一个 Tile 到寄存器暂存 (隐藏 Global Latency)
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
        #pragma unroll
        for (int p = 0; p < FETCH_B; p++) {
            int idx = (tid + p * NUM_THREADS) * 4;
            if (idx < BK * BN) {
                int b_r = idx / BN; int b_c = idx % BN;
                ldg_B[p] = {0.0f, 0.0f, 0.0f, 0.0f};
                if (k_ptr + b_r < K && col_start + b_c + 3 < N && IS_ALIGNED_16(&B[(k_ptr + b_r) * N + (col_start + b_c)])) {
                    ldg_B[p] = reinterpret_cast<const float4*>(&B[(k_ptr + b_r) * N + (col_start + b_c)])[0];
                } else {
                    for(int i=0; i<4; i++) {
                        if (k_ptr + b_r < K && col_start + b_c + i < N)
                            reinterpret_cast<float*>(&ldg_B[p])[i] = B[(k_ptr + b_r) * N + (col_start + b_c + i)];
                    }
                }
            }
        }

        // 2. 计算当前 Tile (利用寄存器双缓冲掩盖 Shared Latency)
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

        // 3. 将暂存在寄存器中的下一份数据写入 Shared Memory
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
        #pragma unroll
        for (int p = 0; p < FETCH_B; p++) {
            int idx = (tid + p * NUM_THREADS) * 4;
            if (idx < BK * BN) {
                int b_r = idx / BN; int b_c = idx % BN;
                reinterpret_cast<float4*>(&sB[write_idx * (BK * BN) + GET_S_INDEX(b_r, b_c, BN)])[0] = ldg_B[p];
            }
        }
        
        __syncthreads(); 
        read_idx ^= 1;   

        // 为下一轮大循环预取第一份数据
        rA[0][0] = reinterpret_cast<float4*>(&sA[read_idx * (BK * BM) + GET_S_INDEX(0, sA_base + 0, BM)])[0];
        rA[0][1] = reinterpret_cast<float4*>(&sA[read_idx * (BK * BM) + GET_S_INDEX(0, sA_base + 4, BM)])[0];
        rB[0][0] = reinterpret_cast<float4*>(&sB[read_idx * (BK * BN) + GET_S_INDEX(0, sB_base + 0, BN)])[0];
        rB[0][1] = reinterpret_cast<float4*>(&sB[read_idx * (BK * BN) + GET_S_INDEX(0, sB_base + 4, BN)])[0];
    }

    // --- Epilogue: 计算最后一个 Tile ---
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

    // --- Final Write-back ---
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
    
    // 线程布局与 Block 尺寸
    constexpr int total_threads = (BLOCK_Y * BLOCK_X) / 64; 
    dim3 threads(total_threads); 
    dim3 blocks((N + BLOCK_X - 1) / BLOCK_X, (M + BLOCK_Y - 1) / BLOCK_Y);
    
    // 申请双倍 Shared Memory 大小
    size_t shared_mem_size = 2 * (BLOCK_K * BLOCK_Y + BLOCK_K * BLOCK_X) * sizeof(float);

    // 直接实例化模板
    matrix_mul_kernel_warp_tile_pipelined<BLOCK_Y, BLOCK_X, BLOCK_K><<<blocks, threads, shared_mem_size>>>(d_A, d_B, d_C, N, M, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve", &solve, "Pipelined Warp-Tiling GEMM (Macro Configured)",
          py::arg("A"), py::arg("B"), py::arg("C"), 
          py::arg("N"), py::arg("M"), py::arg("K")); 
}