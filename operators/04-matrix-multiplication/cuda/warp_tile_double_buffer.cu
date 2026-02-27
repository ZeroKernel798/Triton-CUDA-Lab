#include <cuda_runtime.h>
#include <torch/extension.h>
#include <string>

#define IS_ALIGNED_16(ptr) ((reinterpret_cast<size_t>(ptr) & 15) == 0)
#define GET_S_INDEX(row, col, width) (((row) * (width)) + (((col) / 4 ^ (row)) * 4) + ((col) % 4))

template<int BM, int BN, int BK>
__global__ void matrix_mul_kernel_warp_tile_pipelined(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, 
                                                     int N, int M, int K) 
{
    // 💡 1. Shared Memory 翻倍，分为 Buffer 0 和 Buffer 1
    extern __shared__ float s_mem[];
    float* sA = s_mem;                                // 大小: 2 * BK * BM
    float* sB = s_mem + 2 * BK * BM;                  // 大小: 2 * BK * BN

    // 💡 2. 编译期计算线程数，用于分配定长寄存器数组
    constexpr int NUM_THREADS = (BM * BN) / 64;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_col = warp_id % (BN / 32); 
    int warp_row = warp_id / (BN / 32); 
    int lane_row = lane_id / 4; 
    int lane_col = lane_id % 4; 

    int row_start = blockIdx.y * BM;
    int col_start = blockIdx.x * BN;

    float4 accum[8][2];
    #pragma unroll
    for(int i = 0; i < 8; i++) {
        accum[i][0] = {0.0f, 0.0f, 0.0f, 0.0f};
        accum[i][1] = {0.0f, 0.0f, 0.0f, 0.0f};
    }

    // Ping-Pong 索引
    int write_idx = 0;
    int read_idx  = 0;

    // 💡 用于 Global -> Shared 的暂存寄存器 (隐藏 Load 延迟)
    constexpr int FETCH_A = (BM * BK + NUM_THREADS * 4 - 1) / (NUM_THREADS * 4);
    constexpr int FETCH_B = (BK * BN + NUM_THREADS * 4 - 1) / (NUM_THREADS * 4);
    float4 ldg_A[FETCH_A];
    float4 ldg_B[FETCH_B];

    // ===================================================================
    // 🚀 Prologue (开场): 加载第 0 个 Tile 到 Buffer 0
    // ===================================================================
    int k_ptr = 0;
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

    // 💡 寄存器级别的 Double Buffer，用于隐藏 Shared -> Register 的延迟
    float4 rA[2][2], rB[2][2]; 
    int sA_base = warp_row * 64 + lane_row * 8; 
    int sB_base = warp_col * 32 + lane_col * 8;

    // 预取第 0 次计算所需的寄存器数据
    rA[0][0] = reinterpret_cast<float4*>(&sA[read_idx * (BK * BM) + GET_S_INDEX(0, sA_base + 0, BM)])[0];
    rA[0][1] = reinterpret_cast<float4*>(&sA[read_idx * (BK * BM) + GET_S_INDEX(0, sA_base + 4, BM)])[0];
    rB[0][0] = reinterpret_cast<float4*>(&sB[read_idx * (BK * BN) + GET_S_INDEX(0, sB_base + 0, BN)])[0];
    rB[0][1] = reinterpret_cast<float4*>(&sB[read_idx * (BK * BN) + GET_S_INDEX(0, sB_base + 4, BN)])[0];

    // ===================================================================
    // 🚀 Main Pipeline: 边算当前 Tile，边加载下一个 Tile
    // ===================================================================
    for (k_ptr = BK; k_ptr < K; k_ptr += BK) {
        write_idx ^= 1; // 切换写入目标到下一个 Buffer

        // 步骤 1：从 Global Memory 异步读取下一个 Tile 到寄存器 (ldg_A, ldg_B)
        // 这一步开始发出全局访存请求
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

        // 步骤 2：在全局访存飞行的同时，用寄存器双缓冲执行当前 Tile 的计算
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            int load_reg_idx = (kk + 1) % 2; 
            int comp_reg_idx = kk % 2;

            // 提前预取下一步需要的寄存器数据
            if (kk < BK - 1) {
                rA[load_reg_idx][0] = reinterpret_cast<float4*>(&sA[read_idx * (BK * BM) + GET_S_INDEX(kk+1, sA_base + 0, BM)])[0];
                rA[load_reg_idx][1] = reinterpret_cast<float4*>(&sA[read_idx * (BK * BM) + GET_S_INDEX(kk+1, sA_base + 4, BM)])[0];
                rB[load_reg_idx][0] = reinterpret_cast<float4*>(&sB[read_idx * (BK * BN) + GET_S_INDEX(kk+1, sB_base + 0, BN)])[0];
                rB[load_reg_idx][1] = reinterpret_cast<float4*>(&sB[read_idx * (BK * BN) + GET_S_INDEX(kk+1, sB_base + 4, BN)])[0];
            }

            // 计算当前步的乘加指令 (FFMA)
            #define OP(idx, a_val) \
                accum[idx][0].x += a_val * rB[comp_reg_idx][0].x; accum[idx][0].y += a_val * rB[comp_reg_idx][0].y; \
                accum[idx][0].z += a_val * rB[comp_reg_idx][0].z; accum[idx][0].w += a_val * rB[comp_reg_idx][0].w; \
                accum[idx][1].x += a_val * rB[comp_reg_idx][1].x; accum[idx][1].y += a_val * rB[comp_reg_idx][1].y; \
                accum[idx][1].z += a_val * rB[comp_reg_idx][1].z; accum[idx][1].w += a_val * rB[comp_reg_idx][1].w;

            OP(0, rA[comp_reg_idx][0].x); OP(1, rA[comp_reg_idx][0].y); OP(2, rA[comp_reg_idx][0].z); OP(3, rA[comp_reg_idx][0].w);
            OP(4, rA[comp_reg_idx][1].x); OP(5, rA[comp_reg_idx][1].y); OP(6, rA[comp_reg_idx][1].z); OP(7, rA[comp_reg_idx][1].w);
            #undef OP
        }

        // 步骤 3：当前 Tile 计算完毕，将寄存器中已拉取的下一个 Tile 写入目标 Buffer
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
        
        __syncthreads(); // 等待所有人将下一个 Buffer 写满
        read_idx ^= 1;   // 切换读取目标

        // 预取下一个 Tile 的第一份寄存器数据
        rA[0][0] = reinterpret_cast<float4*>(&sA[read_idx * (BK * BM) + GET_S_INDEX(0, sA_base + 0, BM)])[0];
        rA[0][1] = reinterpret_cast<float4*>(&sA[read_idx * (BK * BM) + GET_S_INDEX(0, sA_base + 4, BM)])[0];
        rB[0][0] = reinterpret_cast<float4*>(&sB[read_idx * (BK * BN) + GET_S_INDEX(0, sB_base + 0, BN)])[0];
        rB[0][1] = reinterpret_cast<float4*>(&sB[read_idx * (BK * BN) + GET_S_INDEX(0, sB_base + 4, BN)])[0];
    }

    // ===================================================================
    // 🚀 Epilogue (收尾): 计算最后一个 Tile
    // ===================================================================
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

    // --- 写回阶段 (原封不动) ---
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

// 启动器
void solve(torch::Tensor A, torch::Tensor B, torch::Tensor C, 
           int N, int M, int K, int bx, int by, int bk) {
    auto d_A = A.data_ptr<float>(); 
    auto d_B = B.data_ptr<float>(); 
    auto d_C = C.data_ptr<float>();
    
    int total_threads = (bx * by) / 64;
    dim3 threads(total_threads); 
    dim3 blocks((N + bx - 1) / bx, (M + by - 1) / by);
    
    // 💡 必须申请双倍的 Shared Memory！
    size_t shared_mem_size = 2 * (bk * bx + bk * by) * sizeof(float);

    if (bx == 128 && by == 128 && bk == 8) {
        matrix_mul_kernel_warp_tile_pipelined<128, 128, 8><<<blocks, threads, shared_mem_size>>>(d_A, d_B, d_C, N, M, K);
    } else if (bx == 128 && by == 64 && bk == 8) {
        matrix_mul_kernel_warp_tile_pipelined<128, 64, 8><<<blocks, threads, shared_mem_size>>>(d_A, d_B, d_C, N, M, K);
    } else if (bx == 64 && by == 128 && bk == 8) {
        matrix_mul_kernel_warp_tile_pipelined<64, 128, 8><<<blocks, threads, shared_mem_size>>>(d_A, d_B, d_C, N, M, K);
    } else {
        matrix_mul_kernel_warp_tile_pipelined<32, 32, 32><<<blocks, threads, shared_mem_size>>>(d_A, d_B, d_C, N, M, K);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve", &solve, "Pipelined GEMM",
          py::arg("A"), py::arg("B"), py::arg("C"), 
          py::arg("N"), py::arg("M"), py::arg("K"), 
          py::arg("bx"), py::arg("by"), py::arg("bk")); 
}
