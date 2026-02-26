#include <cuda_runtime.h>
#include <torch/extension.h>
#include <string>

// 判断 16 字节对齐（float4 必备）
#define IS_ALIGNED_16(ptr) ((reinterpret_cast<size_t>(ptr) & 15) == 0)

// Swizzle 索引计算：规避 Shared Memory Bank Conflicts
#define GET_S_INDEX(row, col, width) (((row) * (width)) + (((col) / 4 ^ (row)) * 4) + ((col) % 4))

template<int BM, int BN, int BK>
__global__ void matrix_mul_kernel_warp_tile(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, 
                                           int N, int M, int K) 
{
    extern __shared__ float s_mem[];
    float* sA = s_mem;                
    float* sB = s_mem + BK * BM;      

    // 1. 线程索引获取
    int num_threads = blockDim.x * blockDim.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // 修复 1：动态支持不同的 BN/BM 分块，不再写死 2x4 布局
    // 每个 Warp 在横向负责 32 个元素 (因为 lane_col 最大是 3, 3*8 + 8 = 32)
    int warp_col = warp_id % (BN / 32); 
    int warp_row = warp_id / (BN / 32); 

    // Lane 布局保持 8x4 最优访存比
    int lane_row = lane_id / 4; // 0-7
    int lane_col = lane_id % 4; // 0-3

    int row_start = blockIdx.y * BM;
    int col_start = blockIdx.x * BN;

    float4 accum[8][2];
    #pragma unroll
    for(int i = 0; i < 8; i++) {
        accum[i][0] = {0.0f, 0.0f, 0.0f, 0.0f};
        accum[i][1] = {0.0f, 0.0f, 0.0f, 0.0f};
    }

    for (int k_ptr = 0; k_ptr < K; k_ptr += BK) {
        
        // --- 搬运 A (Global -> Shared) 补零版 ---
        #pragma unroll
        for (int p = 0; p < (BM * BK + num_threads * 4 - 1) / (num_threads * 4); p++) {
            int idx = (tid + p * num_threads) * 4;
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
                    // 💡 修复 2：越界强制写 0 覆盖脏数据
                    for(int i=0; i<4; i++) {
                        int r = a_r; 
                        int c = a_c + i;
                        float val = 0.0f;
                        if(row_start + r < M && k_ptr + c < K) {
                            val = A[(row_start + r)*K + (k_ptr + c)];
                        }
                        if ((idx + i) < BM * BK) {
                            sA[GET_S_INDEX(c, r, BM)] = val;
                        }
                    }
                }
            }
        }

        // --- 搬运 B (Global -> Shared) 补零版 ---
        #pragma unroll
        for (int p = 0; p < (BK * BN + num_threads * 4 - 1) / (num_threads * 4); p++) {
            int idx = (tid + p * num_threads) * 4;
            if (idx < BK * BN) {
                int b_r = idx / BN; int b_c = idx % BN;
                const float* g_ptr = &B[(k_ptr + b_r) * N + (col_start + b_c)];
                
                if (k_ptr + b_r < K && (col_start + b_c + 3) < N && IS_ALIGNED_16(g_ptr) && (b_c % 4 == 0)) {
                    reinterpret_cast<float4*>(&sB[GET_S_INDEX(b_r, b_c, BN)])[0] = reinterpret_cast<const float4*>(g_ptr)[0];
                } else {
                    // 💡 修复 3：越界强制写 0 覆盖脏数据
                    for(int i=0; i<4; i++) {
                        int r = b_r; 
                        int c = b_c + i;
                        float val = 0.0f;
                        if(k_ptr + r < K && col_start + c < N) {
                            val = B[(k_ptr + r)*N + (col_start + c)];
                        }
                        if ((idx + i) < BK * BN) {
                            sB[GET_S_INDEX(r, c, BN)] = val;
                        }
                    }
                }
            }
        }
        __syncthreads();

        // --- 计算阶段 ---
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
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

    // --- 写回阶段 ---
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

void solve(torch::Tensor A, torch::Tensor B, torch::Tensor C, 
           int N, int M, int K, int bx, int by, int bk, std::string version) {
    auto d_A = A.data_ptr<float>(); 
    auto d_B = B.data_ptr<float>(); 
    auto d_C = C.data_ptr<float>();
    
    // 修复 4：动态计算线程数。每个线程算 8x8=64 个元素
    // 这样当传入 128x64 时，线程数自动变为 128，而不是写死的 256
    int total_threads = (bx * by) / 64;
    dim3 threads(total_threads); 
    dim3 blocks((N + bx - 1) / bx, (M + by - 1) / by);
    size_t shared_mem_size = (bk * bx + bk * by) * sizeof(float);

    if (bx == 128 && by == 128 && bk == 8) {
        matrix_mul_kernel_warp_tile<128, 128, 8><<<blocks, threads, shared_mem_size>>>(d_A, d_B, d_C, N, M, K);
    } else if (bx == 128 && by == 64 && bk == 8) {
        matrix_mul_kernel_warp_tile<64, 128, 8><<<blocks, threads, shared_mem_size>>>(d_A, d_B, d_C, N, M, K);
    } else if (bx == 64 && by == 128 && bk == 8) {
        matrix_mul_kernel_warp_tile<128, 64, 8><<<blocks, threads, shared_mem_size>>>(d_A, d_B, d_C, N, M, K);
    } else {
        matrix_mul_kernel_warp_tile<32, 32, 32><<<blocks, threads, shared_mem_size>>>(d_A, d_B, d_C, N, M, K);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve", &solve, "Warp Tile Optimized GEMM",
          py::arg("A"), py::arg("B"), py::arg("C"), 
          py::arg("N"), py::arg("M"), py::arg("K"), 
          py::arg("bx"), py::arg("by"), py::arg("bk"),
          py::arg("version")); 
}