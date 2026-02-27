#include <cuda_runtime.h>
#include <torch/extension.h>
#include <string>

// 判断是否对齐 帮助实现float4向量化读取
#define IS_ALIGNED_16(ptr) ((reinterpret_cast<size_t>(ptr) & 15) == 0)

// Swizzle 索引计算 用于规避银行冲突
#define GET_S_INDEX(row, col, width) (((row) * (width)) + (((col) / 4 ^ (row)) * 4) + ((col) % 4))

template<int BM, int BN, int BK>
__global__ void matrix_mul_kernel_outer_product(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, 
                                               int N, int M, int K) 
{
    // 对齐到 16 字节的共享内存
    __shared__ float sA[BK * BM];
    __shared__ float sB[BK * BN];

    // 线程索引以及线程块的线程数量
    int tx = threadIdx.x; 
    int ty = threadIdx.y; 
    int tid = ty * blockDim.x + tx;
    int num_threads = blockDim.x * blockDim.y; 

    int row_start = blockIdx.y * BM;
    int col_start = blockIdx.x * BN;

    // 寄存器 用于存储累加和
    float4 accum[8][2] = {0.0f};

    // 外层 K 循环
    for (int k_ptr = 0; k_ptr < K; k_ptr += BK) {
        
        // 搬运 A 到 sA 
        #pragma unroll
        for (int p = 0; p < (BM * BK + num_threads * 4 - 1) / (num_threads * 4); p++) {
            int idx = (tid + p * num_threads) * 4;
            if (idx < BM * BK) {
                int a_tile_row = idx / BK; // 在 BM 维度
                int a_tile_col = idx % BK; // 在 BK 维度
                
                // 这个是全局显存的地址
                const float* g_ptr = &A[(row_start + a_tile_row) * K + (k_ptr + a_tile_col)];
                
                // 向量化读取 Global A
                if (row_start + a_tile_row < M && (k_ptr + a_tile_col + 3) < K && IS_ALIGNED_16(g_ptr)) {
                    float4 tmp = reinterpret_cast<const float4*>(g_ptr)[0];
                    sA[GET_S_INDEX(a_tile_col + 0, a_tile_row, BM)] = tmp.x;
                    sA[GET_S_INDEX(a_tile_col + 1, a_tile_row, BM)] = tmp.y;
                    sA[GET_S_INDEX(a_tile_col + 2, a_tile_row, BM)] = tmp.z;
                    sA[GET_S_INDEX(a_tile_col + 3, a_tile_row, BM)] = tmp.w;
                } else {
                    // 越界强制补零 
                    for(int i = 0; i < 4; i++) {
                        int r = a_tile_row; 
                        int c = a_tile_col + i;
                        float val = 0.0f; 
                        if (row_start + r < M && k_ptr + c < K) {
                            val = A[(row_start + r) * K + (k_ptr + c)];
                        }
                        // 只要属于当前处理的 4 个 float 范围，就写入 sA (无论是真实值还是 0)
                        if ((idx + i) < BM * BK) {
                            sA[GET_S_INDEX(c, r, BM)] = val;
                        }
                    }
                }
            }
        }

        // 搬运 B 到 sB
        #pragma unroll
        for (int p = 0; p < (BK * BN + num_threads * 4 - 1) / (num_threads * 4); p++) {
            int idx = (tid + p * num_threads) * 4;
            if (idx < BK * BN) {
                int b_tile_row = idx / BN; // 在 BK 维度
                int b_tile_col = idx % BN; // 在 BN 维度

                const float* g_ptr = &B[(k_ptr + b_tile_row) * N + (col_start + b_tile_col)];

                // 向量化读取并存入 sB
                if (k_ptr + b_tile_row < K && (col_start + b_tile_col + 3) < N && IS_ALIGNED_16(g_ptr) && (b_tile_col % 4 == 0)) {
                    reinterpret_cast<float4*>(&sB[GET_S_INDEX(b_tile_row, b_tile_col, BN)])[0] = reinterpret_cast<const float4*>(g_ptr)[0];
                } else {
                    // 越界强制补零 
                    for(int i = 0; i < 4; i++) {
                        int r = b_tile_row; 
                        int c = b_tile_col + i;
                        float val = 0.0f; 
                        if (k_ptr + r < K && col_start + c < N) {
                            val = B[(k_ptr + r) * N + (col_start + c)];
                        }
                        if ((idx + i) < BK * BN) {
                            sB[GET_S_INDEX(r, c, BN)] = val;
                        }
                    }
                }
            }
        }

        __syncthreads();

        // 外积计算 注意顺序是K->M->N
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
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

    // 写回阶段 
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

// 启动器
void solve(torch::Tensor A, torch::Tensor B, torch::Tensor C, 
           int N, int M, int K, int bx, int by, int bk) {
    auto d_A = A.data_ptr<float>(); 
    auto d_B = B.data_ptr<float>(); 
    auto d_C = C.data_ptr<float>();
    
    dim3 threads(bx / 8, by / 8); 
    dim3 blocks((N + bx - 1) / bx, (M + by - 1) / by);

    if (bx == 128 && by == 128 && bk == 8) {
        matrix_mul_kernel_outer_product<128, 128, 8><<<blocks, threads>>>(d_A, d_B, d_C, N, M, K);
    } else if (bx == 64 && by == 128 && bk == 8) {
        matrix_mul_kernel_outer_product<128, 64, 8><<<blocks, threads>>>(d_A, d_B, d_C, N, M, K);
    } else if (bx == 128 && by == 64 && bk == 8) {
        matrix_mul_kernel_outer_product<64, 128, 8><<<blocks, threads>>>(d_A, d_B, d_C, N, M, K);
    } else{
        matrix_mul_kernel_outer_product<32, 32, 32><<<blocks, threads>>>(d_A, d_B, d_C, N, M, K);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve", &solve, "Swizzled Optimized GEMM",
          py::arg("A"), py::arg("B"), py::arg("C"), 
          py::arg("N"), py::arg("M"), py::arg("K"), 
          py::arg("bx"), py::arg("by"), py::arg("bk")); 
}