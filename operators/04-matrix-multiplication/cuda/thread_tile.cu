#include <cuda_runtime.h>
#include <torch/extension.h>
#include <string>

// 纯净版外积 Kernel：无 float4，无 Swizzle
template<int BM, int BN, int BK>
__global__ void matrix_mul_kernel_outer_product_pure(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, 
                                                     int N, int M, int K) 
{
    // 共享内存
    __shared__ float sA[BK * BM];      // 布局: [BK][BM]  这里的 A 在 shared memory 中是被转置存放的，为了减少银行冲突
    __shared__ float sB[BK * BN];      // 布局: [BK][BN]

    // 线程索引以及线程块的线程数量
    int tx = threadIdx.x; 
    int ty = threadIdx.y; 
    int tid = ty * blockDim.x + tx;
    int num_threads = blockDim.x * blockDim.y; 

    // 当前 Block 负责全局矩阵的起始位置 
    int row_start = blockIdx.y * BM;
    int col_start = blockIdx.x * BN;

    // 寄存器 存储结果
    float accum[8][8] = {0.0f};

    // 外层 K 循环
    for (int k_offset = 0; k_offset < K; k_offset += BK) {
        
        // 搬运 A 到 sA 采用类似于网格跨步循环的思路
        // 将大小为 [BM][BK] 的数据搬运到 [BK][BM] 的 sA 中
        for (int idx = tid; idx < BM * BK; idx += num_threads) {
            int a_tile_row = idx / BK; // 在 BM 维度
            int a_tile_col = idx % BK; // 在 BK 维度
            
            float val = 0.0f;
            // 边界检查
            if (row_start + a_tile_row < M && k_offset + a_tile_col < K) {
                val = A[(row_start + a_tile_row) * K + (k_offset + a_tile_col)];
            }
            // 注意存入的索引是 a_tile_col * BM + a_tile_row，实现了转置
            sA[a_tile_col * BM + a_tile_row] = val;
        }

        // 搬运 B 到 sB 采用类似于网格跨步循环的思路
        // 目标：将大小为 [BK][BN] 的数据搬运到 [BK][BN] 的 sB 中
        for (int idx = tid; idx < BK * BN; idx += num_threads) {
            int b_tile_row = idx / BN; // 在 BK 维度
            int b_tile_col = idx % BN; // 在 BN 维度

            float val = 0.0f;
            // 边界检查
            if (k_offset + b_tile_row < K && col_start + b_tile_col < N) {
                val = B[(k_offset + b_tile_row) * N + (col_start + b_tile_col)];
            }
            // 正常存入 无需转置
            sB[b_tile_row * BN + b_tile_col] = val;
        }

        __syncthreads();

        // 采用外积的方式进行计算
        // 这里注意 外积方案中 计算阶段的索引顺序是 K->M->N
        for (int kk = 0; kk < BK; kk++) {
            // 当前线程负责输出 C 中 8x8 的小块
            // 先把所需的 A 的 8 个元素和 B 的 8 个元素读到寄存器里
            float reg_A[8];
            float reg_B[8];
            
            for (int i = 0; i < 8; i++) {
                // 读取 sA 的第 kk 行 (其实是原矩阵 A 的第 kk 列)
                reg_A[i] = sA[kk * BM + ty * 8 + i]; 
            }
            for (int j = 0; j < 8; j++) {
                // 读取 sB 的第 kk 行
                reg_B[j] = sB[kk * BN + tx * 8 + j]; 
            }

            // 计算一个列向量 (8x1) 乘以一个行向量 (1x8) -> 得到 8x8 矩阵，累加到 accum
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    accum[i][j] += reg_A[i] * reg_B[j];
                }
            }
        }
        __syncthreads();
    }

    // 写回阶段 
    // 将 8x8 寄存器中的结果写回全局内存 C
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            int g_r = row_start + ty * 8 + i;
            int g_c = col_start + tx * 8 + j;
            if (g_r < M && g_c < N) {
                C[g_r * N + g_c] = accum[i][j];
            }
        }
    }
}

// 启动器
void solve(torch::Tensor A, torch::Tensor B, torch::Tensor C, 
           int N, int M, int K, int bx, int by, int bk, std::string version) {
    auto d_A = A.data_ptr<float>(); 
    auto d_B = B.data_ptr<float>(); 
    auto d_C = C.data_ptr<float>();
    
    // 配置 Block 线程数：(BN/8, BM/8)
    int total_threads = (bx / 8) * (by / 8);
    dim3 threads(bx / 8, by / 8); 
    dim3 blocks((N + bx - 1) / bx, (M + by - 1) / by);
    
    // 模版分发
    if (bx == 128 && by == 128 && bk == 8) {
        matrix_mul_kernel_outer_product_pure<128, 128, 8><<<blocks, threads>>>(d_A, d_B, d_C, N, M, K);
    } else if (bx == 64 && by == 128 && bk == 8) {
        matrix_mul_kernel_outer_product_pure<128, 64, 8><<<blocks, threads>>>(d_A, d_B, d_C, N, M, K);
    } else if (bx == 128 && by == 64 && bk == 8) {
        matrix_mul_kernel_outer_product_pure<64, 128, 8><<<blocks, threads>>>(d_A, d_B, d_C, N, M, K);
    } else {
        matrix_mul_kernel_outer_product_pure<32, 32, 32><<<blocks, threads>>>(d_A, d_B, d_C, N, M, K);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve", &solve, "Pure Outer Product GEMM",
          py::arg("A"), py::arg("B"), py::arg("C"), 
          py::arg("N"), py::arg("M"), py::arg("K"), 
          py::arg("bx"), py::arg("by"), py::arg("bk"),
          py::arg("version")); 
}