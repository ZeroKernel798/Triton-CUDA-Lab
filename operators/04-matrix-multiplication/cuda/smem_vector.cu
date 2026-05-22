#include <cuda_runtime.h>
#include <torch/extension.h>
#include <string>

// 这里是对每次搬运到共享内存的数据尺寸的定义
#ifndef BM
#define BM 32
#endif
#ifndef BN
#define BN 32
#endif
#ifndef BK
#define BK 32
#endif

// 这里是设置线程块的参数
#ifndef BLOCK_X
#define BLOCK_X 16
#endif
#ifndef BLOCK_Y
#define BLOCK_Y 16
#endif


// 定义宏 方便操作
#define FLOAT4(var) (reinterpret_cast<float4*>(&(var))[0])
#define CFLOAT4(var) (reinterpret_cast<const float4*>(&(var))[0])


__global__ void smem_tiled_float4_sgemm_kernel(const float* A, const float* B, float* C, int N, int M, int K) 
{
    // 基于共享内存 + float4 优化的矩阵乘法
    // 申请分块共享内存
    __shared__ float sA[BM * BK];
    __shared__ float sB[BK * BN];

    // 256个线程解释为 32行 * 8列(每列对应float4)，数据搬运和分块计算，都按照这个来划分
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int tile_row = tid / 8;
    int tile_col = (tid % 8) * 4;

    // 全局矩阵的绝对起始边界
    int row_start = blockIdx.y * BM;
    int col_start = blockIdx.x * BN;

    // 每个线程专属的4个累加器
    float sum[4] = {0.0f};

    // 大 K 维度的 Tiling 循环
    for(int k = 0; k < K; k += BK){
        
        // 向量化搬运数据到 Shared Memory
        // 搬运 A 到 sA 
        if ((row_start + tile_row) < M && (k + tile_col) < K) {
            FLOAT4(sA[tile_row * BK + tile_col]) = CFLOAT4(A[(row_start + tile_row) * K + k + tile_col]);
        } else {
            FLOAT4(sA[tile_row * BK + tile_col]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        // 搬运 B 到 sB
        if ((k + tile_row) < K && (col_start + tile_col) < N) {
            FLOAT4(sB[tile_row * BN + tile_col]) = CFLOAT4(B[(k + tile_row) * N + col_start + tile_col]);
        } else {
            FLOAT4(sB[tile_row * BN + tile_col]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        __syncthreads();

        // 这里是调试和优化的热点区域，sA sB 是否 float4 访问有很大区别
        // 高性能 1x4 内积计算 (外层做 KK 循环)
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            // 寄存器复用核心：读一次 sA 的元素，直接广播给当前线程负责的 4 个计算
            float a_val = sA[tile_row * BK + kk];

            // 完美的连续读取 sB 的 4 个列元素
            sum[0] += a_val * sB[kk * BN + tile_col + 0];
            sum[1] += a_val * sB[kk * BN + tile_col + 1];
            sum[2] += a_val * sB[kk * BN + tile_col + 2];
            sum[3] += a_val * sB[kk * BN + tile_col + 3];
        }

        // 步进同步，防止下一轮搬运覆盖未算完的数据
        __syncthreads();
    }

    // 向量化写回全局内存 C
    int global_c_row = row_start + tile_row;
    int global_c_col = col_start + tile_col;

    if (global_c_row < M && global_c_col < N) {
        FLOAT4(C[global_c_row * N + global_c_col]) = FLOAT4(sum[0]);
    }
}


void solve(torch::Tensor A, torch::Tensor B, torch::Tensor C, int N, int M, int K) {
    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    dim3 threadsPerBlock(BLOCK_X, BLOCK_Y);
    dim3 blocksPerGrid((N + BN - 1) / BN, (M + BM - 1) / BM); 

    smem_tiled_float4_sgemm_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N, M, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("solve", &solve, "SMem Tiled Matrix Multiplication Use Float4",
          py::arg("A"), py::arg("B"), py::arg("C"), 
          py::arg("N"), py::arg("M"), py::arg("K")); 
}