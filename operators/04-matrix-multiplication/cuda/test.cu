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
#define BLOCK_X 32
#endif
#ifndef BLOCK_Y
#define BLOCK_Y 32
#endif

__global__ void smem_tiled_matmul_kernel(const float* A, const float* B, float* C, int N, int M, int K) 
{
    // 基于共享内存的分块矩阵乘法实现
    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];

    // 先搬运数据到共享内存 一个线程搬运一个即可
    float sum = 0.0f;
    for(int k = 0; k < (K + BK - 1) / BK; k++){
        // 先搬运矩阵 A 到共享内存
        int a_row = blockIdx.y * BM + threadIdx.y;
        int a_col = k * BK + threadIdx.x;
        if(a_row < M && a_col < K)
            sA[threadIdx.y][threadIdx.x] = A[a_row * K + a_col];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        
        // 搬运矩阵 B 到共享内存
        int b_row = k * BK + threadIdx.y;
        int b_col = blockIdx.x * BN + threadIdx.x;
        if(b_row < K && b_col < N)
            sB[threadIdx.y][threadIdx.x] = B[b_row * N + b_col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // 计算数据
        for(int i = 0; i < BK; ++i)
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];

        __syncthreads();
    }

    // 把数据输出到矩阵 C
    int row = blockIdx.y * BM + threadIdx.y;
    int col = blockIdx.x * BN + threadIdx.x;
    if(row < M && col < N) C[row * N + col] = sum;
}


void solve(torch::Tensor A, torch::Tensor B, torch::Tensor C, int N, int M, int K) {
    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    dim3 threadsPerBlock(BLOCK_X, BLOCK_Y);
    dim3 blocksPerGrid((N + BLOCK_X - 1) / BLOCK_X, (M + BLOCK_Y - 1) / BLOCK_Y); 

    smem_tiled_matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N, M, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("solve", &solve, "SMem Tiled Matrix Multiplication",
          py::arg("A"), py::arg("B"), py::arg("C"), 
          py::arg("N"), py::arg("M"), py::arg("K")); 
}