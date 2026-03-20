#include <cuda_runtime.h>
#include <torch/extension.h>
#include <string>


#ifndef BLOCK_X
#define BLOCK_X 32
#endif
#ifndef BLOCK_Y
#define BLOCK_Y 32
#endif
#ifndef BLOCK_K
#define BLOCK_K 32
#endif

template<int BX, int BY, int BK>
__global__ void matrix_mul_kernel_tiled(const float* A, const float* B, float* C, int N, int M, int K) 
{
    __shared__ float sA[BY][BK];
    __shared__ float sB[BK][BX];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * BY + ty;
    int col = blockIdx.x * BX + tx;

    float sum = 0.0f;

    for(int k_offset = 0; k_offset < K; k_offset += BK){
        // 协作搬运 A 到共享内存
        #pragma unroll
        for(int i = ty * blockDim.x + tx; i < BY * BK; i += blockDim.x * blockDim.y){
            int tile_row = i / BK;
            int tile_col = i % BK;
            int g_r = tile_row + blockIdx.y * BY;
            int g_c = tile_col + k_offset;
            sA[tile_row][tile_col] = (g_r < M && g_c < K) ? A[g_r * K + g_c] : 0.0f;
        }
        
        // 协作搬运 B 到共享内存
        #pragma unroll
        for(int i = ty * blockDim.x + tx; i < BK * BX; i += blockDim.x * blockDim.y){
            int tile_row = i / BX;
            int tile_col = i % BX;
            int g_r = tile_row + k_offset;
            int g_c = tile_col + blockIdx.x * BX;
            sB[tile_row][tile_col] = (g_r < K && g_c < N) ? B[g_r * N + g_c] : 0.0f;
        }

        __syncthreads(); 

        // 计算当前 Tile 的部分和
        #pragma unroll
        for(int i = 0; i < BK; i++) {
            sum += sA[ty][i] * sB[i][tx];
        }
        
        __syncthreads(); 
    }

    if(row < M && col < N) C[row * N + col] = sum;
}


void solve(torch::Tensor A, torch::Tensor B, torch::Tensor C, int N, int M, int K) {
    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    // 直接使用宏定义设置 Grid 和 Block
    dim3 threadsPerBlock(BLOCK_X, BLOCK_Y);
    dim3 blocksPerGrid((N + BLOCK_X - 1) / BLOCK_X, (M + BLOCK_Y - 1) / BLOCK_Y); 

    // 直接实例化模板，不再需要复杂的 if-else 分发
    matrix_mul_kernel_tiled<BLOCK_X, BLOCK_Y, BLOCK_K><<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N, M, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    // Python 端调用接口也随之简化
    m.def("solve", &solve, "Tiled Matrix Multiplication (Macro Configured)",
          py::arg("A"), py::arg("B"), py::arg("C"), 
          py::arg("N"), py::arg("M"), py::arg("K")); 
}