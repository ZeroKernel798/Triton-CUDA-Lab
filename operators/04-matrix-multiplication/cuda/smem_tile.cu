#include <cuda_runtime.h>
#include <torch/extension.h>
#include <string>

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
        // 协作搬运 A
        #pragma unroll
        for(int i = ty * BX + tx; i < BY * BK; i += BX * BY){
            int tile_row = i / BK;
            int tile_col = i % BK;
            int g_r = tile_row + blockIdx.y * BY;
            int g_c = tile_col + k_offset;
            sA[tile_row][tile_col] = (g_r < M && g_c < K) ? A[g_r * K + g_c] : 0.0f;
        }
        // 协作搬运 B
        #pragma unroll
        for(int i = ty * BX + tx; i < BK * BX; i += BX * BY){
            int tile_row = i / BX;
            int tile_col = i % BX;
            int g_r = tile_row + k_offset;
            int g_c = tile_col + blockIdx.x * BX;
            sB[tile_row][tile_col] = (g_r < K && g_c < N) ? B[g_r * N + g_c] : 0.0f;
        }

        __syncthreads(); 

        // 计算阶段
        #pragma unroll
        for(int i = 0; i < BK; i++) sum += sA[ty][i] * sB[i][tx];
        
        __syncthreads(); 
    }

    if(row < M && col < N) C[row * N + col] = sum;
}


void solve(torch::Tensor A, torch::Tensor B, torch::Tensor C, 
           int N, int M, int K, int bx, int by, int bk) {

    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    dim3 threadsPerBlock(bx, by);
    dim3 blocksPerGrid((N + bx - 1) / bx, (M + by - 1) / by); 

    // 🚀 分块版：根据 bx, by, bk 分发模板
    if (bx == 32 && by == 8 && bk == 32) {
        matrix_mul_kernel_tiled<32, 8, 32><<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N, M, K);
    } 
    else if (bx == 32 && by == 16 && bk == 32) {
        matrix_mul_kernel_tiled<32, 16, 32><<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N, M, K);
    }
    else if (bx == 32 && by == 32 && bk == 32) {
        matrix_mul_kernel_tiled<32, 32, 32><<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N, M, K);
    }
    else {
        AT_ERROR("smem_tile: No matched template for the given bx, by, bk.");
    }
    

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve", &solve, "Unified Matrix Multiplication Dispatcher",
          py::arg("A"), py::arg("B"), py::arg("C"), 
          py::arg("N"), py::arg("M"), py::arg("K"), 
          py::arg("bx"), py::arg("by"), py::arg("bk")); 
}