#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void matrix_mul_kernel(const float* A, const float* B, float* C, int N, int M, int K) 
{
    // 朴素矩阵乘法 A*B = C   A:MK B:KN C:MN
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if(row < M && col < N){
        float sum = 0.0f; 
        for(int i = 0; i < K; ++i){
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum; 
    }
}

void solve(torch::Tensor A, torch::Tensor B, torch::Tensor C, int N, int M, int K, int bx, int by) {
    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    // 朴素版本逻辑：一个线程算一个点
    dim3 threadsPerBlock(bx, by);
    dim3 blocksPerGrid((N + bx - 1) / bx, (M + by - 1) / by); 

    // 调用最开始那个不带共享内存的朴素 Kernel
    matrix_mul_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N, M, K);
  
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("solve", &solve, "2D Matrix Multiplication",
          py::arg("A"), py::arg("B"), py::arg("C"), py::arg("N"), 
          py::arg("M"), py::arg("K"), py::arg("bx"), py::arg("by")); 
}