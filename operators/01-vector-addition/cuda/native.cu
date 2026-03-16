#include <cuda_runtime.h>
#include <torch/extension.h>

#ifndef BLOCK_X
#define BLOCK_X 16
#endif

#ifndef BLOCK_Y
#define BLOCK_Y 16
#endif

__global__ void matrix_add_kernel(const float* A, const float* B, float* C, int N) 
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    
    if(x < N && y < N){
        C[y * N + x] = A[y * N + x] + B[y * N + x];
    }
}

void solve(torch::Tensor A, torch::Tensor B, torch::Tensor C, int N) {
    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    // 使用编译宏来确定线程配置参数
    dim3 threadsPerBlock(BLOCK_X, BLOCK_Y);
    dim3 blocksPerGrid((N + BLOCK_X - 1) / BLOCK_X, (N + BLOCK_Y - 1) / BLOCK_Y);

    matrix_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("solve", &solve, "2D Matrix Addition (JIT Optimized)",
          py::arg("A"), py::arg("B"), py::arg("C"), py::arg("N"));
}