#include <torch/extension.h>

// 这里设置一个默认值 防止外部未能传入宏 导致报错
#ifndef BLOCK_X
#define BLOCK_X 16
#endif

#ifndef BLOCK_Y
#define BLOCK_Y 16
#endif

__global__ void matrix_add_kernel(const float* A, const float* B, float* C, int N) 
{
    // 矩阵计算核函数的朴素实现
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < N && y < N){
        C[y * N + x] = A[y * N + x] + B[y * N + x]; 
    }
}

void solve(torch::Tensor A, torch::Tensor B, torch::Tensor C, int N) {
    // 设置线程块和网格块的规模
    dim3 threadsPerBlock(BLOCK_X, BLOCK_Y);
    dim3 blocksPerGrid((N + BLOCK_X - 1) / BLOCK_X, (N + BLOCK_Y - 1) / BLOCK_Y);

    // 启动核函数
    matrix_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("solve", &solve, "Native 2D Matrix Addition Kernel",
          py::arg("A"), py::arg("B"), py::arg("C"), py::arg("N"));
}