#include <cuda_runtime.h>
#include <torch/extension.h>


#ifndef BLOCK_X
#define BLOCK_X 16
#endif

#ifndef BLOCK_Y
#define BLOCK_Y 16
#endif

__global__ void tranpose_native_kernel(const float* __restrict__ input, 
                                              float* __restrict__ output, 
                                              int rows, int cols) 
{
    // tranpose 算子的朴素实现
    int x = blockIdx.x * BLOCK_X + threadIdx.x;
    int y = blockIdx.y * BLOCK_Y + threadIdx.y;

    if (x < cols && y < rows) {
        output[x * rows + y] = input[y * cols + x];
    }
}

void solve(torch::Tensor input, torch::Tensor output, int rows, int cols) {
    // 我们先设置线程参数
    dim3 threadsPerBlock(BLOCK_X, BLOCK_Y);
    dim3 blocksPerGrid((cols + BLOCK_X - 1) / BLOCK_X, (rows + BLOCK_Y - 1) / BLOCK_Y);

    // 启核函数
    tranpose_native_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("solve", &solve, "Matrix Transpose",
          py::arg("input"), 
          py::arg("output"), 
          py::arg("rows"), 
          py::arg("cols")); 
}