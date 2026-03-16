#include <cuda_runtime.h>
#include <torch/extension.h>

// 宏由编译时注入，此处设定默认值防止 IDE 报错
#ifndef BLOCK_X
#define BLOCK_X 16
#endif

#ifndef BLOCK_Y
#define BLOCK_Y 16
#endif

__global__ void matrix_transpose_native_kernel(const float* __restrict__ input, 
                                              float* __restrict__ output, 
                                              int rows, int cols) 
{
    // 直接使用宏，编译器会将其视为常量，优化效果更好
    int x = blockIdx.x * BLOCK_X + threadIdx.x;
    int y = blockIdx.y * BLOCK_Y + threadIdx.y;

    if (x < cols && y < rows) {
        output[x * rows + y] = input[y * cols + x];
    }
}

void solve(torch::Tensor input, torch::Tensor output, int rows, int cols) {
    auto device = input.device();
    cudaSetDevice(device.index());

    const float* d_input = input.data_ptr<float>();
    float* d_output = output.data_ptr<float>();

    // 线程块大小直接使用宏定义
    dim3 threadsPerBlock(BLOCK_X, BLOCK_Y);
    dim3 blocksPerGrid((cols + BLOCK_X - 1) / BLOCK_X,
                       (rows + BLOCK_Y - 1) / BLOCK_Y);

    matrix_transpose_native_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, rows, cols);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("solve", &solve, "Matrix Transpose",
          py::arg("input"), 
          py::arg("output"), 
          py::arg("rows"), 
          py::arg("cols")); 
}