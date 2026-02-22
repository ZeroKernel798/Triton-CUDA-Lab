
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) 
{
    // 实现矩阵转置的核函数 朴素版本
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < cols && y < rows) {
        int input_idx = y * cols + x; // 原矩阵的索引
        int output_idx = x * rows + y; // 转置矩阵的索引
        output[output_idx] = input[input_idx];
    }
}

void solve(torch::Tensor input, torch::Tensor output, int rows, int cols, int bx, int by) {
    // 1. 必要的安全检查（这是框架的修养）
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "Output must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    const float* d_input = input.data_ptr<float>();
    float* d_output = output.data_ptr<float>();

    dim3 threadsPerBlock(bx, by);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, rows, cols);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("solve", &solve, "2D Matrix Transpose",
          py::arg("input"), py::arg("output"), py::arg("rows"), py::arg("cols"), py::arg("bx"), py::arg("by")); // 显式命名
}