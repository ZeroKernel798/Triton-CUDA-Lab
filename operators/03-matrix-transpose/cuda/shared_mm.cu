#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) 
{
    // 利用了共享内存 实现了读写过程的访问合并
    __shared__ float data[32][33];

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    // 先取数据
    if(x < cols && y < rows){
        data[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }

    // 操作共享内存 必须做线程块内的线程同步
    __syncthreads();

    int x_new = blockDim.y * blockIdx.y + threadIdx.x;
    int y_new = blockDim.x * blockIdx.x + threadIdx.y;
    if(x_new < rows && y_new < cols){
        output[y_new * rows  + x_new] = data[threadIdx.x][threadIdx.y];
    }
}

void solve(torch::Tensor input, torch::Tensor output, int rows, int cols, int bx, int by){
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