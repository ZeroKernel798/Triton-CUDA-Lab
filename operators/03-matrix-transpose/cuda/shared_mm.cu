#include <cuda_runtime.h>
#include <torch/extension.h>

#ifndef BLOCK_X
#define BLOCK_X 32
#endif

#ifndef BLOCK_Y
#define BLOCK_Y 32
#endif

#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void matrix_transpose_shared_kernel(const float* __restrict__ input, 
                                               float* __restrict__ output, 
                                               int rows, int cols) 
{
    __shared__ float data[32][33];

    // 读取数据到共享内存 (Global -> Shared)
    // 此时读操作在 Global Memory 是合并的
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < cols && y < rows) {
        data[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }

    // 必须同步，确保 Tile 数据全部加载完成
    __syncthreads();

    // 将数据写回全局内存 (Shared -> Global)
    // 通过交换索引，使得写操作在 Global Memory 也是合并的
    // 计算转置后的新坐标
    int x_new = blockDim.y * blockIdx.y + threadIdx.x;
    int y_new = blockDim.x * blockIdx.x + threadIdx.y;

    if (x_new < rows && y_new < cols) {
        // 在写回时，我们读取共享内存是按列读，但写 Global 是按行写
        output[y_new * rows + x_new] = data[threadIdx.x][threadIdx.y];
    }
}

void solve(torch::Tensor input, torch::Tensor output, int rows, int cols) {
    auto device = input.device();
    cudaSetDevice(device.index());

    const float* d_input = input.data_ptr<float>();
    float* d_output = output.data_ptr<float>();

    // 直接使用宏定义的维度
    dim3 threadsPerBlock(BLOCK_X, BLOCK_Y);
    dim3 blocksPerGrid((cols + BLOCK_X - 1) / BLOCK_X,
                       (rows + BLOCK_Y - 1) / BLOCK_Y);

    matrix_transpose_shared_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, rows, cols);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("solve", &solve, "Shared Memory Matrix Transpose (Macro Version)",
          py::arg("input"), 
          py::arg("output"), 
          py::arg("rows"), 
          py::arg("cols")); 
}

