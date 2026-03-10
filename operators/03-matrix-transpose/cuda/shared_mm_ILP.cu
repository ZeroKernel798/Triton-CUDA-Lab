#include <torch/extension.h>
#include <cuda_runtime.h>

template <int BLOCK_ROWS>
__global__ void matrix_transpose_kernel(const float* __restrict__ input, 
                                       float* __restrict__ output, 
                                       int rows, int cols) 
{
    const int TILE_DIM = 32;
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int tx = threadIdx.x; // 0-31
    int ty = threadIdx.y; // 0-(BLOCK_ROWS-1)

    int x = blockIdx.x * TILE_DIM + tx;
    int y = blockIdx.y * TILE_DIM + ty;

    // 读取阶段
    #pragma unroll
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (x < cols && (y + i) < rows) {
            tile[ty + i][tx] = input[(y + i) * cols + x];
        }
    }

    __syncthreads();

    // 写入阶段 
    int x_new = blockIdx.y * TILE_DIM + tx;
    int y_new = blockIdx.x * TILE_DIM + ty;

    #pragma unroll
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (x_new < rows && (y_new + i) < cols) {
            output[(y_new + i) * rows + x_new] = tile[tx][ty + i];
        }
    }
}

// 修正后的 solve 函数
void solve(torch::Tensor input, torch::Tensor output, int rows, int cols, int bx, int by) {
    // 强制 bx 必须为 32
    const int TILE_DIM = 32;

    const float* d_input = input.data_ptr<float>();
    float* d_output = output.data_ptr<float>();
    
    dim3 threadsPerBlock(32, by); 
    dim3 blocksPerGrid((cols + TILE_DIM - 1) / TILE_DIM,
                       (rows + TILE_DIM - 1) / TILE_DIM);

    // 只需要根据 by (BLOCK_ROWS) 进行分发
    switch (by) {
        case 8:  
            matrix_transpose_kernel<8><<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, rows, cols); 
            break;
        case 16: 
            matrix_transpose_kernel<16><<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, rows, cols); 
            break;
        case 32: 
            matrix_transpose_kernel<32><<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, rows, cols); 
            break;
        default: 
            AT_ERROR("Unsupported tuning config by=", by);
    }
}

// Pybind11 绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve", &solve, "Matrix Transpose with Shared Memory and Tiling",
          py::arg("input"), py::arg("output"), py::arg("rows"), py::arg("cols"), 
          py::arg("bx"), py::arg("by"));
}