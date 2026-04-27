#include <torch/extension.h>

#ifndef BLOCK_X
#define BLOCK_X 32
#endif

#ifndef BLOCK_Y
#define BLOCK_Y 32
#endif

#define TILE_DIM 32

__global__ void transpose_smem_kernel(const float* __restrict__ input, 
                                               float* __restrict__ output, 
                                               int rows, int cols) 
{
    // transpose 算子的共享内存优化方案 通过共享内存实现读、写操作均能访问合并
    // 为了方便进行 padding 消除银行冲突 我们固定共享内存大小为 32 32 
    // 之后 padding 32 33 能消除银行冲突
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    // 计算当前 Block 负责的 Tile 起始基地址
    int tile_origin_col = blockIdx.x * TILE_DIM;
    int tile_origin_row = blockIdx.y * TILE_DIM;

    // 先搬运数据，为了方便后续调整线程数目的同时保持逻辑正确，这里采用类网格跨步循环的思路
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int total_elements = TILE_DIM * TILE_DIM;
    int stride = blockDim.x * blockDim.y;

    #pragma unroll
    for (int i = tid; i < total_elements; i += stride) {
        // 将一维的偏移量 i 转换回 Tile 内部的二维坐标
        int r = i >> 5; 
        int c = i & 31;

        // 计算在全局内存中的实际坐标 (加上该 Block 处理的 Tile 基地址)
        int global_r = tile_origin_row + r;
        int global_c = tile_origin_col + c;

        if (global_r < rows && global_c < cols) {
            tile[r][c] = input[global_r * cols + global_c];
        }
    }
    __syncthreads();


    // 转置，然后写出数据
    #pragma unroll
    for (int i = tid; i < total_elements; i += stride) {
        int r = i >> 5; 
        int c = i & 31; 

        int out_global_r = tile_origin_col + r; 
        int out_global_c = tile_origin_row + c;

        if (out_global_r < cols && out_global_c < rows) {
            output[out_global_r * rows + out_global_c] = tile[c][r];
        }
    }

}

void solve(torch::Tensor input, torch::Tensor output, int rows, int cols) {
    dim3 threadsPerBlock(BLOCK_X, BLOCK_Y);
    dim3 blocksPerGrid((cols + TILE_DIM - 1) / TILE_DIM,
                       (rows + TILE_DIM - 1) / TILE_DIM);

    transpose_smem_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        rows, 
        cols);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("solve", &solve, "Shared Memory Matrix Transpose",
          py::arg("input"), 
          py::arg("output"), 
          py::arg("rows"), 
          py::arg("cols")); 
}

