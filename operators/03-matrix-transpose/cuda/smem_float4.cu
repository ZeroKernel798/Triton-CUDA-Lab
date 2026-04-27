#include <torch/extension.h>

#ifndef BLOCK_X
#define BLOCK_X 32
#endif

#ifndef BLOCK_Y
#define BLOCK_Y 32
#endif

#define TILE_DIM 32

__global__ void transpose_smem_float4_kernel(const float* __restrict__ input, 
                                             float* __restrict__ output, 
                                             int rows, int cols) 
{
    // 共享内存 + padding 消除 bank conflict，与 smem.cu 相同
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int tile_origin_col = blockIdx.x * TILE_DIM;
    int tile_origin_row = blockIdx.y * TILE_DIM;

    int tid    = threadIdx.y * blockDim.x + threadIdx.x;
    int stride = blockDim.x * blockDim.y;

    // float4 路径要求：tile 完整（不是边界块）且行列均为 4 的倍数（保证地址对齐）
    bool full_tile = (tile_origin_row + TILE_DIM <= rows) &&
                     (tile_origin_col + TILE_DIM <= cols) &&
                     ((rows & 3) == 0) && ((cols & 3) == 0);

    if (full_tile) {
        // ---- float4 Load: 每次读 4 个连续列（同行），共 256 次 float4 ----
        // i -> (r, cg): r = i/8 (0..31), cg = i%8 (0..7), c = cg*4 (0,4,...,28)
        int total_f4 = (TILE_DIM * TILE_DIM) >> 2;  // 256
        #pragma unroll
        for (int i = tid; i < total_f4; i += stride) {
            int r  = i >> 3;
            int cg = i & 7;
            int c  = cg << 2;
            float4 v = *reinterpret_cast<const float4*>(
                &input[(tile_origin_row + r) * cols + tile_origin_col + c]);
            tile[r][c]     = v.x;
            tile[r][c + 1] = v.y;
            tile[r][c + 2] = v.z;
            tile[r][c + 3] = v.w;
        }
        __syncthreads();

        // ---- float4 Store: 读 tile 同一列的 4 个连续行，写到输出的 4 个连续列 ----
        // i -> (rg, c): rg = i/32 (0..7), c = i%32 (0..31), r = rg*4 (0,4,...,28)
        #pragma unroll
        for (int i = tid; i < total_f4; i += stride) {
            int rg = i >> 5;
            int c  = i & 31;
            int r  = rg << 2;
            float4 v = {tile[c][r], tile[c][r + 1], tile[c][r + 2], tile[c][r + 3]};
            *reinterpret_cast<float4*>(
                &output[(tile_origin_col + c) * rows + tile_origin_row + r]) = v;
        }
    } else {
        // ---- Scalar fallback：边界 tile 或维度非 4 对齐，逻辑与 smem.cu 完全相同 ----
        int total_elements = TILE_DIM * TILE_DIM;
        #pragma unroll
        for (int i = tid; i < total_elements; i += stride) {
            int r = i >> 5;
            int c = i & 31;
            int global_r = tile_origin_row + r;
            int global_c = tile_origin_col + c;
            if (global_r < rows && global_c < cols) {
                tile[r][c] = input[global_r * cols + global_c];
            }
        }
        __syncthreads();

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
}

void solve(torch::Tensor input, torch::Tensor output, int rows, int cols) {
    dim3 threadsPerBlock(BLOCK_X, BLOCK_Y);
    dim3 blocksPerGrid((cols + TILE_DIM - 1) / TILE_DIM,
                       (rows + TILE_DIM - 1) / TILE_DIM);

    transpose_smem_float4_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        rows, 
        cols);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("solve", &solve, "Shared Memory Matrix Transpose With Float4",
          py::arg("input"), 
          py::arg("output"), 
          py::arg("rows"), 
          py::arg("cols")); 
}
