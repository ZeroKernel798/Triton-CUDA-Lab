#include <torch/extension.h>

#ifndef BLOCK_X
#define BLOCK_X 32
#endif

#ifndef BLOCK_Y
#define BLOCK_Y 8
#endif

#define TILE_DIM 32

// XOR Swizzle 原理（以 float4 粒度操作）:
//
// Padding 版（smem_float4.cu）：tile[32][33]
//   bank(row, col) = (row * 33 + col) % 32 = (row + col) % 32
//   → 行号参与 bank 计算，同列不同行线程访问不同 bank → 0-way 冲突
//   代价：每行浪费 1 个 float = 32 float 共享内存（128 bytes/block）
//
// Swizzle 版（本文件）：tile[32][32]（无浪费）
//   写入时：phys_f4_col = logical_f4_col XOR (row & 7)
//   读取时：phys_f4_col = logical_f4_col XOR (row & 7)（保持一致）
//   → 把 32-way 冲突降为 4-way（32 行中每 8 行 XOR 模式相同，形成 4 组）
//   收益：tile 更小，L1/共享内存利用率更高，可能多塞一个 block/SM

__global__ void transpose_smem_float4_swizzle_kernel(const float* __restrict__ input, 
                                                     float* __restrict__ output, 
                                                     int rows, int cols) 
{
    // 无 padding，依赖 swizzle 减少 bank conflict
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int tile_origin_col = blockIdx.x * TILE_DIM;
    int tile_origin_row = blockIdx.y * TILE_DIM;

    int tid    = threadIdx.y * blockDim.x + threadIdx.x;
    int stride = blockDim.x * blockDim.y;

    bool full_tile = (tile_origin_row + TILE_DIM <= rows) &&
                     (tile_origin_col + TILE_DIM <= cols) &&
                     ((rows & 3) == 0) && ((cols & 3) == 0);

    if (full_tile) {
        int total_f4 = (TILE_DIM * TILE_DIM) >> 2;  // 256

        // ---- float4 Load: global 读合并，写入 swizzled 共享内存位置 ----
        // i -> (r, cg): r = i/8 (0..31), cg = i%8 (0..7)
        // phys_cg = cg XOR (r & 7)：行号低 3 位参与 XOR，打散同列访问的 bank
        #pragma unroll
        for (int i = tid; i < total_f4; i += stride) {
            int r  = i >> 3;
            int cg = i & 7;

            float4 v = *reinterpret_cast<const float4*>(
                &input[(tile_origin_row + r) * cols + tile_origin_col + cg * 4]);

            int phys_cg = cg ^ (r & 7);
            *reinterpret_cast<float4*>(&tile[r][phys_cg * 4]) = v;
        }
        __syncthreads();

        // ---- float4 Store: 从 swizzled 位置读，写回 global（连续 4 列，合并写）----
        // i -> (rg, c): rg = i/32 (0..7), c = i%32 (0..31), r = rg*4
        // 读 tile[c] 的第 rg 个 float4 组，物理列 = rg XOR (c & 7)
        #pragma unroll
        for (int i = tid; i < total_f4; i += stride) {
            int rg = i >> 5;
            int c  = i & 31;
            int r  = rg << 2;

            int phys_rg = rg ^ (c & 7);
            float4 v = *reinterpret_cast<float4*>(&tile[c][phys_rg * 4]);

            *reinterpret_cast<float4*>(
                &output[(tile_origin_col + c) * rows + tile_origin_row + r]) = v;
        }
    } else {
        // ---- Scalar fallback：边界 tile 或维度非 4 对齐，与 smem.cu 相同逻辑 ----
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

    transpose_smem_float4_swizzle_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        rows, 
        cols);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("solve", &solve, "Shared Memory Matrix Transpose With Float4 + XOR Swizzle",
          py::arg("input"), 
          py::arg("output"), 
          py::arg("rows"), 
          py::arg("cols")); 
}
