#include <cuda_runtime.h>
#include <cstdio>

// 使用模板参数：TILE_DIM 是共享内存块的大小，BLOCK_ROWS 是线程块的行数
template <int TILE_DIM, int BLOCK_ROWS>
__global__ void matrix_transpose_template_kernel(const float* input, float* output, int rows, int cols) 
{
    // +1 依然是为了消除 Bank Conflict
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 计算当前线程在全局矩阵中的起始位置
    int x = blockIdx.x * TILE_DIM + tx;
    int y = blockIdx.y * TILE_DIM + ty;

    // --- 1. 读取阶段 ---
    // 编译器看到 TILE_DIM / BLOCK_ROWS 是常量，会直接展开循环
    #pragma unroll
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (x < cols && (y + i) < rows) {
            tile[ty + i][tx] = input[(y + i) * cols + x];
        }
    }

    __syncthreads();

    // --- 2. 写入阶段 (转置) ---
    // 这里的逻辑依然保持：让 tx 对应输出矩阵的连续列，实现写合并
    int x_new = blockIdx.y * TILE_DIM + tx;
    int y_new = blockIdx.x * TILE_DIM + ty;

    #pragma unroll
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (x_new < rows && (y_new + i) < cols) {
            output[(y_new + i) * rows + x_new] = tile[tx][ty + i];
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    // 可以测试多组参数，不同gpu最优可能不同
    const int TILE_DIM = 32;
    const int BLOCK_ROWS = 8;

    dim3 threadsPerBlock(TILE_DIM, BLOCK_ROWS); 
    // 注意：grid 的计算要和 TILE_DIM 一致
    dim3 blocksPerGrid((cols + TILE_DIM - 1) / TILE_DIM,
                       (rows + TILE_DIM - 1) / TILE_DIM);

    matrix_transpose_template_kernel<TILE_DIM, BLOCK_ROWS><<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    
    // cudaDeviceSynchronize(); 
}