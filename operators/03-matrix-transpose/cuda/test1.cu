#include <cuda_runtime.h>

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) 
{
    // 利用了共享内存 实现了读写过程的访问合并 但受限于线程块大小 无法达到128字节的高效读取
    __shared__ float data[16][17];

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

extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16); 
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
