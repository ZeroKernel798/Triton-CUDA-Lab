#include <cuda_runtime.h>


__global__ void matrix_add(const float* A, const float* B, float* C, int N) 
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if(x < N && y < N){
        C[y * N + x] = A[y * N + x] + B[y * N + x];
    }
}


// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N+15)/16, (N+15)/16);

    matrix_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    // cudaDeviceSynchronize();
}
