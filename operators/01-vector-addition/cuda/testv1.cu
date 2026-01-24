#include <cuda_runtime.h>

extern "C" __global__ void vector_add_kernel(const float* A, const float* B, float* C, size_t N) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        C[tid] = A[tid] + B[tid];
    }
}

extern "C" void solve(const float* A, const float* B, float* C, size_t N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    vector_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}