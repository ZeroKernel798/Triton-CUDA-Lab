#include <torch/extension.h> 

__global__ void vector_add_kernel(const float* A, const float* B, float* C, int64_t N) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        C[tid] = A[tid] + B[tid];
    }
}

void solve(torch::Tensor A, torch::Tensor B, torch::Tensor C, int64_t N, int block_size) {
    // 注意传入block_size作为参数 以便后续设置批量参数来进行调优
    TORCH_CHECK(A.is_cuda(), "Tensor A must be on CUDA");
    TORCH_CHECK(A.is_contiguous(), "Tensor A must be contiguous");

    int threadsPerBlock = block_size;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    vector_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        C.data_ptr<float>(), 
        N
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve", &solve, "Vector Addition Kernel (Pybind11)");
}