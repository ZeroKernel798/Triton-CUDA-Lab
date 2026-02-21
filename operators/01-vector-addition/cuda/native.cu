#include <torch/extension.h> 

__global__ void vector_add_kernel(const float* A, const float* B, float* C, int64_t N) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        C[tid] = A[tid] + B[tid];
    }
}

void solve(torch::Tensor A, torch::Tensor B, torch::Tensor C, int64_t N, int block_size) {
    // 传入block_size作为参数 以便后续设置批量参数来进行调优
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
    namespace py = pybind11;
    m.def("solve", &solve, "Vector Addition Kernel (Pybind11)",
          py::arg("A"), py::arg("B"), py::arg("C"), py::arg("N"), 
          py::arg("block_size")); // 显式命名
}