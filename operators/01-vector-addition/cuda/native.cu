#include <torch/extension.h>

// 1. 定义默认宏（防止框架没传参数时报错）
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

__global__ void vector_add_kernel(const float* A, const float* B, float* C, int64_t N) {
    // 这里的 BLOCK_SIZE 是宏，编译时直接替换为数字
    int64_t tid = (int64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (tid < N) {
        C[tid] = A[tid] + B[tid];
    }
}

// 2. solve 函数签名变干净了，不再需要 block_size 参数
void solve(torch::Tensor A, torch::Tensor B, torch::Tensor C, int64_t N) {
    // 线程块大小直接使用宏
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    vector_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        C.data_ptr<float>(), 
        N
    );
}

// 3. 绑定也变简单了
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve", &solve, "Vector Addition Kernel (Macro Optimized)",
          py::arg("A"), py::arg("B"), py::arg("C"), py::arg("N"));
}