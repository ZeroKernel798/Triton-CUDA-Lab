#include <torch/extension.h>

// 这里设置一个默认值 防止外部未能传入宏 导致报错
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

// 输入指针用__restrict__维护 暗示编译器 提高效率
__global__ void vector_add_kernel(const float* __restrict__ A, 
                                         const float* __restrict__ B, 
                                         float* __restrict__ C, 
                                         int N) {
    // 朴素向量加法的实现 
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N)
        C[tid] = A[tid] + B[tid];
}

void solve(torch::Tensor A, torch::Tensor B, torch::Tensor C, int N) {
    // 设置线程块和网格块的规模
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 将torch映射成指针，并发射核函数
    vector_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve", &solve, "Native Vector Addition Kernel",
          py::arg("A"), py::arg("B"), py::arg("C"), py::arg("N"));
}