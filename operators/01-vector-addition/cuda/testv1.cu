#include <torch/extension.h> // 核心头文件：包含了 Pybind11 和张量操作

// 1. Kernel 逻辑保持不变 (可以去掉 extern "C"，Pybind 不依赖它)
__global__ void vector_add_kernel(const float* A, const float* B, float* C, int64_t N) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        C[tid] = A[tid] + B[tid];
    }
}

// 2. 编写包装函数 (不再需要手动在 Python 端搞 data_ptr)
void solve(torch::Tensor A, torch::Tensor B, torch::Tensor C, int64_t N) {
    
    // 安全检查：Pybind 会在报错时给出非常清晰的提示，而不是直接段错误崩溃
    TORCH_CHECK(A.is_cuda(), "Tensor A must be on CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Tensor A must be Float32");
    TORCH_CHECK(A.is_contiguous(), "Tensor A must be contiguous"); // 确保内存连续

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // 关键点：使用 .data_ptr<float>() 直接拿到 PyTorch 内部的显存指针
    vector_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        C.data_ptr<float>(), 
        N
    );
    
    // 注意：如果是为了性能测试，这里通常不需要 cudaDeviceSynchronize
    // 因为 PyTorch 的 Stream 机制会自动管理同步
}

// 3. 定义 Pybind11 模块入口
// TORCH_EXTENSION_NAME 会由 KernelEngine 里的 load 函数自动传入
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve", &solve, "Vector Addition Kernel (Pybind11)");
}