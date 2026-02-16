#include <cuda_runtime.h>
#include <torch/extension.h>

// 1. CUDA Kernel 保持不变
__global__ void matrix_add_kernel(const float* A, const float* B, float* C, int N) 
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    
    if(x < N && y < N){
        C[y * N + x] = A[y * N + x] + B[y * N + x];
    }
}

// 2. Pybind11 包装函数
// 我们不再使用 extern "C"，而是直接定义符合 Pybind11 接口的函数
void solve(torch::Tensor A, torch::Tensor B, torch::Tensor C, int N) {
    // 检查张量是否在 GPU 上以及是否连续
    // 这里为了性能，假设用户在 Python 端已经处理好了
    
    // 获取设备指针
    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    // 计算网格规模
    // 使用 LaTeX 公式理解：$blocks = \lceil N/16 \rceil$
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);

    // 启动内核
    matrix_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    // 注意：框架中的主进程会调用 torch.cuda.synchronize()，所以这里不用强行同步
}

// 3. 定义 Pybind11 模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve", &solve, "Matrix Addition (CUDA)");
}