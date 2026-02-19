#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void matrix_add_kernel(const float* A, const float* B, float* C, int N) 
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    
    if(x < N && y < N){
        C[y * N + x] = A[y * N + x] + B[y * N + x];
    }
}

void solve(torch::Tensor A, torch::Tensor B, torch::Tensor C, int N, int block_size) {
    // 获取设备指针
    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    // 计算网格规模
    int block_size_1d = std::sqrt(block_size);
    dim3 threadsPerBlock(block_size_1d, block_size_1d);
    dim3 blocksPerGrid((N + block_size_1d - 1) / block_size_1d, (N + block_size_1d - 1) / block_size_1d);

    // 启动内核
    matrix_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve", &solve, "Matrix Addition (CUDA)");
}