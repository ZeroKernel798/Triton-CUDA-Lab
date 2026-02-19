#include <cuda_runtime.h>
#include <torch/extension.h>

// 1. 内核保持不变，负责 X 方向的向量化处理
__global__ void matrix_add_kernel_vec(const float* A, const float* B, float* C, int N) 
{
    // x_base 是 float4 的索引
    int x_base = blockDim.x * blockIdx.x + threadIdx.x;
    int x = x_base * 4;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(x >= N || y >= N) return;

    // 检查对齐：起始位置必须是 4 的倍数且剩余长度足够
    // (y * N + x) % 4 == 0 保证了 16 字节对齐
    if((x + 3) < N && ((y * N + x) % 4 == 0)){
        const float4* A_ptr = reinterpret_cast<const float4*>(&(A[y * N + x]));
        const float4* B_ptr = reinterpret_cast<const float4*>(&(B[y * N + x]));
        float4* C_ptr = reinterpret_cast<float4*>(&(C[y * N + x]));

        float4 a = *A_ptr;
        float4 b = *B_ptr;
        float4 c;

        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;

        *C_ptr = c;
    }
    else {
        // Fallback: 处理边界或非对齐部分
        for(int i = x; i < x + 4 && i < N; i++){
            C[y * N + i] = A[y * N + i] + B[y * N + i];
        }
    }
}

void solve(torch::Tensor A, torch::Tensor B, torch::Tensor C, int N, int block_size) {
    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    // 一个线程负责 4 个 float
    int n_vec = (N + 3) / 4; 
    
    int block_size_1d = std::sqrt(block_size);
    dim3 threadsPerBlock(block_size_1d, block_size_1d);
    dim3 blocksPerGrid((n_vec + block_size_1d - 1) / block_size_1d, (N + block_size_1d - 1) / block_size_1d);

    matrix_add_kernel_vec<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve", &solve, "Matrix Addition with float4 vectorization");
}