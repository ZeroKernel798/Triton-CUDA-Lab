#include <cuda_runtime.h>
#include <torch/extension.h>

#ifndef BLOCK_X
#define BLOCK_X 16
#endif

#ifndef BLOCK_Y
#define BLOCK_Y 16
#endif

__global__ void matrix_add_kernel_vec(const float* A, const float* B, float* C, int N) 
{
    // x_base 是处理 float4 的线程索引
    // blockDim.x 在运行时会自动等于 BLOCK_X
    int x_base = blockDim.x * blockIdx.x + threadIdx.x;
    int x = x_base * 4;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(x >= N || y >= N) return;

    // 16 字节对齐检查
    // 只有当起始地址是 16 字节(4个float)的倍数，且剩余元素足够 4 个时，才使用 float4
    if((x + 3) < N && ((y * N + x) % 4 == 0)){
        const float4* A_ptr = reinterpret_cast<const float4*>(&(A[y * N + x]));
        const float4* B_ptr = reinterpret_cast<const float4*>(&(B[y * N + x]));
        float4* C_ptr = reinterpret_cast<float4*>(&(C[y * N + x]));

        // LDG.E.128 指令：一次读取 128 bit
        float4 a = *A_ptr;
        float4 b = *B_ptr;
        float4 c;

        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;

        // STG.E.128 指令：一次写入 128 bit
        *C_ptr = c;
    }
    else {
        // Fallback: 处理末尾对齐不足的部分
        for(int i = x; i < x + 4 && i < N; i++){
            C[y * N + i] = A[y * N + i] + B[y * N + i];
        }
    }
}

void solve(torch::Tensor A, torch::Tensor B, torch::Tensor C, int N) {
    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    // 一个线程负责 4 个 float，逻辑网格宽度缩小 4 倍
    int n_vec = (N + 3) / 4; 
    
    // 使用编译时宏
    dim3 threadsPerBlock(BLOCK_X, BLOCK_Y);
    dim3 blocksPerGrid((n_vec + BLOCK_X - 1) / BLOCK_X, (N + BLOCK_Y - 1) / BLOCK_Y);

    matrix_add_kernel_vec<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("solve", &solve, "2D Vectorized Matrix Addition (float4)",
          py::arg("A"), py::arg("B"), py::arg("C"), py::arg("N"));
}