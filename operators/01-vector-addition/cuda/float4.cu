#include <torch/extension.h>
#include <cuda_runtime.h>

// 1. 宏定义，InferX 框架会在编译时通过 -DBLOCK_SIZE=xxx 注入
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

__global__ void vector_add_float4_kernel(const float* __restrict__ A, 
                                         const float* __restrict__ B, 
                                         float* __restrict__ C, 
                                         int64_t N) {
    // 🌟 使用宏 BLOCK_SIZE 代替 blockDim.x，方便编译器做循环展开和寄存器预分配
    int64_t tid = (int64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    // 每个线程处理 4 个连续的 float
    int64_t offset = tid * 4;

    // 2. 向量化处理部分 (128-bit Load/Store)
    if (offset + 3 < N) {
        // 使用内置 float4 类型实现合并访存
        float4 a = *reinterpret_cast<const float4*>(&A[offset]);
        float4 b = *reinterpret_cast<const float4*>(&B[offset]);
        float4 res;

        res.x = a.x + b.x;
        res.y = a.y + b.y;
        res.z = a.z + b.z;
        res.w = a.w + b.w;

        *reinterpret_cast<float4*>(&C[offset]) = res;
    } 
    // 3. 边界残余处理 (Tail Handling)
    // 只有最后一组活跃线程中的部分线程会进入这里
    else if (offset < N) {
        for (int64_t i = offset; i < N; ++i) {
            C[i] = A[i] + B[i];
        }
    }
}

void solve(torch::Tensor A, torch::Tensor B, torch::Tensor C, int64_t N) {
    // 线程块大小直接写死为宏常量
    const int threadsPerBlock = BLOCK_SIZE; 
    
    // 计算总共需要多少个“向量化步长”
    // 计算公式: ceil(N / 4)
    int64_t n_vector = (N + 3) / 4; 
    int64_t blocksPerGrid = (n_vector + threadsPerBlock - 1) / threadsPerBlock;

    vector_add_float4_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve", &solve, "Vector Addition float4 (Macro Optimized)",
          py::arg("A"), py::arg("B"), py::arg("C"), py::arg("N"));
}