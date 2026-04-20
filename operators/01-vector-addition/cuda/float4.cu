#include <torch/extension.h>
#include <cuda_runtime.h>

// 这里设置一个默认值 防止外部未能传入宏 导致报错
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

__global__ void vector_add_float4_kernel(const float* __restrict__ A, 
                                         const float* __restrict__ B, 
                                         float* __restrict__ C, 
                                         int N) {
    // float4 优化的向量加法核函数
    // 注意这个地方我们每个线程处理的一个 float4 向量
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 指针转换 方便进行操作
    const float4* A4 = reinterpret_cast<const float4*>(A);
    const float4* B4 = reinterpret_cast<const float4*>(B);
    float4* C4 = reinterpret_cast<float4*>(C);

    if(tid * 4 + 3 < N){
        // 向量读取 提高效率
        float4 AData = A4[tid];
        float4 BData = B4[tid];
        float4 CData;

        CData.x = AData.x + BData.x;
        CData.y = AData.y + BData.y;
        CData.z = AData.z + BData.z;
        CData.w = AData.w + BData.w;

        C4[tid] = CData;
    }
    else{
       // 处理尾部的几个数据
       for(int i = 0; i < 4; ++i){
            int idx = tid * 4 + i;
            if(idx < N)
                C[idx] = A[idx] + B[idx];

       }
    }
}

void solve(torch::Tensor A, torch::Tensor B, torch::Tensor C, int N) {
    // 计算的是用 float4 读取所需要的向量个数
    int nVectors = (N + 4 - 1) / 4; 
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (nVectors + threadsPerBlock - 1) / threadsPerBlock;

    // 启动核函数
    vector_add_float4_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve", &solve, "Float4 Vector Addition Kernel",
          py::arg("A"), py::arg("B"), py::arg("C"), py::arg("N"));
}