#include <cuda_runtime.h>
#include <torch/extension.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

__global__ void matrix_add_vec_kernel(const float* A, const float* B, float* C, int Ne) {
    // 这是 1D Flattened 之后，使用 float4 优化的矩阵加法核函数
    // 注意一个线程处理的是一个 float4 的向量元素
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    const float4* A4 = reinterpret_cast<const float4*>(A);
    const float4* B4 = reinterpret_cast<const float4*>(B);
    float4* C4 = reinterpret_cast<float4*>(C);

    if(tid * 4 + 3 < Ne){
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
        // 退化成单线程操作
        for(int i = 0; i < 4; ++i){
            int idx = tid * 4 + i;
            if(idx < Ne)
                C[idx] = A[idx] + B[idx];
        }
    }
}


void solve(torch::Tensor A, torch::Tensor B, torch::Tensor C, int N) {
    int Ne = N * N;
    int nVectors = (Ne + 3) / 4;
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (nVectors + threadsPerBlock - 1) / threadsPerBlock;

    // 核函数下发的过程
    matrix_add_vec_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        Ne
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("solve", &solve, "1D Flattened Vectorized Matrix Addition",
          py::arg("A"), py::arg("B"), py::arg("C"), py::arg("N"));
}