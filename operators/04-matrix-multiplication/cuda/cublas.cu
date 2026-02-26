#include <torch/extension.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// 移除 #include <THC/THC.h>
#include <ATen/cuda/CUDAContext.h> // 现代 PyTorch 获取 cuBLAS 句柄的头文件
#include <ATen/cuda/Exceptions.h>

void solve(torch::Tensor A, torch::Tensor B, torch::Tensor C, int N, int M, int K) {
    // cuBLAS 默认是 列优先 (Column Major)，而 C++ Tensor 是 行优先 (Row Major)
    // 技巧：计算 C = A * B (Row Major) 等价于计算 C^T = B^T * A^T (Column Major)
    // 所以传参顺序会变成 B, A
    
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    
    const float alpha = 1.0f;
    const float beta = 0.0f;

    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    // cublasSgemm 参数说明:
    // handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc
    // 由于 Row-Major 转 Col-Major，逻辑如下：
    // 我们想算 C(M,N) = A(M,K) * B(K,N)
    // 传入 cuBLAS: 算 C(N,M) = B(N,K) * A(K,M)
    
    cublasStatus_t status = cublasSgemm(
        handle, 
        CUBLAS_OP_N,   // B 不转置
        CUBLAS_OP_N,   // A 不转置
        N,             // 结果矩阵的行 (对应 B 的列/C 的列)
        M,             // 结果矩阵的列 (对应 A 的行/C 的行)
        K,             // 共享维度
        &alpha, 
        d_B, N,        // B 作为左矩阵，领先维度为 N
        d_A, K,        // A 作为右矩阵，领先维度为 K
        &beta, 
        d_C, N         // C 结果，领先维度为 N
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        AT_ERROR("cuBLAS SGEMM failed with status: ", status);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // 注意：cuBLAS 版本不需要 bx, by 参数，因为内部自动调优
    m.def("solve", &solve, "cuBLAS Matrix Multiplication",
          py::arg("A"), py::arg("B"), py::arg("C"), 
          py::arg("N"), py::arg("M"), py::arg("K"));
}