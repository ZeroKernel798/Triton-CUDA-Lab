#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void solve(torch::Tensor A, torch::Tensor B, torch::Tensor C, int N) {
    // 1. 获取 PyTorch 当前 Stream 的 cuBLAS 句柄
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    // 2. 设置系数 C = 1.0 * A + 1.0 * B
    const float alpha = 1.0f;
    const float beta = 1.0f;

    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    // 3. 调用 Sgeam
    // 虽然 cuBLAS 是列主序，但对于元素对齐的加法：
    // (N x N) 的行主序矩阵在物理内存上等同于 (N x N) 的列主序矩阵
    // 只要 A, B, C 的形状和步长一致，结果就是正确的
    cublasStatus_t status = cublasSgeam(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N, // A, B 均不转置
        N, N,                     // m, n (矩阵行列数)
        &alpha,
        d_A, N,                   // lda
        &beta,
        d_B, N,                   // ldb
        d_C, N                    // ldc
    );

    // 4. 错误检查
    if (status != CUBLAS_STATUS_SUCCESS) {
        AT_ERROR("cuBLAS Sgeam failed with error code ", status);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("solve", &solve, "cuBLAS Matrix Addition Wrapper",
          py::arg("A"), py::arg("B"), py::arg("C"), py::arg("N")); 
}