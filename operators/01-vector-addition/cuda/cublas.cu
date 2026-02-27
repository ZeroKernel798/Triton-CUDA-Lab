#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void solve(torch::Tensor A, torch::Tensor B, torch::Tensor C, int64_t N) {
    // 1. 获取 PyTorch 当前 Stream 的 cuBLAS 句柄
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    const float alpha = 1.0f;
    const float beta = 1.0f;

    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    // 2. 调用 Sgeam 实现 C = 1.0 * A + 1.0 * B
    // 将向量视为 N x 1 的矩阵
    cublasStatus_t status = cublasSgeam(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N, 
        N, 1,                     // m = N, n = 1
        &alpha,
        d_A, N,                   // lda = N
        &beta,
        d_B, N,                   // ldb = N
        d_C, N                    // ldc = N
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        AT_ERROR("cuBLAS Sgeam (Vector Add) failed with error code ", status);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("solve", &solve, "cuBLAS Vector Addition Wrapper",
          py::arg("A"), py::arg("B"), py::arg("C"), py::arg("N"));
}