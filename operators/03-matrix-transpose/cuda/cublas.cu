#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h> 
#include <ATen/cuda/Exceptions.h>

void solve(torch::Tensor input, torch::Tensor output, int rows, int cols, int bx, int by) {
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    const float alpha = 1.0f;
    const float beta = 0.0f;

    const float* d_A = input.data_ptr<float>();
    float* d_C = output.data_ptr<float>();

    // =====================================================================
    // 正确的映射逻辑：
    // m = 目标矩阵 C 的行数 (cuBLAS视角下为 rows)
    // n = 目标矩阵 C 的列数 (cuBLAS视角下为 cols)
    // lda = 源矩阵 A 的主维度 (PyTorch 的 cols)
    // ldc = 目标矩阵 C 的主维度 (PyTorch 的 rows)
    // =====================================================================
    cublasStatus_t status = cublasSgeam(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N, 
        rows, cols,               // m, n
        &alpha,
        d_A, cols,                // lda 应该是输入矩阵的列数
        &beta,
        nullptr, rows,            // ldb 随意，因为 beta 是 0
        d_C, rows                 // ldc 应该是输出矩阵的列数（即原矩阵行数）
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        AT_ERROR("cuBLAS Sgeam failed with error code ", status);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("solve", &solve, "cuBLAS Matrix Transpose Wrapper",
          py::arg("input"), 
          py::arg("output"), 
          py::arg("rows"), 
          py::arg("cols"), 
          py::arg("bx"), 
          py::arg("by")); 
}