#include <cuda_runtime.h>
#include <torch/extension.h>


#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

#ifndef VEC
#define VEC 1
#endif


__global__ void rmsnorm_native_kernel(
    const float* X,
    const float* gamma,
    float* Y,
    int M,
    int N,
    float eps
) {
    // rmsnorm 算子的朴素实现
    // 基本思路是我们采用一个 warp 或者说 一个 block 负责一行，这样的话可以避免原子加法等，方便进行规约

}


void solve(
    torch::Tensor X,
    torch::Tensor gamma,
    torch::Tensor Y,
    int M,
    int N,
    float eps
) {
    
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("solve", &solve, "RMSNorm Native Kernel",
          py::arg("X"),
          py::arg("gamma"),
          py::arg("Y"),
          py::arg("M"),
          py::arg("N"),
          py::arg("eps"));
}
