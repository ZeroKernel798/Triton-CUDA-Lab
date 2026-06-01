#include <torch/extension.h>


#ifndef BLOCK_X
#define BLOCK_X 256
#endif

__global__ void native_gemv_kernel(
    const float* __restrict__ x,
    const float* __restrict__ A,
    float* __restrict__ y,
    int K,
    int N
) {
    // gemv 算子的朴素实现: y = x @ A, 左为向量 x (1, K), 右为矩阵 A (K, N), 输出 y (1, N)
    // 一个线程负责一个输出列 j，沿 K 行做归约。
    // 同一 warp 内相邻线程访问 A[i*N + j] 是连续地址，天然 coalesced。
    // x/A 标记 const __restrict__：只读且不混叠，编译器可走只读缓存路径并放心做展开。
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(col < N){
        float sum = 0.0f;
        // 部分展开归约循环，提高 ILP 以隐藏 load 延迟。
        #pragma unroll 4
        for(int i = 0; i < K; ++i){
            sum += x[i] * A[i * N + col];
        }
        y[col] = sum;
    }
}


void solve(
    torch::Tensor x,
    torch::Tensor A,
    torch::Tensor y,
    int K,
    int N
) {
    // 这个版本是一个线程负责一个输出列 + 沿 K 行串行归约。
    dim3 threadsPerBlock(BLOCK_X);
    dim3 blocksPerGrid((N + BLOCK_X - 1) / BLOCK_X);

    native_gemv_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        x.data_ptr<float>(),
        A.data_ptr<float>(),
        y.data_ptr<float>(),
        K,
        N
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("solve", &solve, "GEMV Native Kernel",
          py::arg("x"),
          py::arg("A"),
          py::arg("y"),
          py::arg("K"),
          py::arg("N"));
}
