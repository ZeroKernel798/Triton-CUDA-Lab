#include <torch/extension.h>


#ifndef BLOCK_X
#define BLOCK_X 256
#endif

__global__ void float4_gemv_kernel(
    const float* __restrict__ x,
    const float* __restrict__ A,
    float* __restrict__ y,
    int K,
    int N
) {
    // 使用 float4 优化 gemv: 沿 K 向量化读 x，沿 N 向量化写 y（读 A 也顺带向量化）。
    // y = x @ A, 左为向量 x (1, K), 右为矩阵 A (K, N), 输出 y (1, N)。
    // 每个线程负责 4 个相邻输出列 col..col+3。假设 K、N 均为 4 的倍数（对齐数据）。
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if(col >= N) return;

    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    // 一次 float4 读 x[i..i+3]，对应 4 行 A 各做一次 float4 读，累加到 4 个列累加器。
    for(int i = 0; i < K; i += 4){
        float4 xv = *reinterpret_cast<const float4*>(x + i);
        float xs[4] = {xv.x, xv.y, xv.z, xv.w};
        #pragma unroll
        for(int k = 0; k < 4; ++k){
            float4 a = *reinterpret_cast<const float4*>(A + (i + k) * N + col);
            acc.x += xs[k] * a.x;
            acc.y += xs[k] * a.y;
            acc.z += xs[k] * a.z;
            acc.w += xs[k] * a.w;
        }
    }

    // float4 写回 y[col..col+3]。
    *reinterpret_cast<float4*>(y + col) = acc;
}


void solve(
    torch::Tensor x,
    torch::Tensor A,
    torch::Tensor y,
    int K,
    int N
) {
    // 每个线程负责 4 列，因此只需 N/4 个线程。
    dim3 threadsPerBlock(BLOCK_X);
    dim3 blocksPerGrid((N / 4 + BLOCK_X - 1) / BLOCK_X);

    float4_gemv_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        x.data_ptr<float>(),
        A.data_ptr<float>(),
        y.data_ptr<float>(),
        K,
        N
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("solve", &solve, "GEMV Float4 Kernel",
          py::arg("x"),
          py::arg("A"),
          py::arg("y"),
          py::arg("K"),
          py::arg("N"));
}
