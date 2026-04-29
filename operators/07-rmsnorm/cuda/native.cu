#include <torch/extension.h>


#ifndef BLOCK_X
#define BLOCK_X 32
#endif

#ifndef BLOCK_Y
#define BLOCK_Y 16
#endif

__device__ __forceinline__ float warp_sum(float val){
    #pragma unroll
    for(int offset = 16; offset > 0; offset /= 2){
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void rmsnorm_native_kernel(
    const float* X,
    const float* gamma,
    float* Y,
    int M,
    int N,
    float eps
) {
    // rmsnorm 算子的朴素实现
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(row < M){
        float sum = 0.0f;
        // 这个地方 我们做类似于网格跨步循环的操作来读取数据 使得一个 warp 可以负责一整行的数据
        int lane_id = threadIdx.x % 32;
        for(int i = lane_id; i < N; i += 32){
            sum += X[row * N + i] * X[row * N + i];
        }
        // 这里做碟形规约，让 warp 中所有的线程都拿到完整的平方和
        sum = warp_sum(sum);
        // 这里根据平方和、权重、偏置，计算最终的结果
        float scale = 1.0f / sqrtf(sum / N + eps);
        for(int i = lane_id; i < N; i += 32){
            Y[row * N + i] = X[row * N + i] * scale * gamma[i];
        }
    }
}


void solve(
    torch::Tensor X,
    torch::Tensor gamma,
    torch::Tensor Y,
    int M,
    int N,
    float eps
) {
    // 这个版本是 warp 负责一行 + two pass 读取
    dim3 threadsPerBlock(BLOCK_X, BLOCK_Y);
    dim3 blocksPerGrid(1, (M + BLOCK_Y - 1) / BLOCK_Y);

    rmsnorm_native_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        X.data_ptr<float>(),
        gamma.data_ptr<float>(),
        Y.data_ptr<float>(),
        M,
        N,
        eps
    );
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
