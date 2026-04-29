#include <torch/extension.h>


#ifndef BLOCK_X
#define BLOCK_X 32
#endif

__device__ __forceinline__ float warp_sum(float val){
    #pragma unroll
    for(int offset = 16; offset > 0; offset /= 2){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void rmsnorm_rowblock_kernel(
    const float* X,
    const float* gamma,
    float* Y,
    int M,
    int N,
    float eps
) {
    // rmsnorm 算子的 row block 实现
    int row = blockIdx.x;
    if(row < M){
        __shared__ float warp_sums[(BLOCK_X + 31) / 32];
        __shared__ float scale_smem;

        int tid = threadIdx.x;
        int lane_id = tid % 32;
        int warp_id = tid / 32;
        int num_warps = (blockDim.x + 31) / 32;

        float sum = 0.0f;
        for(int i = tid; i < N; i += blockDim.x){
            sum += X[row * N + i] * X[row * N + i];
        }

        // 第一阶段：每个 warp 内部规约，lane 0 得到该 warp 的 partial sum。
        sum = warp_sum(sum);

        if(lane_id == 0){
            warp_sums[warp_id] = sum;
        }
        __syncthreads();

        // 第二阶段：第一个 warp 规约所有 warp 的 partial sum。
        float block_sum = 0.0f;
        if(warp_id == 0){
            block_sum = lane_id < num_warps ? warp_sums[lane_id] : 0.0f;
            block_sum = warp_sum(block_sum);
        }

        if(tid == 0){
            scale_smem = 1.0f / sqrtf(block_sum / N + eps);
        }
        __syncthreads();

        float scale = scale_smem;
        for(int i = tid; i < N; i += blockDim.x){
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
    // 这个版本是 一个 block 负责一行 + two pass 读取
    dim3 threadsPerBlock(BLOCK_X);
    dim3 blocksPerGrid(M);

    rmsnorm_rowblock_kernel<<<blocksPerGrid, threadsPerBlock>>>(
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
    m.def("solve", &solve, "RMSNorm Row Block Kernel",
          py::arg("X"),
          py::arg("gamma"),
          py::arg("Y"),
          py::arg("M"),
          py::arg("N"),
          py::arg("eps"));
}
