#include <torch/extension.h>


#ifndef BLOCK_X
#define BLOCK_X 256
#endif

#ifndef TILE_K
#define TILE_K 512
#endif

__global__ void smem_gemv_kernel(
    const float* __restrict__ x,
    const float* __restrict__ A,
    float* __restrict__ y,
    int K,
    int N
) {
    // 在 vector.cu (一线程 4 列 + float4) 基础上，把 x 分块缓存进 shared memory。
    // 同一 block 的 BLOCK_X 个线程负责 BLOCK_X*4 个相邻列，但共享同一条 x；
    // x 每个 tile 只从 global 读一次进 smem，block 内所有线程复用，省掉 x 的重复读。
    __shared__ __align__(16) float sx[TILE_K];

    int col = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for(int tile = 0; tile < K; tile += TILE_K){
        // 以 float4 为单位把 x 的一个 tile 搬进 smem，一次搬 4 个，减少加载指令。
        // tile 是 4 的倍数 => x+tile 16B 对齐；K 是 4 的倍数 => 每个 float4 都完整不越界。
        float4* sx4 = reinterpret_cast<float4*>(sx);
        const float4* x4 = reinterpret_cast<const float4*>(x + tile);
        for(int t = threadIdx.x; t * 4 < TILE_K && (tile + t * 4) < K; t += blockDim.x){
            sx4[t] = x4[t];
        }
        __syncthreads();

        // 用 smem 里的 x 累加这一段对 4 个列的贡献。
        if(col < N){
            int kk = min(TILE_K, K - tile);
            for(int t = 0; t < kk; ++t){
                float xi = sx[t];
                float4 a = *reinterpret_cast<const float4*>(A + (tile + t) * N + col);
                acc.x += xi * a.x;
                acc.y += xi * a.y;
                acc.z += xi * a.z;
                acc.w += xi * a.w;
            }
        }
        // 下一轮覆盖 smem 前要同步，避免还有线程在读旧 tile。
        __syncthreads();
    }

    if(col < N){
        *reinterpret_cast<float4*>(y + col) = acc;
    }
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

    smem_gemv_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        x.data_ptr<float>(),
        A.data_ptr<float>(),
        y.data_ptr<float>(),
        K,
        N
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("solve", &solve, "GEMV Shared-Memory Kernel",
          py::arg("x"),
          py::arg("A"),
          py::arg("y"),
          py::arg("K"),
          py::arg("N"));
}
