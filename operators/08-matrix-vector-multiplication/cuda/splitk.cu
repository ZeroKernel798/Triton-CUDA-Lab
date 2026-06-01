#include <torch/extension.h>


#ifndef BLOCK_X
#define BLOCK_X 256
#endif

#ifndef SPLIT_K
#define SPLIT_K 4
#endif

#ifndef TILE_K
#define TILE_K 512
#endif

// 阶段 1: 每个 split 段算自己那段 K 的部分和，写到临时 buffer partial[SPLIT_K, N]。
// 各 split 写各自的行，没有竞争（不再需要 atomicAdd）。block 内用 smem 缓存本段 x 复用。
__global__ void splitk_partial_kernel(
    const float* __restrict__ x,
    const float* __restrict__ A,
    float* __restrict__ partial,
    int K,
    int N
) {
    __shared__ __align__(16) float sx[TILE_K];

    int col = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int split = blockIdx.y;

    // 每段长度向上对齐到 4，保证各段起点 k_start 是 4 的倍数（float4 对齐）。
    int k_per_split = ((K + SPLIT_K - 1) / SPLIT_K + 3) / 4 * 4;
    int k_start = split * k_per_split;
    int k_end = min(K, k_start + k_per_split);

    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for(int tile = k_start; tile < k_end; tile += TILE_K){
        int tile_len = min(TILE_K, k_end - tile);

        // 以 float4 为单位把本段 x 的一个 tile 搬进 smem。
        float4* sx4 = reinterpret_cast<float4*>(sx);
        const float4* x4 = reinterpret_cast<const float4*>(x + tile);
        for(int t = threadIdx.x; t * 4 < tile_len; t += blockDim.x){
            sx4[t] = x4[t];
        }
        __syncthreads();

        if(col < N){
            for(int t = 0; t < tile_len; ++t){
                float xi = sx[t];
                float4 a = *reinterpret_cast<const float4*>(A + (tile + t) * N + col);
                acc.x += xi * a.x;
                acc.y += xi * a.y;
                acc.z += xi * a.z;
                acc.w += xi * a.w;
            }
        }
        __syncthreads();
    }

    // 写到 partial 的第 split 行，各 split 互不重叠，无需 atomic。
    if(col < N){
        *reinterpret_cast<float4*>(partial + split * N + col) = acc;
    }
}

// 阶段 2: 沿 SPLIT_K 维把 partial[0..SPLIT_K-1][col] 规约相加，写回 y[col]。
__global__ void splitk_reduce_kernel(
    const float* __restrict__ partial,
    float* __restrict__ y,
    int N
) {
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if(col >= N) return;

    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    #pragma unroll
    for(int s = 0; s < SPLIT_K; ++s){
        float4 p = *reinterpret_cast<const float4*>(partial + s * N + col);
        acc.x += p.x;
        acc.y += p.y;
        acc.z += p.z;
        acc.w += p.w;
    }
    *reinterpret_cast<float4*>(y + col) = acc;
}


void solve(
    torch::Tensor x,
    torch::Tensor A,
    torch::Tensor y,
    int K,
    int N
) {
    // 临时 buffer 存各 split 的部分和，[SPLIT_K, N]，与 x/A 同 device、同 dtype。
    auto partial = torch::empty({SPLIT_K, N}, x.options());

    dim3 threadsPerBlock(BLOCK_X);
    dim3 colGrid((N / 4 + BLOCK_X - 1) / BLOCK_X);

    // 阶段 1: grid.x 覆盖列 (每线程 4 列)，grid.y = SPLIT_K 覆盖 K 维分段。
    dim3 partialGrid(colGrid.x, SPLIT_K);
    splitk_partial_kernel<<<partialGrid, threadsPerBlock>>>(
        x.data_ptr<float>(),
        A.data_ptr<float>(),
        partial.data_ptr<float>(),
        K,
        N
    );

    // 阶段 2: 沿 SPLIT_K 规约 partial -> y。
    splitk_reduce_kernel<<<colGrid, threadsPerBlock>>>(
        partial.data_ptr<float>(),
        y.data_ptr<float>(),
        N
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("solve", &solve, "GEMV Split-K (smem + two-stage) Kernel",
          py::arg("x"),
          py::arg("A"),
          py::arg("y"),
          py::arg("K"),
          py::arg("N"));
}
