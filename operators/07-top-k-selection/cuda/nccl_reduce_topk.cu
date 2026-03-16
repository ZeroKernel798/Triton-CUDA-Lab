#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>
#include <ATen/cuda/CUDAContext.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

// 插入排序：保持数组有序
__device__ __forceinline__ void insert_into_topk(float* topk_array, float val, int k) {
    if (val <= topk_array[k - 1]) return;
    topk_array[k - 1] = val;
    for (int j = k - 1; j > 0; --j) {
        if (topk_array[j] > topk_array[j - 1]) {
            float tmp = topk_array[j];
            topk_array[j] = topk_array[j - 1];
            topk_array[j - 1] = tmp;
        } else break;
    }
}

// Kernel 1: 局部海选 (利用 Shared Memory 规约)
__global__ void local_topk_kernel_smem(const float* __restrict__ input, float* __restrict__ temp_storage, int N, int k) {
    extern __shared__ float s_topk[]; 
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    float* my_s_ptr = &s_topk[tid * k];
    for (int i = 0; i < k; ++i) my_s_ptr[i] = -FLT_MAX;

    for (int i = idx; i < N; i += stride) {
        insert_into_topk(my_s_ptr, input[i], k);
    }
    __syncthreads();

    // Block 内由线程 0 汇总 (当 k 较小时，这样比树状规约更稳定)
    if (tid == 0) {
        float block_final[128]; 
        for (int i = 0; i < k; ++i) block_final[i] = -FLT_MAX;
        for (int t = 0; t < blockDim.x; ++t) {
            float* other_s_ptr = &s_topk[t * k];
            for (int i = 0; i < k; ++i) insert_into_topk(block_final, other_s_ptr[i], k);
        }
        for (int i = 0; i < k; ++i) temp_storage[blockIdx.x * k + i] = block_final[i];
    }
}

// Kernel 2: 并行汇总 (代替原来的单线程版本)
__global__ void final_local_aggregation_parallel(const float* __restrict__ temp_storage, float* __restrict__ local_final, int num_candidates_groups, int k) {
    // num_candidates_groups 是 block 的数量
    extern __shared__ float s_candidates[]; 
    int tid = threadIdx.x;

    // 每个线程处理一部分 block 产生的结果
    float local_best[128];
    for(int i=0; i<k; ++i) local_best[i] = -FLT_MAX;

    for (int i = tid; i < num_candidates_groups; i += blockDim.x) {
        for(int j=0; j<k; ++j) {
            insert_into_topk(local_best, temp_storage[i * k + j], k);
        }
    }

    // 写回 Shared Memory 进行最后的 Block 内规约
    for(int i=0; i<k; ++i) s_candidates[tid * k + i] = local_best[i];
    __syncthreads();

    if (tid == 0) {
        float final_best[128];
        for(int i=0; i<k; ++i) final_best[i] = -FLT_MAX;
        for (int t = 0; t < blockDim.x; ++t) {
            float* other_s_ptr = &s_candidates[t * k];
            for(int i=0; i<k; ++i) insert_into_topk(final_best, other_s_ptr[i], k);
        }
        for(int i=0; i<k; ++i) local_final[i] = final_best[i];
    }
}

void solve(torch::Tensor input, torch::Tensor output, int N, int k, int rank, int world_size) {
    auto device = input.device();
    cudaSetDevice(device.index());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int num_blocks = std::min((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 1024);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    torch::Tensor temp_storage = torch::full({num_blocks * k}, -FLT_MAX, options);

    size_t smem1 = BLOCK_SIZE * k * sizeof(float);
    local_topk_kernel_smem<<<num_blocks, BLOCK_SIZE, smem1, stream>>>(
        input.data_ptr<float>(), temp_storage.data_ptr<float>(), N, k);

    // 核心改进：用 256 个线程并行汇总，而不是 1 个
    size_t smem2 = BLOCK_SIZE * k * sizeof(float);
    final_local_aggregation_parallel<<<1, BLOCK_SIZE, smem2, stream>>>(
        temp_storage.data_ptr<float>(), output.data_ptr<float>(), num_blocks, k);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve", &solve, "Parallel Local Top-K",
          py::arg("input"), py::arg("output"), py::arg("N"), py::arg("k"), py::arg("rank"), py::arg("world_size"));
}