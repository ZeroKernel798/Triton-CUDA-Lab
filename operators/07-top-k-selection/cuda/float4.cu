#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

// 宏注入，由 Runner 编译时提供
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

// 辅助函数：插入排序 (保持原逻辑)
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

// 辅助函数：合并两个 top-k 数组
__device__ __forceinline__ void merge_topk(float* dest, float* src, int k) {
    for (int i = 0; i < k; ++i) insert_into_topk(dest, src[i], k);
}

// 1. 局部海选：向量化优化版
__global__ void local_topk_kernel_opt(const float* __restrict__ input, float* __restrict__ temp_storage, int N, int k) {
    int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x; // 使用宏
    int stride = gridDim.x * BLOCK_SIZE;

    float local_topk[100]; 
    for (int i = 0; i < k; ++i) local_topk[i] = -FLT_MAX;

    // 安全检查：只有地址对齐且剩余元素足够时才用 float4
    uintptr_t addr = reinterpret_cast<uintptr_t>(input);
    bool is_aligned = (addr % 16 == 0);
    
    if (is_aligned && N >= 4) {
        const float4* input4 = reinterpret_cast<const float4*>(input);
        int N4 = N / 4;
        for (int i = tid; i < N4; i += stride) {
            float4 val4 = input4[i];
            insert_into_topk(local_topk, val4.x, k);
            insert_into_topk(local_topk, val4.y, k);
            insert_into_topk(local_topk, val4.z, k);
            insert_into_topk(local_topk, val4.w, k);
        }
        // 处理余数 (单线程处理末尾)
        if (tid == 0) {
            for (int i = N4 * 4; i < N; ++i) insert_into_topk(local_topk, input[i], k);
        }
    } else {
        for (int i = tid; i < N; i += stride) insert_into_topk(local_topk, input[i], k);
    }

    for (int i = 0; i < k; ++i) temp_storage[tid * k + i] = local_topk[i];
}

// 2. 决赛：并行共享内存规约版
__global__ void final_topk_kernel_parallel(const float* __restrict__ temp_storage, float* __restrict__ output, int num_candidates_per_thread, int k) {
    // 这里的共享内存由 solve 动态传入大小
    extern __shared__ float shared_topk[]; 
    
    int tid = threadIdx.x;
    float* my_topk = &shared_topk[tid * k];
    for (int i = 0; i < k; ++i) my_topk[i] = -FLT_MAX;

    // 每个线程处理一部分候选者
    for (int i = 0; i < num_candidates_per_thread; ++i) {
        int idx = tid * num_candidates_per_thread + i;
        // 注意：此处需要判断 idx 是否超出 temp_storage 的总长度
        insert_into_topk(my_topk, temp_storage[idx], k);
    }
    __syncthreads();

    // 最后的汇总 (在 block 内进行 merge)
    if (tid == 0) {
        for (int i = 0; i < k; ++i) output[i] = -FLT_MAX;
        for (int p = 0; p < blockDim.x; ++p) {
            merge_topk(output, &shared_topk[p * k], k);
        }
    }
}

void solve(torch::Tensor input, torch::Tensor output, int N, int k) {
    auto device = input.device();
    cudaSetDevice(device.index());
    
    // 向量化读取通常需要内存连续
    auto input_contig = input.contiguous();

    // 局部海选配置
    int num_blocks = 512; 
    int total_threads = num_blocks * BLOCK_SIZE;

    // 创建临时缓冲区
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    torch::Tensor temp_storage = torch::full({total_threads * k}, -FLT_MAX, options);

    const float* d_input = input_contig.data_ptr<float>();
    float* d_temp = temp_storage.data_ptr<float>();
    float* d_output = output.data_ptr<float>();

    // Step 1: 局部海选
    local_topk_kernel_opt<<<num_blocks, BLOCK_SIZE>>>(d_input, d_temp, N, k);

    // Step 2: 最终决赛 (固定使用 256 线程进行并行汇总)
    int final_threads = 256;
    int total_candidates = total_threads * k;
    int candidates_per_thread = (total_candidates + final_threads - 1) / final_threads;
    size_t shared_mem_size = final_threads * k * sizeof(float);
    
    // 动态申请共享内存
    cudaFuncSetAttribute(final_topk_kernel_parallel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared_mem_size);
    
    final_topk_kernel_parallel<<<1, final_threads, shared_mem_size>>>(d_temp, d_output, candidates_per_thread, k);
}

// Pybind11 接口
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("solve", &solve, "Optimal Top K (Macro Version)",
          py::arg("input"), 
          py::arg("output"), 
          py::arg("N"), 
          py::arg("k"));
}