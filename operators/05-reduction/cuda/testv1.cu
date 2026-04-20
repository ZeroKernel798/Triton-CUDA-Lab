#include <cuda_runtime.h>
#include <torch/extension.h>

// 宏定义保护，由 Python 端的 get_macros 注入
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

// 辅助函数：Warp 内规约（Shuffle 指令极快，且不需要共享内存）
__device__ __forceinline__ float warp_sum(float val){
    #pragma unroll
    for(int offset = 16; offset > 0; offset /= 2){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template<int threadsPerBlock>
__global__ void reduction_kernel_v2(const float* __restrict__ input, float* __restrict__ output, int N){
    // 编译器现在知道 threadsPerBlock 是常量，静态分配共享内存
    __shared__ float s_data[(threadsPerBlock + 31) / 32];

    int tid = threadIdx.x;
    int bid = blockIdx.x; 
    int id = bid * threadsPerBlock + tid;
    int stride = gridDim.x * threadsPerBlock;

    // 1. Grid-stride Loop：每个线程累加多个位置的数据（处理非 2 幂次或超大规模数据的关键）
    float val = 0.0f;
    for(int i = id; i < N; i += stride){
        val += input[i];
    }

    // 2. Warp-level Reduction
    val = warp_sum(val); 

    // 每个 Warp 的“领头羊”（0号线程）写入共享内存
    if(tid % 32 == 0) s_data[tid / 32] = val;
    __syncthreads();

    // 3. Block-level Reduction：由第一个 Warp 对刚才存入 s_data 的各 Warp 和进行最后规约
    if(tid < ((threadsPerBlock + 31) / 32)){
        float warp_res = s_data[tid];
        warp_res = warp_sum(warp_res);
        
        // 4. Global-level Reduction：原子操作写入全局
        if(tid == 0) {
            atomicAdd(output, warp_res);
        }
    }
}

// 移除 block_size 参数，直接用宏启动
void solve(torch::Tensor input, torch::Tensor output, int N) {
    // 将 output 清零（规约计算是累加过程）
    cudaMemset(output.data_ptr<float>(), 0, sizeof(float));

    // 动态计算 Grid Size 以跑满 GPU SM
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    
    // 经典的负载平衡策略：每个 SM 跑 8 个 Block 左右通常能较好地掩盖访存延迟
    const int blocksPerGrid = std::min((N + BLOCK_SIZE - 1) / BLOCK_SIZE, sm_count * 8);

    // 直接实例化并启动，不再需要冗长的 switch-case
    reduction_kernel_v2<BLOCK_SIZE><<<blocksPerGrid, BLOCK_SIZE>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        N
    );
}

// 模块定义
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve", &solve, "Grid-stride Reduction with Warp Shuffle (Macro Configured)",
          py::arg("input"),
          py::arg("output"),
          py::arg("N")
          // 不再暴露 block_size 给 python，由框架在编译时注入宏
    );
}