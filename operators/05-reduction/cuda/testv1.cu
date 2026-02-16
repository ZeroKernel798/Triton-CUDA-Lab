#include <cuda_runtime.h>
#include <torch/extension.h>

// 辅助函数：Warp 内规约
__device__ __forceinline__ float warp_sum(float val){
    #pragma unroll
    for(int offset = 16; offset > 0; offset /= 2){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template<int threadsPerBlock>
__global__ void reduction_kernel_v2(const float* input, float* output, int N){
    // 静态共享内存：存放每个 Warp 的局部和
    __shared__ float s_data[(threadsPerBlock + 31) / 32];

    int tid = threadIdx.x;
    int bid = blockIdx.x; 
    int id = bid * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;

    // 1. Grid-stride Loop：每个线程累加多个位置的数据
    float val = 0.0f;
    for(int i = id; i < N; i += stride){
        val += input[i];
    }

    // 2. Warp-level Reduction：使用 Shuffle 指令
    val = warp_sum(val); 

    // 每个 Warp 的第 0 号线程写入共享内存
    if(tid % 32 == 0) s_data[tid / 32] = val;
    __syncthreads();

    // 3. Block-level Reduction：第一个 Warp 对共享内存中的数据进行二次规约
    if(tid < ((blockDim.x + 31) / 32)){
        float warp_res = s_data[tid];
        warp_res = warp_sum(warp_res);
        
        // 4. Global-level Reduction：通过 AtomicAdd 写入最终结果
        if(tid == 0) {
            atomicAdd(output, warp_res);
        }
    }
}

// Pybind11 接口函数
void solve(torch::Tensor input, torch::Tensor output, int N) {
    // 预热：在运行前将 output 清零（框架如果没处理，这里必须处理）
    cudaMemset(output.data_ptr<float>(), 0, sizeof(float));

    const int threadsPerBlock = 512; 
    
    // 获取 SM 数量进行动态负载平衡
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    
    // 经典的 Grid Size 调优逻辑
    const int blocksPerGrid = std::min((N + threadsPerBlock - 1) / threadsPerBlock, sm_count * 8);

    reduction_kernel_v2<threadsPerBlock><<<blocksPerGrid, threadsPerBlock>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        N
    );
}

// 模块定义
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve", &solve, "Grid-stride Reduction with Warp Shuffle (CUDA)");
}