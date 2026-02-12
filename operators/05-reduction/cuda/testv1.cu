#include <cuda_runtime.h>

// 辅助函数
__device__ __forceinline__ float sum(float val){
    for(int offset = 16; offset > 0; offset /= 2){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template<int threadsPreBlock>
__global__ void reduction(const float* input, float* output, int N){
    // 静态共享内存
    __shared__ float data[(threadsPreBlock + 31) / 32];

    int tid = threadIdx.x; // 0 --- threadsPreBlock-1
    int bid = blockIdx.x; 
    int id = bid * blockDim.x + tid;

    // 先搬运数据
    float val = id < N ? input[id] : 0.0f;

    // 下面进行规约过程
    // 先计算单个warp的和
    val = sum(val); 

    // 每个warp内的第一个元素把局部warp和写入到共享内存
    if(tid % 32 == 0) data[tid / 32] = val;
    __syncthreads();

    // 对共享内存内部的数据进行第二次规约 完成整个block的规约
     if(tid < ((blockDim.x + 31) / 32)){
        // 从共享内存读取数据 注意非整除的情况
        float warp_sum = data[tid];
        warp_sum = sum(warp_sum);
        
        if(tid == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

// input, output are device pointers
// extern "C" void solve(const float* input, float* output, int N) 
// {
//     const int threadsPreBlock = 1024; 
//     const int blocksPerGrid = (N + threadsPreBlock - 1) / threadsPreBlock;
//     reduction<threadsPreBlock><<<blocksPerGrid, threadsPreBlock>>>(input, output, N);
// }

template<int threadsPreBlock>
__global__ void reduction_v2(const float* input, float* output, int N){
    // 采用固定规模大小的线程来完成任务 即grid_stride累加
    // 静态共享内存
    __shared__ float data[(threadsPreBlock + 31) / 32];

    int tid = threadIdx.x; // 0 --- threadsPreBlock-1
    int bid = blockIdx.x; 
    int id = bid * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;

    // 数据的循环搬运
    float val = 0.0f;
    // 注意索引起点为id
    for(int i = id; i < N; i+=stride){
        val += input[i];
    }

    // 下面进行规约过程
    // 先计算单个warp的和
    val = sum(val); 

    // 每个warp内的第一个元素把局部warp和写入到共享内存
    if(tid % 32 == 0) data[tid / 32] = val;
    __syncthreads();

    // 对共享内存内部的数据进行第二次规约 完成整个block的规约
     if(tid < ((blockDim.x + 31) / 32)){
        // 从共享内存读取数据 注意非整除的情况
        float warp_sum = data[tid];
        warp_sum = sum(warp_sum);
        
        if(tid == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) 
{
    cudaMemset(output, 0, sizeof(float));
    const int threadsPreBlock = 512; 
    // grid数量 控制一个上限 防止无限增加
    int sm = 0;
    cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0);
    const int blocksPerGrid = min((N + threadsPreBlock - 1) / threadsPreBlock, sm * 8); //可设置为4-12倍
    reduction_v2<threadsPreBlock><<<blocksPerGrid, threadsPreBlock>>>(input, output, N);
}

