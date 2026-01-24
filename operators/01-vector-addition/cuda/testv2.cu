#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    // 用float4处理向量加法 提高读取效率
    // 每一个线程处理一个float4向量 找地址注意4倍关系
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = tid * 4;
    if(offset >= N) return ;

    if(offset + 3 < N){
        const float4* A_data = reinterpret_cast<const float4*>(&(A[offset]));
        const float4* B_data = reinterpret_cast<const float4*>(&(B[offset]));
        float4* C_data = reinterpret_cast<float4*>(&(C[offset]));

        float4 A_val = *A_data;
        float4 B_val = *B_data;
        float4 C_val;

        C_val.x = A_val.x + B_val.x;
        C_val.y = A_val.y + B_val.y;
        C_val.z = A_val.z + B_val.z;
        C_val.w = A_val.w + B_val.w;

        *C_data = C_val;
    }
    else if(offset < N){
        for(int i = offset; i < N; i++){
            C[i] = A[i] + B[i];
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 128;
    int n_vector = (N + 3) / 4; //表示float4向量的个数
    int blocksPerGrid = (n_vector + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    // cudaDeviceSynchronize();
}
