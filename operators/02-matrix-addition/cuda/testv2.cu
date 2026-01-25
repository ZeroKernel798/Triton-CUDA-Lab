#include <cuda_runtime.h>

__global__ void matrix_add(const float* A, const float* B, float* C, int N) 
{
    // 矩阵加法 采用float4进行加速 x方向一个线程负责一个float4 y负责一个float
    int x_base = blockDim.x * blockIdx.x + threadIdx.x;
    int x = x_base * 4;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(x >= N || y >= N) return ;
    if((x + 3) < N && (y * N) % 4 == 0){
        // float4处理
        const float4* A_data = reinterpret_cast<const float4*>(&(A[y * N + x]));
        const float4* B_data = reinterpret_cast<const float4*>(&(B[y * N + x]));
        float4* C_data = reinterpret_cast<float4*>(&(C[y * N + x]));

        float4 A_val = *A_data;
        float4 B_val = *B_data;
        float4 C_val;

        // 开始加法计算
        C_val.x = A_val.x + B_val.x;
        C_val.y = A_val.y + B_val.y;
        C_val.z = A_val.z + B_val.z;
        C_val.w = A_val.w + B_val.w;

        *C_data = C_val; 
    }
    else{
        // float处理
        for(int i = x; i < N; i++){
            C[y * N + i] = A[y * N + i] + B[y * N + i];
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int n_vec = (N + 3) / 4; // N个数据对应的float4向量个数
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((n_vec+15)/16, (N+15)/16);

    matrix_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    // cudaDeviceSynchronize();
}
