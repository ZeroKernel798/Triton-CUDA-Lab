#include <cuda_runtime.h>

__global__ void matrix_add(const float* A, const float* B, float* C, int Ne) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = tid * 4;

    if (offset + 3 < Ne) {
        // 直接用 tid 索引 float4 指针，编译器会自动处理 *16 的偏移
        float4 Av = reinterpret_cast<const float4*>(A)[tid];
        float4 Bv = reinterpret_cast<const float4*>(B)[tid];
        
        float4 Cv;
        Cv.x = Av.x + Bv.x;
        Cv.y = Av.y + Bv.y;
        Cv.z = Av.z + Bv.z;
        Cv.w = Av.w + Bv.w;

        reinterpret_cast<float4*>(C)[tid] = Cv;
    } 
    else if (offset < Ne) {
        // 处理末尾不足 4 个的部分
        for (int i = offset; i < Ne; i++) {
            C[i] = A[i] + B[i];
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int Ne = N * N;
    int threadsPerBlock = 256;
    int n_vec = (Ne + 3) / 4; // float4向量的个数
    int blocksPerGrid = (n_vec + threadsPerBlock - 1) / threadsPerBlock;
    matrix_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, Ne);
    // cudaDeviceSynchronize();
}

