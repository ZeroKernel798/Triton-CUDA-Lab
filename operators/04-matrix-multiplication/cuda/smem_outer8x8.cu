#include <cuda_runtime.h>
#include <torch/extension.h>
#include <string>

// 这里是对每次搬运到共享内存的数据尺寸的定义
#ifndef BM
#define BM 128
#endif
#ifndef BN
#define BN 128
#endif
#ifndef BK
#define BK 16
#endif

// 这里是设置线程块的参数
#ifndef BLOCK_X
#define BLOCK_X 16
#endif
#ifndef BLOCK_Y
#define BLOCK_Y 16
#endif


// 定义宏 方便操作
#define FLOAT4(var) (reinterpret_cast<float4*>(&(var))[0])
#define CFLOAT4(var) (reinterpret_cast<const float4*>(&(var))[0])


__global__ void smem_outer8x8_sgemm_kernel(const float* A, const float* B, float* C, int N, int M, int K) 
{
    // 基于共享内存 + 外积优化的矩阵乘法
    // 申请分块共享内存
    __shared__ float sA[BM * BK];
    __shared__ float sB[BK * BN];

    // 256 个线程，搬运数据的时候，针对 A 和 B 矩阵有不同的分块方案
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int load_a_row = tid / 4;
    int load_a_col = (tid % 4) * 4;  
    int load_b_row = tid / 32;
    int load_b_col = (tid % 32) * 4;

    // 256 线程，计算的时候，按照一个线程负责 8x8 的块来进行划分
    int tile_row = tid / 16;
    int tile_col = tid % 16 * 8;

    // 全局分块的 row 和 col 起始值
    int row_base = blockIdx.y * BM;
    int col_base = blockIdx.x * BN;

    // 单个线程累加器
    float sum[8][8] = {0.0f};

    // k 维度循环来搬运数据到共享内存
    for(int k = 0; k < K; k += BK){
        // 先搬运数据到共享内存 sA 记得需要搬运两次
        if((row_base + load_a_row) < M && (k + load_a_col) < K)
            FLOAT4(sA[load_a_row * BK + load_a_col]) = CFLOAT4(A[(row_base + load_a_row) * K + k + load_a_col]); 
        else{
            FLOAT4(sA[load_a_row * BK + load_a_col]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        if((row_base + load_a_row + 64) < M && (k + load_a_col) < K)
            FLOAT4(sA[(load_a_row + 64) * BK + load_a_col]) = CFLOAT4(A[(row_base + load_a_row + 64) * K + k + load_a_col]); 
        else{
            FLOAT4(sA[(load_a_row + 64) * BK + load_a_col]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        // 搬运数据到共享内存 sB 记得需要搬运两次
        if((k + load_b_row) < K && (col_base + load_b_col) < N)
        FLOAT4(sB[load_b_row * BN + load_b_col]) = CFLOAT4(B[(k + load_b_row) * N + col_base + load_b_col]);
        else
            FLOAT4(sB[load_b_row * BN + load_b_col]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        if((k + load_b_row + 8) < K && (col_base + load_b_col) < N)
            FLOAT4(sB[(load_b_row + 8) * BN + load_b_col]) = CFLOAT4(B[(k + load_b_row + 8) * N + col_base + load_b_col]);
        else
            FLOAT4(sB[(load_b_row + 8) * BN + load_b_col]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        __syncthreads();

        // 做外积计算，这里有理论证明，方形的最合适，所以我们采用 8x8 的分块。
        for(int kk = 0; kk < BK; kk++){
            float reg_a[8], reg_b[8];

            // 加载数据
            for(int i = 0; i < 8; ++i){
                reg_a[i] = sA[(tile_row * 8 + i) * BK + kk];
            }

            FLOAT4(reg_b[0]) = FLOAT4(sB[kk * BN + tile_col]);
            FLOAT4(reg_b[4]) = FLOAT4(sB[kk * BN + tile_col + 4]);

            // 计算结果
            for(int i = 0; i < 8; ++i){
                for(int j = 0; j < 8; ++j){
                    sum[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for(int i = 0; i < 8; ++i){
        // 每个线程真实的全局 C 矩阵物理坐标
        int global_c_row = row_base + tile_row * 8 + i;
        int global_c_col = col_base + tile_col;

        // 安全边界判定
        if (global_c_row < M && global_c_col < N) {
            FLOAT4(C[global_c_row * N + global_c_col]) = FLOAT4(sum[i][0]);
            FLOAT4(C[global_c_row * N + global_c_col + 4]) = FLOAT4(sum[i][4]);
        }
    }
}


void solve(torch::Tensor A, torch::Tensor B, torch::Tensor C, int N, int M, int K) {
    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    dim3 threadsPerBlock(BLOCK_X, BLOCK_Y);
    dim3 blocksPerGrid((N + BN - 1) / BN, (M + BM - 1) / BM); 

    smem_outer8x8_sgemm_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N, M, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("solve", &solve, "SMem Tiled Matrix Multiplication Use Float4 And Outer Product",
          py::arg("A"), py::arg("B"), py::arg("C"), 
          py::arg("N"), py::arg("M"), py::arg("K")); 
}