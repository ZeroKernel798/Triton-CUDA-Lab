#include <torch/extension.h>

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


__global__ void smem_outer8x8_at_sgemm_kernel(const float* A, const float* B, float* C, int N, int M, int K) 
{
    // 基于共享内存 + 外积优化 + A 矩阵转置的矩阵乘法
    // 核心逻辑就是，在内层读取 sA 进行计算的时候，我们尽可能减少 LSU 的压力，减少气泡，所以我们在外层转置写入，内层向量化读取来优化
    // 申请分块共享内存
    __shared__ float sA[BK][BM];
    __shared__ float sB[BK][BN];

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
        // 先搬运数据到共享内存 sA，转置写入，为了后续减少 LSU 的压力
        // 读取矩阵 A 的位置不需要变化，但是写入的时候，需要做转置写入，无法用 float4 来加速
        if((row_base + load_a_row) < M && (k + load_a_col) < K){
            float4 temp = CFLOAT4(A[(row_base + load_a_row) * K + k + load_a_col]); 
            // 转置按列写入
            sA[load_a_col + 0][load_a_row] = temp.x;
            sA[load_a_col + 1][load_a_row] = temp.y;
            sA[load_a_col + 2][load_a_row] = temp.z;
            sA[load_a_col + 3][load_a_row] = temp.w;
        }
        else{
            sA[load_a_col + 0][load_a_row] = 0.0f;
            sA[load_a_col + 1][load_a_row] = 0.0f;
            sA[load_a_col + 2][load_a_row] = 0.0f;
            sA[load_a_col + 3][load_a_row] = 0.0f;
        }

        if((row_base + load_a_row + 64) < M && (k + load_a_col) < K){
            float4 temp = CFLOAT4(A[(row_base + load_a_row + 64) * K + k + load_a_col]); 
            sA[load_a_col + 0][load_a_row + 64] = temp.x;
            sA[load_a_col + 1][load_a_row + 64] = temp.y;
            sA[load_a_col + 2][load_a_row + 64] = temp.z;
            sA[load_a_col + 3][load_a_row + 64] = temp.w;
        }
        else{
            sA[load_a_col + 0][load_a_row + 64] = 0.0f;
            sA[load_a_col + 1][load_a_row + 64] = 0.0f;
            sA[load_a_col + 2][load_a_row + 64] = 0.0f;
            sA[load_a_col + 3][load_a_row + 64] = 0.0f;
        }

        // 搬运数据到共享内存 sB 记得需要搬运两次
        if((k + load_b_row) < K && (col_base + load_b_col) < N)
            FLOAT4(sB[load_b_row][load_b_col]) = CFLOAT4(B[(k + load_b_row) * N + col_base + load_b_col]);
        else
            FLOAT4(sB[load_b_row][load_b_col]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        
        if((k + load_b_row + 8) < K && (col_base + load_b_col) < N)
            FLOAT4(sB[load_b_row + 8][load_b_col]) = CFLOAT4(B[(k + load_b_row + 8) * N + col_base + load_b_col]);
        else
            FLOAT4(sB[load_b_row + 8][load_b_col]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        __syncthreads();

        // 做外积计算，这里有理论证明，方形的最合适，所以我们采用 8x8 的分块。
        for(int kk = 0; kk < BK; kk++){
            alignas(16) float reg_a[8], reg_b[8];

            // 加载数据
            // 这里对 sA 的读取，优化成了 float4 读取，减少计算循环中的空气泡
            FLOAT4(reg_a[0]) = FLOAT4(sA[kk][tile_row * 8]);
            FLOAT4(reg_a[4]) = FLOAT4(sA[kk][tile_row * 8 + 4]);

            FLOAT4(reg_b[0]) = FLOAT4(sB[kk][tile_col]);
            FLOAT4(reg_b[4]) = FLOAT4(sB[kk][tile_col + 4]);

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

        // 安全边界判定，然后每次到一个新的 col 我们需要写两次
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

    smem_outer8x8_at_sgemm_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N, M, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("solve", &solve, "SMem Tiled Matrix Multiplication Use Float4 And Outer Product",
          py::arg("A"), py::arg("B"), py::arg("C"), 
          py::arg("N"), py::arg("M"), py::arg("K")); 
}

