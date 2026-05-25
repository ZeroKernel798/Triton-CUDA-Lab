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

#define SWIZZLE_A(row, col) ((col) ^ ((row >> 2) << 3))

__global__ void double_buffer_sgemm_kernel(const float* A, const float* B, float* C, int N, int M, int K) 
{
    // 申请双缓冲共享内存 [2] 表示 ping-pong buffer
    __shared__ float sA[2][BK][BM];
    __shared__ float sB[2][BK][BN];

    // 256 个线程，搬运数据的时候，针对 A 和 B 矩阵有不同的分块方案
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int load_a_row = tid / 4;
    int load_a_col = (tid % 4) * 4;  
    int load_b_row = tid / 32;
    int load_b_col = (tid % 32) * 4;

    // 256 线程，计算的时候，按照一个线程负责 8x8 的块来进行划分
    int tile_row = tid / 16;
    int tile_col_0 = tid % 16 * 4;

    // 全局分块的 row 和 col 起始值
    int row_base = blockIdx.y * BM;
    int col_base = blockIdx.x * BN;

    // 单个线程累加器
    float sum[8][8] = {0.0f};

    // 先预先填充一块内存
    int k = 0;
    if((row_base + load_a_row) < M && (k + load_a_col) < K){
        float4 temp = CFLOAT4(A[(row_base + load_a_row) * K + k + load_a_col]); 
        sA[0][load_a_col + 0][SWIZZLE_A(load_a_col + 0, load_a_row)] = temp.x;
        sA[0][load_a_col + 1][SWIZZLE_A(load_a_col + 1, load_a_row)] = temp.y;
        sA[0][load_a_col + 2][SWIZZLE_A(load_a_col + 2, load_a_row)] = temp.z;
        sA[0][load_a_col + 3][SWIZZLE_A(load_a_col + 3, load_a_row)] = temp.w;
    }
    else{
        sA[0][load_a_col + 0][SWIZZLE_A(load_a_col + 0, load_a_row)] = 0.0f;
        sA[0][load_a_col + 1][SWIZZLE_A(load_a_col + 1, load_a_row)] = 0.0f;
        sA[0][load_a_col + 2][SWIZZLE_A(load_a_col + 2, load_a_row)] = 0.0f;
        sA[0][load_a_col + 3][SWIZZLE_A(load_a_col + 3, load_a_row)] = 0.0f;
    }

    if((row_base + load_a_row + 64) < M && (k + load_a_col) < K){
        float4 temp = CFLOAT4(A[(row_base + load_a_row + 64) * K + k + load_a_col]); 
        sA[0][load_a_col + 0][SWIZZLE_A(load_a_col + 0, load_a_row + 64)] = temp.x;
        sA[0][load_a_col + 1][SWIZZLE_A(load_a_col + 1, load_a_row + 64)] = temp.y;
        sA[0][load_a_col + 2][SWIZZLE_A(load_a_col + 2, load_a_row + 64)] = temp.z;
        sA[0][load_a_col + 3][SWIZZLE_A(load_a_col + 3, load_a_row + 64)] = temp.w;
    }
    else{
        sA[0][load_a_col + 0][SWIZZLE_A(load_a_col + 0, load_a_row + 64)] = 0.0f;
        sA[0][load_a_col + 1][SWIZZLE_A(load_a_col + 1, load_a_row + 64)] = 0.0f;
        sA[0][load_a_col + 2][SWIZZLE_A(load_a_col + 2, load_a_row + 64)] = 0.0f;
        sA[0][load_a_col + 3][SWIZZLE_A(load_a_col + 3, load_a_row + 64)] = 0.0f;
    }

    if((k + load_b_row) < K && (col_base + load_b_col) < N)
        FLOAT4(sB[0][load_b_row][load_b_col]) = CFLOAT4(B[(k + load_b_row) * N + col_base + load_b_col]);
    else
        FLOAT4(sB[0][load_b_row][load_b_col]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    if((k + load_b_row + 8) < K && (col_base + load_b_col) < N)
        FLOAT4(sB[0][load_b_row + 8][load_b_col]) = CFLOAT4(B[(k + load_b_row + 8) * N + col_base + load_b_col]);
    else
        FLOAT4(sB[0][load_b_row + 8][load_b_col]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    __syncthreads();

    // 开始循环
    int read_stage = 0;
    int write_stage = 1;

    for(int k = BK; k < K; k += BK){
        // 先预先取下一块的数据到寄存器
        float4 next_temp_a0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 next_temp_a1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 next_temp_b0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 next_temp_b1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        if((row_base + load_a_row) < M && (k + load_a_col) < K)
            next_temp_a0 = CFLOAT4(A[(row_base + load_a_row) * K + k + load_a_col]);
        if((row_base + load_a_row + 64) < M && (k + load_a_col) < K)
            next_temp_a1 = CFLOAT4(A[(row_base + load_a_row + 64) * K + k + load_a_col]);
        if((k + load_b_row) < K && (col_base + load_b_col) < N)
            next_temp_b0 = CFLOAT4(B[(k + load_b_row) * N + col_base + load_b_col]);
        if((k + load_b_row + 8) < K && (col_base + load_b_col) < N)
            next_temp_b1 = CFLOAT4(B[(k + load_b_row + 8) * N + col_base + load_b_col]);
        
        // 数据回来的途中，我们直接做本轮计算 
        #pragma unroll
        for(int kk = 0; kk < BK; kk++){
            alignas(16) float reg_a[8], reg_b[8];

            FLOAT4(reg_a[0]) = FLOAT4(sA[read_stage][kk][SWIZZLE_A(kk, tile_row * 8)]);
            FLOAT4(reg_a[4]) = FLOAT4(sA[read_stage][kk][SWIZZLE_A(kk, tile_row * 8 + 4)]);
            
            FLOAT4(reg_b[0]) = FLOAT4(sB[read_stage][kk][tile_col_0]);
            FLOAT4(reg_b[4]) = FLOAT4(sB[read_stage][kk][tile_col_0 + 64]);
            
            #pragma unroll
            for(int i = 0; i < 8; ++i){
                #pragma unroll
                for(int j = 0; j < 8; ++j){
                    sum[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        // 将之前在寄存器的数据写入共享内存供下一次循环使用 
        sA[write_stage][load_a_col + 0][SWIZZLE_A(load_a_col + 0, load_a_row)] = next_temp_a0.x;
        sA[write_stage][load_a_col + 1][SWIZZLE_A(load_a_col + 1, load_a_row)] = next_temp_a0.y;
        sA[write_stage][load_a_col + 2][SWIZZLE_A(load_a_col + 2, load_a_row)] = next_temp_a0.z;
        sA[write_stage][load_a_col + 3][SWIZZLE_A(load_a_col + 3, load_a_row)] = next_temp_a0.w;

        sA[write_stage][load_a_col + 0][SWIZZLE_A(load_a_col + 0, load_a_row + 64)] = next_temp_a1.x;
        sA[write_stage][load_a_col + 1][SWIZZLE_A(load_a_col + 1, load_a_row + 64)] = next_temp_a1.y;
        sA[write_stage][load_a_col + 2][SWIZZLE_A(load_a_col + 2, load_a_row + 64)] = next_temp_a1.z;
        sA[write_stage][load_a_col + 3][SWIZZLE_A(load_a_col + 3, load_a_row + 64)] = next_temp_a1.w;

        FLOAT4(sB[write_stage][load_b_row][load_b_col]) = next_temp_b0;
        FLOAT4(sB[write_stage][load_b_row + 8][load_b_col]) = next_temp_b1;

        // 很重要，既保护 write_stage 的 buffer 写入，又保护计算的时候， read_stage 的 buffer 不被覆盖
        __syncthreads(); 
        
        // 交换 buffer 角标，刚才写入的区域，变成了 read_stage，下一轮计算使用
        read_stage ^= 1;
        write_stage ^= 1;
    }

    // 处理最后一块数据
    #pragma unroll
    for(int kk = 0; kk < BK; kk++){
        alignas(16) float reg_a[8], reg_b[8];

        FLOAT4(reg_a[0]) = FLOAT4(sA[read_stage][kk][SWIZZLE_A(kk, tile_row * 8)]);
        FLOAT4(reg_a[4]) = FLOAT4(sA[read_stage][kk][SWIZZLE_A(kk, tile_row * 8 + 4)]);
        
        FLOAT4(reg_b[0]) = FLOAT4(sB[read_stage][kk][tile_col_0]);
        FLOAT4(reg_b[4]) = FLOAT4(sB[read_stage][kk][tile_col_0 + 64]);
        
        #pragma unroll
        for(int i = 0; i < 8; ++i){
            #pragma unroll
            for(int j = 0; j < 8; ++j){
                sum[i][j] += reg_a[i] * reg_b[j];
            }
        }
    }

    // 数据输出到 c 矩阵
    #pragma unroll
    for(int i = 0; i < 8; ++i){
        int global_c_row = row_base + tile_row * 8 + i;
        int global_c_col_0 = col_base + tile_col_0;
        int global_c_col_1 = col_base + tile_col_0 + 64;

        if (global_c_row < M) {
            if (global_c_col_0 < N) {
                FLOAT4(C[global_c_row * N + global_c_col_0]) = FLOAT4(sum[i][0]);
            }
            if (global_c_col_1 < N) {
                FLOAT4(C[global_c_row * N + global_c_col_1]) = FLOAT4(sum[i][4]);
            }
        }
    }
}


void solve(torch::Tensor A, torch::Tensor B, torch::Tensor C, int N, int M, int K) {
    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    dim3 threadsPerBlock(BLOCK_X, BLOCK_Y);
    dim3 blocksPerGrid((N + BN - 1) / BN, (M + BM - 1) / BM); 

    double_buffer_sgemm_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N, M, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("solve", &solve, "Double Buffer MatMul Kernel",
          py::arg("A"), py::arg("B"), py::arg("C"), 
          py::arg("N"), py::arg("M"), py::arg("K")); 
}

