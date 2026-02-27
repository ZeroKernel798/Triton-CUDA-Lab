#include <cuda_runtime.h>
#include <torch/extension.h>
#include <string>

// 纯外积版本 暂无任何优化
template<int BM, int BN, int BK>
__global__ void matrix_mul_kernel_outer_product_pure(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, 
                                                     int N, int M, int K) 
{
    // 共享内存
    __shared__ float sA[BK * BM];   //注意布局 为BK BM 需要transpose后写入
    __shared__ float sB[BK * BN];

    // 获取二维线程索引
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 计算线程在线程块内的顺序索引以及线程块的总线程数量
    int idx = ty * blockDim.x + tx;
    int thread_nums = blockDim.x * blockDim.y;

    // 当前 Block 负责全局矩阵的起始位置 逻辑索引
    int row_start = blockIdx.y * BM;
    int col_start = blockIdx.x * BN;

    // 使用寄存器存储累加和 每个线程负责8x8的tile
    float accum[8][8] = {0.0f};

    // 分块处理过程 按照BK的步进来处理
    // 线程块先协力一起搬运数据 然后单个线程进行自己的计算
    for(int k_offset = 0; k_offset < K; k_offset += BK){
        // 仿照网格跨步循环的思路 实现线程块跨步循环搬运数据
        // 先搬运矩阵A到共享内存
        #pragma unroll
        for(int i = idx; i < BK * BM; i += thread_nums){
            // 计算当前拿的数据 在共享内存的行列逻辑索引
            int a_tile_r = i / BK;
            int a_tile_c = i % BK;
            float val = 0.0f;

            // 注意判断数据是否有效
            if((row_start + a_tile_r) < M && (k_offset + a_tile_c < K))
                val = A[(row_start + a_tile_r) * K + (k_offset + a_tile_c)];

            // 写入共享内存 注意写出变成了transpose
            sA[a_tile_c * BM + a_tile_r] = val;
        }

        // 搬运矩阵B到共享内存
        #pragma unroll
        for(int i = idx; i < BK * BN; i += thread_nums){
            // 计算当前拿的数据 在共享内存的行列逻辑索引
            int b_tile_r = i / BN;
            int b_tile_c = i % BN;
            float val = 0.0f;

            // 注意判断数据是否有效
            if((k_offset + b_tile_r) < K && (col_start + b_tile_c) < N)
                val = B[(k_offset + b_tile_r) * N + (col_start + b_tile_c)];

            // 写入共享内存 注意写出变成了transpose
            sB[b_tile_r * BN + b_tile_c] = val;
        }

        __syncthreads();

        // 单个线程的计算过程
        // 注意这个地方的索引顺序要遵循K->M->N
        #pragma unroll
        for(int kk = 0; kk < BK; kk++){
            // 寄存器
            float reg_a[8];
            float reg_b[8];

            // 先从sA读取八个数据 因为sA transpose 这个地方是取行
            // 如果不好理解sA的一列怎么获取 可以直接回到transpose前的一列逻辑 
            #pragma unroll
            for(int i = 0; i < 8; ++i) reg_a[i] = sA[kk * BM + ty * 8 + i]; 
            // 从sB读取八个数据
            #pragma unroll
            for(int i = 0; i < 8; ++i) reg_b[i] = sB[kk * BN + tx * 8 + i]; 

            // 计算 64 个输出 注意循环的对应关系
            #pragma unroll
            for(int i = 0; i < 8; ++i){
                for(int j = 0; j < 8; ++j){
                    accum[i][j] += reg_a[i] * reg_b[j];
                }
            }

        }
        __syncthreads();

    }

    // 写回阶段 将当前线程获取的值输出给矩阵C
    #pragma unroll
    for(int i = 0; i < 8; ++i){
        for(int j = 0; j < 8; ++j){
            int g_r = row_start + ty * 8 + i;
            int g_c = col_start + tx * 8 + j;
            if (g_r < M && g_c < N) {
                C[g_r * N + g_c] = accum[i][j];
            }
        }
    }
}

// 启动器
void solve(torch::Tensor A, torch::Tensor B, torch::Tensor C, 
           int N, int M, int K, int bx, int by, int bk) {
    auto d_A = A.data_ptr<float>(); 
    auto d_B = B.data_ptr<float>(); 
    auto d_C = C.data_ptr<float>();
    
    // 配置 Block 线程数：(BN/8, BM/8)
    dim3 threads(bx / 8, by / 8); 
    dim3 blocks((N + bx - 1) / bx, (M + by - 1) / by);
    
    // 模版分发
    if (bx == 128 && by == 128 && bk == 8) {
        matrix_mul_kernel_outer_product_pure<128, 128, 8><<<blocks, threads>>>(d_A, d_B, d_C, N, M, K);
    } else if (bx == 64 && by == 128 && bk == 8) {
        matrix_mul_kernel_outer_product_pure<128, 64, 8><<<blocks, threads>>>(d_A, d_B, d_C, N, M, K);
    } else if (bx == 128 && by == 64 && bk == 8) {
        matrix_mul_kernel_outer_product_pure<64, 128, 8><<<blocks, threads>>>(d_A, d_B, d_C, N, M, K);
    } else {
        matrix_mul_kernel_outer_product_pure<32, 32, 32><<<blocks, threads>>>(d_A, d_B, d_C, N, M, K);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve", &solve, "Pure Outer Product GEMM",
          py::arg("A"), py::arg("B"), py::arg("C"), 
          py::arg("N"), py::arg("M"), py::arg("K"), 
          py::arg("bx"), py::arg("by"), py::arg("bk")); 
}