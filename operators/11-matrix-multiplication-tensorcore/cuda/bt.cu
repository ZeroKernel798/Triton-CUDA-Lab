#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

// TF32 Tensor Core SGEMM —— 基础版 (bt = B transposed)。
// 思路：A 用 cp.async 直达 smem；B 用 LDG + 手动转置写入 smem；
//       再用 ldmatrix 把 A/B 装入寄存器，mma.m16n8k8(tf32) 计算。
// 暂不处理 bank conflict / 访存合并，作为优化链的起点。
// 约束：M,N % 128 == 0，K % 16 == 0（不含越界 fallback）。

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#define FLOAT4(var) (reinterpret_cast<float4 *>(&(var))[0])
#define FLOAT2(var) (reinterpret_cast<float2 *>(&(var))[0])
#define CFLOAT4(var) (reinterpret_cast<const float4 *>(&(var))[0])

// 16B 对齐的 cp.async（绕过 L1，直达 smem）
#define CP_ASYNC_CG(dst, src)                                                  \
  asm volatile(                                                                \
      "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),       \
      "l"(src), "n"(16))
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_GROUP_0() asm volatile("cp.async.wait_group 0;\n" ::)

// ldmatrix：从 smem 把矩阵碎片装进寄存器（b16 粒度，承载 tf32 数据）
#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                      \
  asm volatile(                                                                \
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"     \
      : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                                 \
      : "r"(addr))
#define LDMATRIX_X2(R0, R1, addr)                                              \
  asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"    \
               : "=r"(R0), "=r"(R1)                                            \
               : "r"(addr))

// TF32 mma：D(16x8) = A(16x8) * B(8x8) + C(16x8)，原地累加（C == D）
#define M16N8K8(D0, D1, D2, D3, A0, A1, A2, A3, B0, B1)                        \
  asm volatile(                                                                \
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "                    \
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"      \
      : "+f"(D0), "+f"(D1), "+f"(D2), "+f"(D3)                                 \
      : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1))

// a block calculate c[128][128]
template <const int BM = 128, const int BN = 128, const int BK = 16>
__global__ __launch_bounds__(256, 2) void sgemm_tf32_bt_kernel(float *a, float *b, float *c, int m, int n, int k) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tid = threadIdx.x; // 0~255
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // 搬运映射
    int load_a_row = tid / 4;               // 0~63
    int load_a_col = (tid % 4) * 4;         // 0,4,8,12
    int load_b_row = tid / WARP_SIZE;       // 0~7  (K 维度)
    int load_b_col = (tid % WARP_SIZE) * 4; // 0~124 (N 维度)

    // A 保持 行优先，B 转置为 列优先
    __shared__ float As[BM][BK];
    __shared__ float Bs[BN][BK];

    // 2x4 warp tiling
    // 一列 warp 负责上下 64x128，每个 warp 负责 64 x 32 的 C 矩阵块
    int warp_id_m = warp_id / 4; // 0, 1
    int warp_id_n = warp_id % 4; // 0, 1, 2, 3

    // 寄存器总量：M 维 4 块 * N 维 4 块 * 每块 4 个寄存器 = 64
    float sum[4][4][4] = {0.f};

    // 主循环
    for (int bk = 0; bk < k; bk += BK) {

        // 1. 使用 cp.async 加载 A 矩阵 (16 bytes 对齐)
        uint32_t smem_a0 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_row][load_a_col]));
        uint32_t smem_a1 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_row + 64][load_a_col]));

        float *global_a0 = &a[(by * BM + load_a_row) * k + bk + load_a_col];
        float *global_a1 = &a[(by * BM + load_a_row + 64) * k + bk + load_a_col];

        CP_ASYNC_CG(smem_a0, global_a0);
        CP_ASYNC_CG(smem_a1, global_a1);
        // 提交所有的异步拷贝任务
        CP_ASYNC_COMMIT_GROUP();

        // 2. 加载 B 矩阵并手动转置写入 smem
        float4 tmp_b0 = FLOAT4(b[(bk + load_b_row) * n + bx * BN + load_b_col]);
        float4 tmp_b1 = FLOAT4(b[(bk + load_b_row + 8) * n + bx * BN + load_b_col]);

        // 将读取的 B 手动转置写入 smem
        Bs[load_b_col + 0][load_b_row] = tmp_b0.x;
        Bs[load_b_col + 1][load_b_row] = tmp_b0.y;
        Bs[load_b_col + 2][load_b_row] = tmp_b0.z;
        Bs[load_b_col + 3][load_b_row] = tmp_b0.w;
        Bs[load_b_col + 0][load_b_row + 8] = tmp_b1.x;
        Bs[load_b_col + 1][load_b_row + 8] = tmp_b1.y;
        Bs[load_b_col + 2][load_b_row + 8] = tmp_b1.z;
        Bs[load_b_col + 3][load_b_row + 8] = tmp_b1.w;

        CP_ASYNC_WAIT_GROUP_0();
        __syncthreads();

        // 3. Tensor Core 计算阶段 (K 维度走 2 步，每次消耗 8 个 K)
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 8;

            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // 发射 4 次 ldmatrix 获取 A 矩阵块 (4 * 16 = 64 行)
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
                // warp_id_m 跨度是 64
                int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
                int a_col = k_offset + (lane_id / 16) * 4;
                uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&As[a_row][a_col]));
                LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
            }

            // 发射 4 次 ldmatrix 获取 B 矩阵块 (4 * 8 = 32 列)
#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                // warp_id_n 跨度是 32
                int b_row = warp_id_n * 32 + n_idx * 8 + (lane_id % 8);
                int b_col = k_offset + ((lane_id / 8) % 2) * 4;
                uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[b_row][b_col]));
                LDMATRIX_X2(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
            }

            // MMA 核心运算：4x4 的 1688
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
                for (int n_idx = 0; n_idx < 4; ++n_idx) {
                    M16N8K8(sum[m_idx][n_idx][0],
                            sum[m_idx][n_idx][1],
                            sum[m_idx][n_idx][2],
                            sum[m_idx][n_idx][3],
                            reg_a[m_idx][0],
                            reg_a[m_idx][1],
                            reg_a[m_idx][2],
                            reg_a[m_idx][3],
                            reg_b[n_idx][0],
                            reg_b[n_idx][1]);
                }
            }
        }
        __syncthreads();
    }

    // Tensor Core m16n8k8 C 寄存器碎片的标准排布映射法则
    int t_row = lane_id / 4;       // 0~7
    int t_col = (lane_id % 4) * 2; // 0, 2, 4, 6

#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            int c_base_row = by * BM + warp_id_m * 64 + m_idx * 16;
            int c_base_col = bx * BN + warp_id_n * 32 + n_idx * 8;

            FLOAT2(c[(c_base_row + t_row) * n + c_base_col + t_col]) = FLOAT2(sum[m_idx][n_idx][0]);
            FLOAT2(c[(c_base_row + t_row + 8) * n + c_base_col + t_col]) = FLOAT2(sum[m_idx][n_idx][2]);
        }
    }
}

void solve(torch::Tensor A, torch::Tensor B, torch::Tensor C, int N, int M, int K) {
    float *d_A = A.data_ptr<float>();
    float *d_B = B.data_ptr<float>();
    float *d_C = C.data_ptr<float>();

    constexpr int BM = 128, BN = 128, BK = 16;
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((N + BN - 1) / BN, (M + BM - 1) / BM);

    sgemm_tf32_bt_kernel<BM, BN, BK><<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("solve", &solve, "TF32 Tensor Core SGEMM (bt)",
          py::arg("A"), py::arg("B"), py::arg("C"),
          py::arg("N"), py::arg("M"), py::arg("K"));
}
