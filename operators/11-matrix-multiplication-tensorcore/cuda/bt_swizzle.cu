#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

// TF32 Tensor Core SGEMM —— bt + XOR swizzle 消除 shared memory bank conflict。
// 在 bt 基础上，对 As / Bs 的列索引（K 维，16 宽）按行索引的有效 bit 做 XOR 扰动，
// 使 float4 对齐的 4 个落点 {0,4,8,12} 错开 bank：
//   As: ldmatrix x4 读取无冲突；
//   Bs: ldmatrix x2 读取无冲突（其读取是 As 模式的子集，复用同一 swizzle 思路），
//       LDG 转置写入降到 ~8-way（受 16-wide + float4 对齐的硬约束限制）。
// 注：原文 Bs 的 #define 与推导式不一致，这里采用推导出的“ldmatrix 0 冲突”版本，
//     以远端数值/ncu 验证为准。
// 约束：M,N % 128 == 0，K % 16 == 0。

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#define FLOAT4(var) (reinterpret_cast<float4 *>(&(var))[0])
#define FLOAT2(var) (reinterpret_cast<float2 *>(&(var))[0])

// As: 用 row 的 bit1~2 扰动 col 的 bit2~3
#define SWIZZLE_A(row, col) ((col) ^ ((((row) >> 1) & 0x3) << 2))
// Bs: 同时利用 row 的 bit1~2 与 bit3~4，保证 ldmatrix 的两行打包不被破坏
#define SWIZZLE_B(row, col) ((col) ^ (((((row) >> 1) ^ ((row) >> 3)) & 0x3) << 2))

#define CP_ASYNC_CG(dst, src)                                                  \
  asm volatile(                                                                \
      "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),       \
      "l"(src), "n"(16))
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_GROUP_0() asm volatile("cp.async.wait_group 0;\n" ::)

#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                      \
  asm volatile(                                                                \
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"     \
      : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                                 \
      : "r"(addr))
#define LDMATRIX_X2(R0, R1, addr)                                              \
  asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"    \
               : "=r"(R0), "=r"(R1)                                            \
               : "r"(addr))

#define M16N8K8(D0, D1, D2, D3, A0, A1, A2, A3, B0, B1)                        \
  asm volatile(                                                                \
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "                    \
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"      \
      : "+f"(D0), "+f"(D1), "+f"(D2), "+f"(D3)                                 \
      : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1))

template <const int BM = 128, const int BN = 128, const int BK = 16>
__global__ __launch_bounds__(256, 2) void sgemm_tf32_bt_swizzle_kernel(float *a, float *b, float *c, int m, int n, int k) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    int load_a_row = tid / 4;
    int load_a_col = (tid % 4) * 4;
    int load_b_row = tid / WARP_SIZE;
    int load_b_col = (tid % WARP_SIZE) * 4;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BN][BK];

    int warp_id_m = warp_id / 4;
    int warp_id_n = warp_id % 4;

    float sum[4][4][4] = {0.f};

    for (int bk = 0; bk < k; bk += BK) {

        // 1. cp.async 加载 A（写入位置做 swizzle）
        uint32_t smem_a0 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_row][SWIZZLE_A(load_a_row, load_a_col)]));
        uint32_t smem_a1 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_row + 64][SWIZZLE_A(load_a_row + 64, load_a_col)]));

        float *global_a0 = &a[(by * BM + load_a_row) * k + bk + load_a_col];
        float *global_a1 = &a[(by * BM + load_a_row + 64) * k + bk + load_a_col];

        CP_ASYNC_CG(smem_a0, global_a0);
        CP_ASYNC_CG(smem_a1, global_a1);
        CP_ASYNC_COMMIT_GROUP();

        // 2. 加载 B 并转置写入 smem（写入位置做 swizzle）
        float4 tmp_b0 = FLOAT4(b[(bk + load_b_row) * n + bx * BN + load_b_col]);
        float4 tmp_b1 = FLOAT4(b[(bk + load_b_row + 8) * n + bx * BN + load_b_col]);

        Bs[load_b_col + 0][SWIZZLE_B(load_b_col + 0, load_b_row)] = tmp_b0.x;
        Bs[load_b_col + 1][SWIZZLE_B(load_b_col + 1, load_b_row)] = tmp_b0.y;
        Bs[load_b_col + 2][SWIZZLE_B(load_b_col + 2, load_b_row)] = tmp_b0.z;
        Bs[load_b_col + 3][SWIZZLE_B(load_b_col + 3, load_b_row)] = tmp_b0.w;
        Bs[load_b_col + 0][SWIZZLE_B(load_b_col + 0, load_b_row + 8)] = tmp_b1.x;
        Bs[load_b_col + 1][SWIZZLE_B(load_b_col + 1, load_b_row + 8)] = tmp_b1.y;
        Bs[load_b_col + 2][SWIZZLE_B(load_b_col + 2, load_b_row + 8)] = tmp_b1.z;
        Bs[load_b_col + 3][SWIZZLE_B(load_b_col + 3, load_b_row + 8)] = tmp_b1.w;

        CP_ASYNC_WAIT_GROUP_0();
        __syncthreads();

        // 3. Tensor Core 计算（读取位置做同样 swizzle）
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 8;

            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
                int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
                int a_col = k_offset + (lane_id / 16) * 4;
                uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&As[a_row][SWIZZLE_A(a_row, a_col)]));
                LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
            }

#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                int b_row = warp_id_n * 32 + n_idx * 8 + (lane_id % 8);
                int b_col = k_offset + ((lane_id / 8) % 2) * 4;
                uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[b_row][SWIZZLE_B(b_row, b_col)]));
                LDMATRIX_X2(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
            }

#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
                for (int n_idx = 0; n_idx < 4; ++n_idx) {
                    M16N8K8(sum[m_idx][n_idx][0], sum[m_idx][n_idx][1], sum[m_idx][n_idx][2], sum[m_idx][n_idx][3],
                            reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3],
                            reg_b[n_idx][0], reg_b[n_idx][1]);
                }
            }
        }
        __syncthreads();
    }

    int t_row = lane_id / 4;
    int t_col = (lane_id % 4) * 2;

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

    sgemm_tf32_bt_swizzle_kernel<BM, BN, BK><<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("solve", &solve, "TF32 Tensor Core SGEMM (bt + swizzle)",
          py::arg("A"), py::arg("B"), py::arg("C"),
          py::arg("N"), py::arg("M"), py::arg("K"));
}
