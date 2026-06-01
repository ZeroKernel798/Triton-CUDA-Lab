#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

// TF32 Tensor Core SGEMM —— 终版：bcf + grid swizzle + double buffer，反超 cuBLAS。
// 在 swizzle_bcf（0 bank conflict）之上叠加：
//   1. grid swizzle：把 block 的 global tile 访问轨迹折叠成宽 8 的竖条，
//      让相邻 block 复用同一批 B tile，把 L2 命中率拉满（TC 算力强、访存时延凸显时收益明显）；
//   2. double buffer：As/Bs 各开 2 份 smem，用 cp.async 预取下一 K tile，
//      与当前 tile 的 ldmatrix/mma 计算重叠，隐藏 global->smem 延迟。
// 约束：M,N % 128 == 0，K % 16 == 0。grid swizzle 仅在 gridDim.x % 8 == 0 时启用，
//       否则回退为恒等映射（避免越界）。

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#define FLOAT4(var) (reinterpret_cast<float4 *>(&(var))[0])
#define FLOAT2(var) (reinterpret_cast<float2 *>(&(var))[0])

#define SWIZZLE_A(row, col) ((col) ^ ((((row) >> 1) & 0x3) << 2))
#define SWIZZLE_B_F2(row, col) ((col) ^ (((row) & 0x7) << 3))

#define CP_ASYNC_CG(dst, src)                                                  \
  asm volatile(                                                                \
      "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),       \
      "l"(src), "n"(16))
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))

#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                      \
  asm volatile(                                                                \
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"     \
      : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                                 \
      : "r"(addr))

#define M16N8K8(D0, D1, D2, D3, A0, A1, A2, A3, B0, B1)                        \
  asm volatile(                                                                \
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "                    \
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"      \
      : "+f"(D0), "+f"(D1), "+f"(D2), "+f"(D3)                                 \
      : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1))

template <const int BM = 128, const int BN = 128, const int BK = 16>
__global__ __launch_bounds__(256, 2) void sgemm_tf32_swizzle_bcf_dbf_kernel(float *a, float *b, float *c, int m, int n, int k) {
    // ---- grid swizzle：宽 8 的竖条折叠，提升 L2 复用 ----
    const int SWIZZLE_W = 8;
    int bx, by;
    if (gridDim.x % SWIZZLE_W == 0) {
        int linear_id = blockIdx.y * gridDim.x + blockIdx.x;
        bx = (linear_id % SWIZZLE_W) + (linear_id / (SWIZZLE_W * gridDim.y)) * SWIZZLE_W;
        by = (linear_id / SWIZZLE_W) % gridDim.y;
    } else {
        bx = blockIdx.x;
        by = blockIdx.y;
    }

    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    int load_a_row = tid / 4;               // 0~63 (M)
    int load_a_col = (tid % 4) * 4;         // 0,4,8,12 (K)
    int load_b_row = tid / WARP_SIZE;       // 0~7 (K)
    int load_b_col = (tid % WARP_SIZE) * 4; // 0~124 (N)

    __shared__ float As[2][BM][BK]; // double buffer [M=128][K=16]
    __shared__ float Bs[2][BK][BN]; // double buffer [K=16][N=128]

    int warp_id_m = warp_id / 4;
    int warp_id_n = warp_id % 4;

    float sum[4][4][4] = {0.f};

    // 把"加载第 bk 个 K tile 到 buf"封装成宏，prologue 与主循环复用
#define LOAD_TILE(buf, bk_)                                                                       \
    do {                                                                                          \
        uint32_t sa0 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[buf][load_a_row][SWIZZLE_A(load_a_row, load_a_col)]));       \
        uint32_t sa1 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[buf][load_a_row + 64][SWIZZLE_A(load_a_row + 64, load_a_col)])); \
        CP_ASYNC_CG(sa0, &a[(by * BM + load_a_row) * k + (bk_) + load_a_col]);                    \
        CP_ASYNC_CG(sa1, &a[(by * BM + load_a_row + 64) * k + (bk_) + load_a_col]);               \
        uint32_t sb0 = static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[buf][load_b_row][SWIZZLE_B_F2(load_b_row, load_b_col)]));    \
        uint32_t sb1 = static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[buf][load_b_row + 8][SWIZZLE_B_F2(load_b_row + 8, load_b_col)])); \
        CP_ASYNC_CG(sb0, &b[((bk_) + load_b_row) * n + bx * BN + load_b_col]);                    \
        CP_ASYNC_CG(sb1, &b[((bk_) + load_b_row + 8) * n + bx * BN + load_b_col]);                \
        CP_ASYNC_COMMIT_GROUP();                                                                   \
    } while (0)

    int num_tiles = k / BK;

    // prologue：预取 tile 0 到 buffer 0
    LOAD_TILE(0, 0);

    for (int tile = 0; tile < num_tiles; ++tile) {
        int cur = tile & 1;

        if (tile + 1 < num_tiles) {
            // 预取下一个 tile 到另一 buffer（与当前 tile 计算重叠）
            LOAD_TILE((tile + 1) & 1, (tile + 1) * BK);
            CP_ASYNC_WAIT_GROUP(1); // 保留最新一组在途，确保当前 tile 已就绪
        } else {
            CP_ASYNC_WAIT_GROUP(0); // 最后一个 tile，等全部完成
        }
        __syncthreads();

        // 计算当前 tile
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 8;

            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
                int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
                int a_col = k_offset + (lane_id / 16) * 4;
                uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&As[cur][a_row][SWIZZLE_A(a_row, a_col)]));
                LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
            }

#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                int n_base = warp_id_n * 32 + n_idx * 8;
                int b_col = n_base + (lane_id / 4);
                int b_row_0 = k_offset + (lane_id % 4);
                int b_row_1 = k_offset + (lane_id % 4) + 4;
                reg_b[n_idx][0] = __float_as_uint(Bs[cur][b_row_0][SWIZZLE_B_F2(b_row_0, b_col)]);
                reg_b[n_idx][1] = __float_as_uint(Bs[cur][b_row_1][SWIZZLE_B_F2(b_row_1, b_col)]);
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
        __syncthreads(); // 计算完成后再允许覆写本 buffer
    }
#undef LOAD_TILE

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

    sgemm_tf32_swizzle_bcf_dbf_kernel<BM, BN, BK><<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("solve", &solve, "TF32 Tensor Core SGEMM (bcf + grid swizzle + double buffer)",
          py::arg("A"), py::arg("B"), py::arg("C"),
          py::arg("N"), py::arg("M"), py::arg("K"));
}
