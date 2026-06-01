#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// FlashAttention-2 with Tensor Core (mma.sync), bf16 / fp32 累加 —— *优化版*。
// 在 v1 (基础版) 之上加:
//   1. kPad=8: smem 每行 padding, 错开 bank, 降低 ldmatrix / cp.async 的 bank conflict;
//   2. kStage=2: K tile 用 cp.async 双缓冲 (计算当前 tile 时预取下一 tile K), 隐藏 g2s 延迟。
// 其余 (split-Q 布局、bf16 mma f32 累加、寄存器 online softmax、P 复用 R_S) 与 v1 相同。
// 对照 LeetCUDA flash_attn_mma_split_q.cu 的 stage>1 路径。
// 约束: M%64==0, N%64==0, d ∈ {64,128}。第一版需在 4090 上验证迭代。

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                      \
  asm volatile(                                                                \
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"     \
      : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                                 \
      : "r"(addr))

#define LDMATRIX_X2(R0, R1, addr)                                              \
  asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"    \
               : "=r"(R0), "=r"(R1)                                            \
               : "r"(addr))

#define LDMATRIX_X2_T(R0, R1, addr)                                            \
  asm volatile(                                                                \
      "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"       \
      : "=r"(R0), "=r"(R1)                                                     \
      : "r"(addr))

#define BF16_MMA_16816_F32(RD0, RD1, RD2, RD3, RA0, RA1, RA2, RA3, RB0, RB1,   \
                           RC0, RC1, RC2, RC3)                                 \
  asm volatile(                                                                \
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "                   \
      "{%0,  %1,  %2,  %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, "      \
      "%13};\n"                                                                \
      : "=r"(RD0), "=r"(RD1), "=r"(RD2), "=r"(RD3)                             \
      : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0),  \
        "r"(RC1), "r"(RC2), "r"(RC3))

#define CP_ASYNC_CG(dst, src)                                                  \
  asm volatile(                                                                \
      "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),       \
      "l"(src), "n"(16))
#define CP_ASYNC_COMMIT() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))

template <typename T, int kWarpSize = WARP_SIZE>
__device__ __forceinline__ T warp_reduce_max(T val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1)
    val = max(val, __shfl_xor_sync(0xffffffff, val, mask, kWarpSize));
  return val;
}
template <typename T, int kWarpSize = WARP_SIZE>
__device__ __forceinline__ T warp_reduce_sum(T val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1)
    val += __shfl_xor_sync(0xffffffff, val, mask, kWarpSize);
  return val;
}

template <int kHeadDim>
__global__ void flash_attn_mma_v2_kernel(const __nv_bfloat16 *Q,
                                         const __nv_bfloat16 *K,
                                         const __nv_bfloat16 *V,
                                         __nv_bfloat16 *O, int M, int N,
                                         float scale) {
  constexpr int kMmaAtomM = 16, kMmaAtomN = 8, kMmaAtomK = 16;
  constexpr int kWarpTileSeqLenK = 8;
  constexpr int kWarpTileHeadDimV = kHeadDim / 8;
  constexpr int Br = 64, Bc = 64;
  constexpr int kPad = 8;       // swizzle padding
  constexpr int kStage = 2;     // K 双缓冲
  const int Tc = (N + Bc - 1) / Bc;

  const int Q_tile_id = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  int load_smem_Q_Br = tid / (128 / Br);
  int load_smem_Q_d = (tid % (128 / Br)) * (kHeadDim / (128 / Br));
  int load_smem_K_Bc = tid / (128 / Bc);
  int load_smem_K_d = (tid % (128 / Bc)) * (kHeadDim / (128 / Bc));
  int load_smem_V_Bc = load_smem_K_Bc;
  int load_smem_V_d = load_smem_K_d;
  int load_gmem_Q_Br = Q_tile_id * Br + load_smem_Q_Br;
  if (load_gmem_Q_Br >= M)
    return;

  extern __shared__ __nv_bfloat16 smem[];
  constexpr int Q_tile_size = Br * (kHeadDim + kPad);
  constexpr int KV_tile_size = Bc * (kHeadDim + kPad);
  __nv_bfloat16 *Q_smem = smem;
  __nv_bfloat16 *K_smem = Q_smem + Q_tile_size;       // 2 * KV_tile_size
  __nv_bfloat16 *V_smem = K_smem + kStage * KV_tile_size;
  uint32_t smem_Q_base = __cvta_generic_to_shared(Q_smem);
  uint32_t smem_K_base = __cvta_generic_to_shared(K_smem);
  uint32_t smem_V_base = __cvta_generic_to_shared(V_smem);

  float lane_row_max_old[2] = {-INFINITY, -INFINITY};
  float lane_row_sum_old[2] = {0.0f, 0.0f};

  uint32_t R_Q[4];
  uint32_t R_K[kWarpTileSeqLenK][2];
  uint32_t R_V[kWarpTileHeadDimV][2];
  uint32_t R_S[kWarpTileSeqLenK][4];
  uint32_t R_O[kWarpTileHeadDimV][4];
  uint32_t R_D[kWarpTileHeadDimV][4];
#pragma unroll
  for (int j = 0; j < kWarpTileHeadDimV; ++j)
#pragma unroll
    for (int r = 0; r < 4; ++r)
      R_D[j][r] = 0u;

  // 预取 Q + K[stage0]
  {
    int gmem = load_gmem_Q_Br * kHeadDim + load_smem_Q_d;
    uint32_t sp = smem_Q_base + (load_smem_Q_Br * (kHeadDim + kPad) +
                                 load_smem_Q_d) *
                                    sizeof(__nv_bfloat16);
#pragma unroll
    for (int i = 0; i < (kHeadDim / (128 / Br)); i += 8)
      CP_ASYNC_CG(sp + i * 2, &Q[gmem + i]);
    CP_ASYNC_COMMIT();
  }
  {
    int kv_row = 0 * Bc + load_smem_K_Bc;
    int gmem = kv_row * kHeadDim + load_smem_K_d;
    uint32_t sp = smem_K_base + (0 * KV_tile_size +
                                 load_smem_K_Bc * (kHeadDim + kPad) +
                                 load_smem_K_d) *
                                    sizeof(__nv_bfloat16);
#pragma unroll
    for (int i = 0; i < (kHeadDim / (128 / Bc)); i += 8)
      CP_ASYNC_CG(sp + i * 2, &K[gmem + i]);
    CP_ASYNC_COMMIT();
  }
  CP_ASYNC_WAIT(0); // 等 Q、K0
  __syncthreads();

#pragma unroll 1
  for (int tile = 0; tile < Tc; ++tile) {
    int smem_sel = tile % kStage;
    int smem_sel_next = (tile + 1) % kStage;

    // 预取当前 V (单缓冲)
    {
      int kv_row = tile * Bc + load_smem_V_Bc;
      int gmem = kv_row * kHeadDim + load_smem_V_d;
      uint32_t sp = smem_V_base + (load_smem_V_Bc * (kHeadDim + kPad) +
                                   load_smem_V_d) *
                                      sizeof(__nv_bfloat16);
#pragma unroll
      for (int i = 0; i < (kHeadDim / (128 / Bc)); i += 8)
        CP_ASYNC_CG(sp + i * 2, &V[gmem + i]);
      CP_ASYNC_COMMIT();
    }
    // 预取下一 tile 的 K -> smem_sel_next (双缓冲)
    if ((tile + 1) < Tc) {
      int kv_row = (tile + 1) * Bc + load_smem_K_Bc;
      int gmem = kv_row * kHeadDim + load_smem_K_d;
      uint32_t sp = smem_K_base + (smem_sel_next * KV_tile_size +
                                   load_smem_K_Bc * (kHeadDim + kPad) +
                                   load_smem_K_d) *
                                      sizeof(__nv_bfloat16);
#pragma unroll
      for (int i = 0; i < (kHeadDim / (128 / Bc)); i += 8)
        CP_ASYNC_CG(sp + i * 2, &K[gmem + i]);
      CP_ASYNC_COMMIT();
    }

    // ---- S = Q @ K^T (用已就绪的 K[smem_sel]) ----
#pragma unroll
    for (int j = 0; j < kWarpTileSeqLenK; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r)
        R_S[j][r] = 0u;

#pragma unroll
    for (int tile_K_d = 0; tile_K_d < (kHeadDim / kMmaAtomK); ++tile_K_d) {
      int lane_smem_Q_Br = warp_id * kMmaAtomM + lane_id % 16;
      int lane_smem_Q_d = tile_K_d * kMmaAtomK + (lane_id / 16) * 8;
      uint32_t Q_ptr = smem_Q_base + (lane_smem_Q_Br * (kHeadDim + kPad) +
                                      lane_smem_Q_d) *
                                         sizeof(__nv_bfloat16);
      LDMATRIX_X4(R_Q[0], R_Q[1], R_Q[2], R_Q[3], Q_ptr);
#pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j) {
        int lane_smem_K_Bc = j * kMmaAtomN + lane_id % 8;
        int lane_smem_K_d = tile_K_d * kMmaAtomK + ((lane_id / 8) % 2) * 8;
        uint32_t K_ptr = smem_K_base + (smem_sel * KV_tile_size +
                                        lane_smem_K_Bc * (kHeadDim + kPad) +
                                        lane_smem_K_d) *
                                           sizeof(__nv_bfloat16);
        LDMATRIX_X2(R_K[j][0], R_K[j][1], K_ptr);
      }
#pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j)
        BF16_MMA_16816_F32(R_S[j][0], R_S[j][1], R_S[j][2], R_S[j][3], R_Q[0],
                           R_Q[1], R_Q[2], R_Q[3], R_K[j][0], R_K[j][1],
                           R_S[j][0], R_S[j][1], R_S[j][2], R_S[j][3]);
    }
    __syncthreads();

    // ---- online softmax ----
    float row_max_new[2] = {-INFINITY, -INFINITY};
    float row_sum_new[2] = {0.0f, 0.0f};
#pragma unroll
    for (int j = 0; j < kWarpTileSeqLenK; ++j) {
      float *s = reinterpret_cast<float *>(&R_S[j][0]);
      row_max_new[0] = max(row_max_new[0], max(s[0], s[1]) * scale);
      row_max_new[1] = max(row_max_new[1], max(s[2], s[3]) * scale);
    }
    row_max_new[0] = warp_reduce_max<float, 4>(row_max_new[0]);
    row_max_new[1] = warp_reduce_max<float, 4>(row_max_new[1]);
    float m_new_0 = max(lane_row_max_old[0], row_max_new[0]);
    float m_new_1 = max(lane_row_max_old[1], row_max_new[1]);

#pragma unroll
    for (int j = 0; j < kWarpTileSeqLenK; ++j) {
      float *s = reinterpret_cast<float *>(&R_S[j][0]);
      __nv_bfloat16 *p = reinterpret_cast<__nv_bfloat16 *>(&R_S[j][0]);
      float e0 = __expf(__fmaf_rn(s[0], scale, -m_new_0));
      float e1 = __expf(__fmaf_rn(s[1], scale, -m_new_0));
      float e2 = __expf(__fmaf_rn(s[2], scale, -m_new_1));
      float e3 = __expf(__fmaf_rn(s[3], scale, -m_new_1));
      row_sum_new[0] += (e0 + e1);
      row_sum_new[1] += (e2 + e3);
      p[0] = __float2bfloat16_rn(e0);
      p[1] = __float2bfloat16_rn(e1);
      p[2] = __float2bfloat16_rn(e2);
      p[3] = __float2bfloat16_rn(e3);
    }
    row_sum_new[0] = warp_reduce_sum<float, 4>(row_sum_new[0]);
    row_sum_new[1] = warp_reduce_sum<float, 4>(row_sum_new[1]);

    // 等 V 就绪 (V 先 commit, next K 后 commit; 留 K 在途)
    if ((tile + 1) < Tc)
      CP_ASYNC_WAIT(1);
    else
      CP_ASYNC_WAIT(0);
    __syncthreads();

    // ---- O += P @ V ----
#pragma unroll
    for (int j = 0; j < kWarpTileHeadDimV; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r)
        R_O[j][r] = 0u;

#pragma unroll
    for (int tile_V_Bc = 0; tile_V_Bc < (Bc / kMmaAtomK); ++tile_V_Bc) {
#pragma unroll
      for (int j = 0; j < kWarpTileHeadDimV; ++j) {
        int lane_smem_V_Bc = tile_V_Bc * kMmaAtomK + lane_id % 16;
        int lane_smem_V_d = j * kMmaAtomN;
        uint32_t V_ptr = smem_V_base + (lane_smem_V_Bc * (kHeadDim + kPad) +
                                        lane_smem_V_d) *
                                           sizeof(__nv_bfloat16);
        LDMATRIX_X2_T(R_V[j][0], R_V[j][1], V_ptr);
      }
      int w = tile_V_Bc * 2;
#pragma unroll
      for (int j = 0; j < kWarpTileHeadDimV; ++j)
        BF16_MMA_16816_F32(R_O[j][0], R_O[j][1], R_O[j][2], R_O[j][3],
                           R_S[w][0], R_S[w][1], R_S[w + 1][0], R_S[w + 1][1],
                           R_V[j][0], R_V[j][1], R_O[j][0], R_O[j][1],
                           R_O[j][2], R_O[j][3]);
    }
    __syncthreads();

    // ---- rescale O, 更新 l/m ----
    float m_old_0 = (tile > 0) ? lane_row_max_old[0] : m_new_0;
    float m_old_1 = (tile > 0) ? lane_row_max_old[1] : m_new_1;
    float resc_0 = __expf(m_old_0 - m_new_0);
    float resc_1 = __expf(m_old_1 - m_new_1);
#pragma unroll
    for (int j = 0; j < kWarpTileHeadDimV; ++j) {
      float *d = reinterpret_cast<float *>(&R_D[j][0]);
      float *o = reinterpret_cast<float *>(&R_O[j][0]);
      d[0] = __fmaf_rn(resc_0, d[0], o[0]);
      d[1] = __fmaf_rn(resc_0, d[1], o[1]);
      d[2] = __fmaf_rn(resc_1, d[2], o[2]);
      d[3] = __fmaf_rn(resc_1, d[3], o[3]);
    }
    lane_row_sum_old[0] = __fmaf_rn(resc_0, lane_row_sum_old[0], row_sum_new[0]);
    lane_row_sum_old[1] = __fmaf_rn(resc_1, lane_row_sum_old[1], row_sum_new[1]);
    lane_row_max_old[0] = m_new_0;
    lane_row_max_old[1] = m_new_1;

    // 等下一 tile 的 K 就绪 (供下轮 S), 且保证本轮 V 用完再被下轮覆盖
    if ((tile + 1) < Tc)
      CP_ASYNC_WAIT(0);
    __syncthreads();
  }

  // ---- 最终归一化 + 写回 ----
  float inv_l_0 = __frcp_rn(lane_row_sum_old[0]);
  float inv_l_1 = __frcp_rn(lane_row_sum_old[1]);
#pragma unroll
  for (int j = 0; j < kWarpTileHeadDimV; ++j) {
    float *d = reinterpret_cast<float *>(&R_D[j][0]);
    __nv_bfloat162 b0 = __float22bfloat162_rn(make_float2(d[0] * inv_l_0, d[1] * inv_l_0));
    __nv_bfloat162 b1 = __float22bfloat162_rn(make_float2(d[2] * inv_l_1, d[3] * inv_l_1));
    uint32_t z0 = *reinterpret_cast<uint32_t *>(&b0);
    uint32_t z1 = *reinterpret_cast<uint32_t *>(&b1);
    uint32_t Z0[4], Z1[4];
    Z0[0] = z0;
    Z1[0] = z1;
    Z0[1] = __shfl_sync(0xffffffff, z0, lane_id + 1, 4);
    Z0[2] = __shfl_sync(0xffffffff, z0, lane_id + 2, 4);
    Z0[3] = __shfl_sync(0xffffffff, z0, lane_id + 3, 4);
    Z1[1] = __shfl_sync(0xffffffff, z1, lane_id + 1, 4);
    Z1[2] = __shfl_sync(0xffffffff, z1, lane_id + 2, 4);
    Z1[3] = __shfl_sync(0xffffffff, z1, lane_id + 3, 4);
    if (lane_id % 4 == 0) {
      int row0 = Q_tile_id * Br + warp_id * kMmaAtomM + lane_id / 4;
      int col = j * kMmaAtomN;
      LDST128BITS(O[(row0 + 0) * kHeadDim + col]) = LDST128BITS(Z0[0]);
      LDST128BITS(O[(row0 + 8) * kHeadDim + col]) = LDST128BITS(Z1[0]);
    }
  }
}

void solve(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
           int M, int N, int d) {
  const float scale = 1.0f / sqrtf((float)d);
  constexpr int Br = 64, kPad = 8, kStage = 2;
  dim3 grid((M + Br - 1) / Br);
  dim3 block(128);

  auto launch = [&](auto headdim) {
    constexpr int kHeadDim = decltype(headdim)::value;
    size_t smem = (Br * (kHeadDim + kPad) + kStage * 64 * (kHeadDim + kPad) +
                   64 * (kHeadDim + kPad)) *
                  sizeof(__nv_bfloat16);
    cudaFuncSetAttribute(flash_attn_mma_v2_kernel<kHeadDim>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
    flash_attn_mma_v2_kernel<kHeadDim><<<grid, block, smem>>>(
        reinterpret_cast<const __nv_bfloat16 *>(Q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(K.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(V.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(O.data_ptr<at::BFloat16>()), M, N,
        scale);
  };

  if (d == 64)
    launch(std::integral_constant<int, 64>{});
  else if (d == 128)
    launch(std::integral_constant<int, 128>{});
  else
    throw std::runtime_error("flashattentionv2(TC): only d=64/128 supported");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  namespace py = pybind11;
  m.def("solve", &solve,
        "FlashAttention V2 (Tensor Core, bf16, FA2 split-Q, cp.async+swizzle)",
        py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("O"), py::arg("M"),
        py::arg("N"), py::arg("d"));
}
