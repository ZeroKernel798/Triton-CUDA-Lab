#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// FlashAttention-2 with Tensor Core (mma.sync PTX), bf16 输入 / fp32 累加。
// 对照 LeetCUDA: kernels/flash-attn/mma/basic/flash_attn_mma_split_q.cu
//                + flash_attn_mma_tiling_qk_F32F16F16F32.cu (fp32 累加路径)
// 适配本仓库: 单 batch/head, 2D 形状 Q[M,d] K[N,d] V[N,d] O[M,d], solve(Q,K,V,O,M,N,d)。
//
// 本文件是 *基础版*: kStage=1 (无 cp.async 双缓冲), kPad=0 (无 swizzle)。优化版见 v2。
// 布局: MMA = m16n8k16, Br=16*4=64, Bc=8*8=64, 4 warps(128 线程), split-Q。
// 重要: bf16 没有 bf16 累加的 mma, 必须 fp32 累加 (f32.bf16.bf16.f32),
//       因此 S/O/D 累加器是 4×f32; softmax 后把 P 转 bf16 压回 R_S 复用为 mma 的 A。
//
// 约束: M 是 Br(64) 的倍数, N 是 Bc(64) 的倍数, d ∈ {64,128} (tile 内不做尾部 mask)。

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// ---- PTX 宏 (摘自 LeetCUDA utils.h, mma 换 bf16 变体) ----
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

// mma m16n8k16, bf16 输入, f32 累加。寄存器接口与 LeetCUDA HMMA16816F32 一致,
// 仅把 PTX 的 .f16.f16 换成 .bf16.bf16。A=4×b32(8 bf16), B=2×b32(4 bf16), C/D=4×f32。
#define BF16_MMA_16816_F32(RD0, RD1, RD2, RD3, RA0, RA1, RA2, RA3, RB0, RB1,   \
                           RC0, RC1, RC2, RC3)                                 \
  asm volatile(                                                                \
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "                   \
      "{%0,  %1,  %2,  %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, "      \
      "%13};\n"                                                                \
      : "=r"(RD0), "=r"(RD1), "=r"(RD2), "=r"(RD3)                             \
      : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0),  \
        "r"(RC1), "r"(RC2), "r"(RC3))

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

// kHeadDim ∈ {64,128}; 固定 Br=Bc=64, 4 warps, kStage=1, kPad=0。
template <int kHeadDim>
__global__ void flash_attn_mma_v1_kernel(const __nv_bfloat16 *Q,
                                         const __nv_bfloat16 *K,
                                         const __nv_bfloat16 *V,
                                         __nv_bfloat16 *O, int M, int N,
                                         float scale) {
  constexpr int kMmaAtomM = 16, kMmaAtomN = 8, kMmaAtomK = 16;
  constexpr int kWarpTileSeqLenK = 8;                  // Bc=8*8=64
  constexpr int kWarpTileHeadDimV = kHeadDim / 8;      // d=64->8, d=128->16
  constexpr int Br = 64, Bc = 64;
  constexpr int kPad = 0;
  const int Tc = (N + Bc - 1) / Bc;

  const int Q_tile_id = blockIdx.x; // 沿 M 的 Q tile
  const int tid = threadIdx.x;      // 0~127
  const int warp_id = tid / WARP_SIZE; // 0~3 = warp_QP
  const int lane_id = tid % WARP_SIZE; // 0~31

  // gmem -> smem 映射 (128 线程, 每行 2 线程)
  int load_smem_Q_Br = tid / (128 / Br);              // tid/2, 0~63
  int load_smem_Q_d = (tid % (128 / Br)) * (kHeadDim / (128 / Br)); // (tid%2)*(d/2)
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
  __nv_bfloat16 *K_smem = Q_smem + Q_tile_size;
  __nv_bfloat16 *V_smem = K_smem + KV_tile_size;
  uint32_t smem_Q_base = __cvta_generic_to_shared(Q_smem);
  uint32_t smem_K_base = __cvta_generic_to_shared(K_smem);
  uint32_t smem_V_base = __cvta_generic_to_shared(V_smem);

  // 在线 softmax 状态 (每线程持 2 行: row r 与 row r+8)
  float lane_row_max_old[2] = {-INFINITY, -INFINITY};
  float lane_row_sum_old[2] = {0.0f, 0.0f};

  uint32_t R_Q[4];
  uint32_t R_K[kWarpTileSeqLenK][2];
  uint32_t R_V[kWarpTileHeadDimV][2];
  uint32_t R_S[kWarpTileSeqLenK][4];       // f32 累加 (8 个 MMA-N)
  uint32_t R_O[kWarpTileHeadDimV][4];      // f32 累加 P@V
  uint32_t R_D[kWarpTileHeadDimV][4];      // f32 输出累加器
#pragma unroll
  for (int j = 0; j < kWarpTileHeadDimV; ++j) {
#pragma unroll
    for (int r = 0; r < 4; ++r)
      R_D[j][r] = 0u;
  }

  // 一次性载入 Q[Br,d] -> smem
  {
    int gmem_addr = load_gmem_Q_Br * kHeadDim + load_smem_Q_d;
    uint32_t smem_ptr =
        smem_Q_base +
        (load_smem_Q_Br * (kHeadDim + kPad) + load_smem_Q_d) * sizeof(__nv_bfloat16);
#pragma unroll
    for (int i = 0; i < (kHeadDim / (128 / Br)); i += 8) {
      asm volatile(
          "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(
              smem_ptr + i * 2),
          "l"(&Q[gmem_addr + i]), "n"(16));
    }
    asm volatile("cp.async.commit_group;\n" ::);
  }

  // 外层: 遍历 KV tile
#pragma unroll 1
  for (int tile = 0; tile < Tc; ++tile) {
    // 载入当前 K[Bc,d], V[Bc,d] -> smem
    {
      int kv_row = tile * Bc + load_smem_K_Bc;
      int gmem_addr = kv_row * kHeadDim + load_smem_K_d;
      uint32_t smem_ptr =
          smem_K_base +
          (load_smem_K_Bc * (kHeadDim + kPad) + load_smem_K_d) * sizeof(__nv_bfloat16);
#pragma unroll
      for (int i = 0; i < (kHeadDim / (128 / Bc)); i += 8) {
        asm volatile(
            "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(
                smem_ptr + i * 2),
            "l"(&K[gmem_addr + i]), "n"(16));
      }
      asm volatile("cp.async.commit_group;\n" ::);
    }
    {
      int kv_row = tile * Bc + load_smem_V_Bc;
      int gmem_addr = kv_row * kHeadDim + load_smem_V_d;
      uint32_t smem_ptr =
          smem_V_base +
          (load_smem_V_Bc * (kHeadDim + kPad) + load_smem_V_d) * sizeof(__nv_bfloat16);
#pragma unroll
      for (int i = 0; i < (kHeadDim / (128 / Bc)); i += 8) {
        asm volatile(
            "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(
                smem_ptr + i * 2),
            "l"(&V[gmem_addr + i]), "n"(16));
      }
      asm volatile("cp.async.commit_group;\n" ::);
    }
    // 等 Q、K 就绪 (留 V 这组在途)
    asm volatile("cp.async.wait_group %0;\n" ::"n"(1));
    __syncthreads();

    // ---- S = Q @ K^T, 沿 d 累加 (f32) ----
#pragma unroll
    for (int j = 0; j < kWarpTileSeqLenK; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r)
        R_S[j][r] = 0u;

#pragma unroll
    for (int tile_K_d = 0; tile_K_d < (kHeadDim / kMmaAtomK); ++tile_K_d) {
      // ldmatrix.x4 取 Q fragment (A)
      int lane_smem_Q_Br = warp_id * kMmaAtomM + lane_id % 16;
      int lane_smem_Q_d = tile_K_d * kMmaAtomK + (lane_id / 16) * 8;
      uint32_t Q_ptr = smem_Q_base + (lane_smem_Q_Br * (kHeadDim + kPad) +
                                      lane_smem_Q_d) *
                                         sizeof(__nv_bfloat16);
      LDMATRIX_X4(R_Q[0], R_Q[1], R_Q[2], R_Q[3], Q_ptr);

      // ldmatrix.x2 取 K fragment (B), K[Bc,d] row-major 即 K^T col-major
#pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j) {
        int lane_smem_K_Bc = j * kMmaAtomN + lane_id % 8;
        int lane_smem_K_d = tile_K_d * kMmaAtomK + ((lane_id / 8) % 2) * 8;
        uint32_t K_ptr = smem_K_base + (lane_smem_K_Bc * (kHeadDim + kPad) +
                                        lane_smem_K_d) *
                                           sizeof(__nv_bfloat16);
        LDMATRIX_X2(R_K[j][0], R_K[j][1], K_ptr);
      }
#pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j) {
        BF16_MMA_16816_F32(R_S[j][0], R_S[j][1], R_S[j][2], R_S[j][3], R_Q[0],
                           R_Q[1], R_Q[2], R_Q[3], R_K[j][0], R_K[j][1],
                           R_S[j][0], R_S[j][1], R_S[j][2], R_S[j][3]);
      }
    }
    __syncthreads();

    // ---- 在线 softmax (寄存器内) ----
    // C fragment: c0,c1 属 row(warp_id*16 + lane/4); c2,c3 属 row+8。
    float row_max_new[2] = {-INFINITY, -INFINITY};
    float row_sum_new[2] = {0.0f, 0.0f};
#pragma unroll
    for (int j = 0; j < kWarpTileSeqLenK; ++j) {
      float *s = reinterpret_cast<float *>(&R_S[j][0]);
      float m0 = max(s[0], s[1]) * scale; // c0,c1
      float m1 = max(s[2], s[3]) * scale; // c2,c3
      row_max_new[0] = max(row_max_new[0], m0);
      row_max_new[1] = max(row_max_new[1], m1);
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
      // 转 bf16 压回 R_S, 复用为 P@V 的 A: R_S[j][0]=(p0,p1), R_S[j][1]=(p2,p3)
      p[0] = __float2bfloat16_rn(e0);
      p[1] = __float2bfloat16_rn(e1);
      p[2] = __float2bfloat16_rn(e2);
      p[3] = __float2bfloat16_rn(e3);
    }
    row_sum_new[0] = warp_reduce_sum<float, 4>(row_sum_new[0]);
    row_sum_new[1] = warp_reduce_sum<float, 4>(row_sum_new[1]);

    // 等 V 就绪
    asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
    __syncthreads();

    // ---- O += P @ V (f32 累加) ----
#pragma unroll
    for (int j = 0; j < kWarpTileHeadDimV; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r)
        R_O[j][r] = 0u;

#pragma unroll
    for (int tile_V_Bc = 0; tile_V_Bc < (Bc / kMmaAtomK); ++tile_V_Bc) {
      // ldmatrix.x2.trans 取 V fragment (B)
#pragma unroll
      for (int j = 0; j < kWarpTileHeadDimV; ++j) {
        int lane_smem_V_Bc = tile_V_Bc * kMmaAtomK + lane_id % 16;
        int lane_smem_V_d = j * kMmaAtomN;
        uint32_t V_ptr = smem_V_base + (lane_smem_V_Bc * (kHeadDim + kPad) +
                                        lane_smem_V_d) *
                                           sizeof(__nv_bfloat16);
        LDMATRIX_X2_T(R_V[j][0], R_V[j][1], V_ptr);
      }
      // A 取自 P (复用 R_S): tile_V_Bc 选 P 的列块 [w, w+1]
      int w = tile_V_Bc * 2;
#pragma unroll
      for (int j = 0; j < kWarpTileHeadDimV; ++j) {
        BF16_MMA_16816_F32(R_O[j][0], R_O[j][1], R_O[j][2], R_O[j][3],
                           R_S[w][0], R_S[w][1], R_S[w + 1][0], R_S[w + 1][1],
                           R_V[j][0], R_V[j][1], R_O[j][0], R_O[j][1],
                           R_O[j][2], R_O[j][3]);
      }
    }
    __syncthreads();

    // ---- rescale O, 更新 l, m ----
    float m_old_0 = (tile > 0) ? lane_row_max_old[0] : m_new_0;
    float m_old_1 = (tile > 0) ? lane_row_max_old[1] : m_new_1;
    float resc_0 = __expf(m_old_0 - m_new_0);
    float resc_1 = __expf(m_old_1 - m_new_1);
#pragma unroll
    for (int j = 0; j < kWarpTileHeadDimV; ++j) {
      float *d = reinterpret_cast<float *>(&R_D[j][0]);
      float *o = reinterpret_cast<float *>(&R_O[j][0]);
      d[0] = __fmaf_rn(resc_0, d[0], o[0]); // c0,c1 -> row r
      d[1] = __fmaf_rn(resc_0, d[1], o[1]);
      d[2] = __fmaf_rn(resc_1, d[2], o[2]); // c2,c3 -> row r+8
      d[3] = __fmaf_rn(resc_1, d[3], o[3]);
    }
    lane_row_sum_old[0] = __fmaf_rn(resc_0, lane_row_sum_old[0], row_sum_new[0]);
    lane_row_sum_old[1] = __fmaf_rn(resc_1, lane_row_sum_old[1], row_sum_new[1]);
    lane_row_max_old[0] = m_new_0;
    lane_row_max_old[1] = m_new_1;

    __syncthreads();
  } // end KV loop

  // ---- 最终归一化 O /= l, 转 bf16 写回 ----
  float inv_l_0 = __frcp_rn(lane_row_sum_old[0]);
  float inv_l_1 = __frcp_rn(lane_row_sum_old[1]);
#pragma unroll
  for (int j = 0; j < kWarpTileHeadDimV; ++j) {
    float *d = reinterpret_cast<float *>(&R_D[j][0]);
    // pack 成 2 个 bf16x2: z0=(c0,c1) row r, z1=(c2,c3) row r+8
    float2 f0 = make_float2(d[0] * inv_l_0, d[1] * inv_l_0);
    float2 f1 = make_float2(d[2] * inv_l_1, d[3] * inv_l_1);
    __nv_bfloat162 b0 = __float22bfloat162_rn(f0);
    __nv_bfloat162 b1 = __float22bfloat162_rn(f1);
    uint32_t z0 = *reinterpret_cast<uint32_t *>(&b0);
    uint32_t z1 = *reinterpret_cast<uint32_t *>(&b1);

    // collective store: 同组 4 线程的 z 拼成 128-bit (8 bf16 = 一行 8 个 d 列)
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
      int row0 = Q_tile_id * Br + warp_id * kMmaAtomM + lane_id / 4; // 0~7
      int col = j * kMmaAtomN;                                       // d 偏移
      int addr0 = (row0 + 0) * kHeadDim + col;
      int addr1 = (row0 + 8) * kHeadDim + col;
      LDST128BITS(O[addr0]) = LDST128BITS(Z0[0]);
      LDST128BITS(O[addr1]) = LDST128BITS(Z1[0]);
    }
  }
}

void solve(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
           int M, int N, int d) {
  const float scale = 1.0f / sqrtf((float)d);
  constexpr int Br = 64;
  dim3 grid((M + Br - 1) / Br);
  dim3 block(128);

  auto launch = [&](auto headdim) {
    constexpr int kHeadDim = decltype(headdim)::value;
    constexpr int kPad = 0;
    size_t smem = (Br * (kHeadDim + kPad) + 2 * 64 * (kHeadDim + kPad)) *
                  sizeof(__nv_bfloat16);
    cudaFuncSetAttribute(flash_attn_mma_v1_kernel<kHeadDim>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
    flash_attn_mma_v1_kernel<kHeadDim><<<grid, block, smem>>>(
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
    throw std::runtime_error("flashattentionv1(TC): only d=64/128 supported");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  namespace py = pybind11;
  m.def("solve", &solve, "FlashAttention V1 (Tensor Core, bf16, FA2 split-Q, base)",
        py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("O"), py::arg("M"),
        py::arg("N"), py::arg("d"));
}
