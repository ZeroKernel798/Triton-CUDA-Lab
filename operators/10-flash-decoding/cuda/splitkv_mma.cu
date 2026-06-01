#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cfloat>

// Flash-Decoding (B 版): GQA group-as-M + Tensor Core + split-KV。
// 把共享同一 KV head 的 group 个 query 堆成 [16, d] (M=16, pad)，对一段 KV 做 attention，
// Q@K^T / P@V 用 bf16 mma (f32 累加) 走 Tensor Core。结构基于 09 的 split-Q mma，单 warp 处理 16 行。
// 阶段1: grid(B*Hkv, num_splits), block=32(1 warp)，算一段 KV 的局部 (m,l,O)，写 partial。
// 阶段2: 沿 num_splits 做 per-query 规约 (与 A/splitkv 相同)。
// 约束: group <= 16 (常见 GQA group=4/8；更大的 MQA 需在 M 维 tiling，未实现)。d ∈ {64,128}。
// 注意: 这是把 09 mma 移植到 decode + split-KV 的第一版，需在 4090 上验证迭代。

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

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

// ---- 阶段 1: group-as-M mma, 一段 KV 的局部 attention ----
template <int HEAD_DIM>
__global__ void flash_decode_mma_stage1_kernel(
    const __nv_bfloat16 *__restrict__ Q, const __nv_bfloat16 *__restrict__ K,
    const __nv_bfloat16 *__restrict__ V, float *__restrict__ O_partial,
    float *__restrict__ m_partial, float *__restrict__ l_partial, int B, int Hq,
    int Hkv, int N, int group, int num_splits, int seg, float scale) {
  constexpr int kMmaAtomK = 16, Bc = 64;
  constexpr int kWarpTileSeqLenK = 8;            // Bc/8
  constexpr int kWarpTileHeadDimV = HEAD_DIM / 8;
  const int lane_id = threadIdx.x; // 0~31 (单 warp)

  const int pid_bkv = blockIdx.x; // b*Hkv + hkv
  const int split = blockIdx.y;
  const int b = pid_bkv / Hkv;
  const int hkv = pid_bkv % Hkv;
  const int q_head0 = hkv * group;

  const long q_base = (long)(b * Hq + q_head0) * HEAD_DIM; // group 个 query 起点
  const long kv_base = (long)(b * Hkv + hkv) * N * HEAD_DIM;
  const int kv_start = split * seg;
  int kv_end = kv_start + seg;
  if (kv_end > N)
    kv_end = N;

  extern __shared__ __nv_bfloat16 smem[];
  __nv_bfloat16 *Q_smem = smem;                 // [16, d]
  __nv_bfloat16 *K_smem = Q_smem + 16 * HEAD_DIM;
  __nv_bfloat16 *V_smem = K_smem + Bc * HEAD_DIM;
  uint32_t smem_Q_base = __cvta_generic_to_shared(Q_smem);
  uint32_t smem_K_base = __cvta_generic_to_shared(K_smem);
  uint32_t smem_V_base = __cvta_generic_to_shared(V_smem);

  // 载入 Q[16,d] -> smem (越界行填 0)
  for (int i = lane_id; i < 16 * HEAD_DIM; i += WARP_SIZE) {
    int row = i / HEAD_DIM, col = i % HEAD_DIM;
    Q_smem[i] = (row < group)
                    ? Q[q_base + (long)row * HEAD_DIM + col]
                    : __float2bfloat16(0.0f);
  }

  float lane_row_max_old[2] = {-FLT_MAX, -FLT_MAX};
  float lane_row_sum_old[2] = {0.0f, 0.0f};
  uint32_t R_Q[4], R_K[kWarpTileSeqLenK][2], R_V[kWarpTileHeadDimV][2];
  uint32_t R_S[kWarpTileSeqLenK][4], R_O[kWarpTileHeadDimV][4],
      R_D[kWarpTileHeadDimV][4];
#pragma unroll
  for (int j = 0; j < kWarpTileHeadDimV; ++j)
#pragma unroll
    for (int r = 0; r < 4; ++r)
      R_D[j][r] = 0u;
  __syncwarp();

  // 遍历段内 KV (Bc 步)
  for (int kv0 = kv_start; kv0 < kv_end; kv0 += Bc) {
    // 载入 K[Bc,d], V[Bc,d] -> smem (越界行填 0)
    for (int i = lane_id; i < Bc * HEAD_DIM; i += WARP_SIZE) {
      int row = i / HEAD_DIM, col = i % HEAD_DIM;
      int kr = kv0 + row;
      __nv_bfloat16 zero = __float2bfloat16(0.0f);
      K_smem[i] = (kr < kv_end) ? K[kv_base + (long)kr * HEAD_DIM + col] : zero;
      V_smem[i] = (kr < kv_end) ? V[kv_base + (long)kr * HEAD_DIM + col] : zero;
    }
    __syncwarp();

    // ---- S = Q @ K^T ----
#pragma unroll
    for (int j = 0; j < kWarpTileSeqLenK; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r)
        R_S[j][r] = 0u;
#pragma unroll
    for (int tile_K_d = 0; tile_K_d < (HEAD_DIM / kMmaAtomK); ++tile_K_d) {
      int q_row = lane_id % 16;
      int q_col = tile_K_d * kMmaAtomK + (lane_id / 16) * 8;
      uint32_t Q_ptr =
          smem_Q_base + (q_row * HEAD_DIM + q_col) * sizeof(__nv_bfloat16);
      LDMATRIX_X4(R_Q[0], R_Q[1], R_Q[2], R_Q[3], Q_ptr);
#pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j) {
        int k_row = j * 8 + lane_id % 8;
        int k_col = tile_K_d * kMmaAtomK + ((lane_id / 8) % 2) * 8;
        uint32_t K_ptr =
            smem_K_base + (k_row * HEAD_DIM + k_col) * sizeof(__nv_bfloat16);
        LDMATRIX_X2(R_K[j][0], R_K[j][1], K_ptr);
      }
#pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j)
        BF16_MMA_16816_F32(R_S[j][0], R_S[j][1], R_S[j][2], R_S[j][3], R_Q[0],
                           R_Q[1], R_Q[2], R_Q[3], R_K[j][0], R_K[j][1],
                           R_S[j][0], R_S[j][1], R_S[j][2], R_S[j][3]);
    }
    __syncwarp();

    // ---- online softmax (per row, warp 内 4-lane 组规约) ----
    float row_max_new[2] = {-FLT_MAX, -FLT_MAX};
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
    __syncwarp();

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
        int v_row = tile_V_Bc * kMmaAtomK + lane_id % 16;
        int v_col = j * 8;
        uint32_t V_ptr =
            smem_V_base + (v_row * HEAD_DIM + v_col) * sizeof(__nv_bfloat16);
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
    __syncwarp();

    // ---- rescale O, 更新 l/m ----
    bool first = (kv0 == kv_start);
    float m_old_0 = first ? m_new_0 : lane_row_max_old[0];
    float m_old_1 = first ? m_new_1 : lane_row_max_old[1];
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
    __syncwarp();
  }

  // ---- 写 partial (不归一化, 留给阶段2): 每 lane 写 2 行 ----
  // C fragment: c0,c1 -> row r=lane/4, 列 j*8 + 2*(lane%4) [,+1]; c2,c3 -> row r+8。
  int r = lane_id / 4;
  int dcol = 2 * (lane_id % 4);
  // 行 r
  if (r < group) {
    int bh = b * Hq + q_head0 + r;
    long base = (long)(bh * num_splits + split) * HEAD_DIM;
#pragma unroll
    for (int j = 0; j < kWarpTileHeadDimV; ++j) {
      float *d = reinterpret_cast<float *>(&R_D[j][0]);
      O_partial[base + j * 8 + dcol] = d[0];
      O_partial[base + j * 8 + dcol + 1] = d[1];
    }
    if (lane_id % 4 == 0) {
      m_partial[bh * num_splits + split] = lane_row_max_old[0];
      l_partial[bh * num_splits + split] = lane_row_sum_old[0];
    }
  }
  // 行 r+8
  if (r + 8 < group) {
    int bh = b * Hq + q_head0 + r + 8;
    long base = (long)(bh * num_splits + split) * HEAD_DIM;
#pragma unroll
    for (int j = 0; j < kWarpTileHeadDimV; ++j) {
      float *d = reinterpret_cast<float *>(&R_D[j][0]);
      O_partial[base + j * 8 + dcol] = d[2];
      O_partial[base + j * 8 + dcol + 1] = d[3];
    }
    if (lane_id % 4 == 0) {
      m_partial[bh * num_splits + split] = lane_row_max_old[1];
      l_partial[bh * num_splits + split] = lane_row_sum_old[1];
    }
  }
}

// ---- 阶段 2: 沿 num_splits 规约 (与 A 相同) ----
template <int HEAD_DIM>
__global__ void flash_decode_mma_stage2_kernel(
    const float *__restrict__ O_partial, const float *__restrict__ m_partial,
    const float *__restrict__ l_partial, __nv_bfloat16 *__restrict__ O,
    int num_splits) {
  const int bh = blockIdx.x;
  const int tid = threadIdx.x;
  float global_m = -FLT_MAX;
  for (int s = 0; s < num_splits; ++s)
    global_m = fmaxf(global_m, m_partial[bh * num_splits + s]);
  float acc = 0.0f, l = 0.0f;
  for (int s = 0; s < num_splits; ++s) {
    float ms = m_partial[bh * num_splits + s];
    float ls = l_partial[bh * num_splits + s];
    float sc = __expf(ms - global_m);
    acc += sc * O_partial[(bh * num_splits + s) * HEAD_DIM + tid];
    l += sc * ls;
  }
  O[bh * HEAD_DIM + tid] = __float2bfloat16(acc / l);
}

void solve(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
           int B, int Hq, int Hkv, int N, int d) {
  const int group = Hq / Hkv;
  if (group > 16)
    throw std::runtime_error("splitkv_mma: group>16 未实现(需 M 维 tiling)");
  const float scale = 1.0f / sqrtf((float)d);
  int num_splits = (N + 255) / 256;
  if (num_splits < 1) num_splits = 1;
  if (num_splits > 64) num_splits = 64;
  const int seg = (N + num_splits - 1) / num_splits;

  auto fopt = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
  auto O_partial = torch::empty({(long)B * Hq * num_splits * d}, fopt);
  auto m_partial = torch::empty({(long)B * Hq * num_splits}, fopt);
  auto l_partial = torch::empty({(long)B * Hq * num_splits}, fopt);

  auto launch = [&](auto headdim) {
    constexpr int HEAD_DIM = decltype(headdim)::value;
    size_t smem = (16 * HEAD_DIM + 64 * HEAD_DIM + 64 * HEAD_DIM) *
                  sizeof(__nv_bfloat16);
    dim3 grid1(B * Hkv, num_splits);
    flash_decode_mma_stage1_kernel<HEAD_DIM><<<grid1, 32, smem>>>(
        reinterpret_cast<const __nv_bfloat16 *>(Q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(K.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(V.data_ptr<at::BFloat16>()),
        O_partial.data_ptr<float>(), m_partial.data_ptr<float>(),
        l_partial.data_ptr<float>(), B, Hq, Hkv, N, group, num_splits, seg,
        scale);
    flash_decode_mma_stage2_kernel<HEAD_DIM><<<dim3(B * Hq), HEAD_DIM>>>(
        O_partial.data_ptr<float>(), m_partial.data_ptr<float>(),
        l_partial.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16 *>(O.data_ptr<at::BFloat16>()),
        num_splits);
  };

  if (d == 64)
    launch(std::integral_constant<int, 64>{});
  else if (d == 128)
    launch(std::integral_constant<int, 128>{});
  else
    throw std::runtime_error("splitkv_mma: only d=64/128 supported");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  namespace py = pybind11;
  m.def("solve", &solve,
        "Flash-Decoding (Tensor Core, group-as-M, split-KV, GQA, bf16)",
        py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("O"), py::arg("B"),
        py::arg("Hq"), py::arg("Hkv"), py::arg("N"), py::arg("d"));
}
