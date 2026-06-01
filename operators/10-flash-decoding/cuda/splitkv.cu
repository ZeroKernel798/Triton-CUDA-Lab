#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cfloat>

// Flash-Decoding (A 版): CUDA Core + split-KV 两阶段, bf16 输入 / fp32 累加。
// decode 阶段 query 长度=1, Q@K^T / P@V 都是 gemv 形态, 不走 Tensor Core。
// 关键: 沿 KV 维 split 解决 decode 并行度不足 (只有 B*Hq 个 query, KV 很长)。
//   阶段1: grid(B*Hq, num_splits), 每个 (head, split) 算该 head 对一段 KV 的局部 (m,l,O)。
//   阶段2: grid(B*Hq), 沿 num_splits 规约 (rescale + 合并) 得最终 O。
// GQA: group = Hq/Hkv, kv_head = q_head / group (MHA 是 Hkv=Hq 的特例)。
// 形状: Q[B,Hq,d] K[B,Hkv,N,d] V[B,Hkv,N,d] O[B,Hq,d]。block = d 线程, d ∈ {64,128}。

template <int HEAD_DIM>
__device__ __forceinline__ float block_reduce_sum(float val, float *smem) {
  int tid = threadIdx.x;
  smem[tid] = val;
  __syncthreads();
#pragma unroll
  for (int s = HEAD_DIM / 2; s > 0; s >>= 1) {
    if (tid < s)
      smem[tid] += smem[tid + s];
    __syncthreads();
  }
  float r = smem[0];
  __syncthreads();
  return r;
}

// ---- 阶段 1: 每个 (bh, split) 算一段 KV 的局部 attention ----
template <int HEAD_DIM>
__global__ void flash_decode_stage1_kernel(
    const __nv_bfloat16 *__restrict__ Q, const __nv_bfloat16 *__restrict__ K,
    const __nv_bfloat16 *__restrict__ V, float *__restrict__ O_partial,
    float *__restrict__ m_partial, float *__restrict__ l_partial, int B, int Hq,
    int Hkv, int N, int num_splits, int seg, float scale) {
  const int bh = blockIdx.x;     // 0 .. B*Hq-1
  const int split = blockIdx.y;  // 0 .. num_splits-1
  const int tid = threadIdx.x;   // 0 .. HEAD_DIM-1 (负责输出维度 tid)

  const int b = bh / Hq;
  const int hq = bh % Hq;
  const int group = Hq / Hkv;
  const int hkv = hq / group;

  const int q_off = (b * Hq + hq) * HEAD_DIM;
  const long kv_base = (long)(b * Hkv + hkv) * N * HEAD_DIM;

  const int kv_start = split * seg;
  int kv_end = kv_start + seg;
  if (kv_end > N)
    kv_end = N;

  __shared__ float red[HEAD_DIM];

  float q_reg = __bfloat162float(Q[q_off + tid]);
  float m = -FLT_MAX, l = 0.0f, acc = 0.0f;

  for (int key = kv_start; key < kv_end; ++key) {
    float prod = q_reg * __bfloat162float(K[kv_base + (long)key * HEAD_DIM + tid]);
    float score = block_reduce_sum<HEAD_DIM>(prod, red) * scale;

    float m_new = fmaxf(m, score);
    float alpha = __expf(m - m_new);
    float p = __expf(score - m_new);
    l = l * alpha + p;
    acc = acc * alpha + p * __bfloat162float(V[kv_base + (long)key * HEAD_DIM + tid]);
    m = m_new;
  }

  const int po = (bh * num_splits + split) * HEAD_DIM + tid;
  O_partial[po] = acc;
  if (tid == 0) {
    m_partial[bh * num_splits + split] = m;
    l_partial[bh * num_splits + split] = l;
  }
}

// ---- 阶段 2: 沿 num_splits 规约 ----
template <int HEAD_DIM>
__global__ void flash_decode_stage2_kernel(
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
    float sc = __expf(ms - global_m); // ms=-FLT_MAX(空段) -> 0
    acc += sc * O_partial[(bh * num_splits + s) * HEAD_DIM + tid];
    l += sc * ls;
  }
  O[bh * HEAD_DIM + tid] = __float2bfloat16(acc / l);
}

void solve(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
           int B, int Hq, int Hkv, int N, int d) {
  const float scale = 1.0f / sqrtf((float)d);

  // 选 split 数: 让每段约 256 个 key, 上限 64, 至少 1。
  int num_splits = (N + 255) / 256;
  if (num_splits < 1)
    num_splits = 1;
  if (num_splits > 64)
    num_splits = 64;
  const int seg = (N + num_splits - 1) / num_splits;

  auto fopt = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
  auto O_partial = torch::empty({(long)B * Hq * num_splits * d}, fopt);
  auto m_partial = torch::empty({(long)B * Hq * num_splits}, fopt);
  auto l_partial = torch::empty({(long)B * Hq * num_splits}, fopt);

  auto launch = [&](auto headdim) {
    constexpr int HEAD_DIM = decltype(headdim)::value;
    dim3 grid1(B * Hq, num_splits);
    dim3 grid2(B * Hq);
    dim3 block(HEAD_DIM);

    flash_decode_stage1_kernel<HEAD_DIM><<<grid1, block>>>(
        reinterpret_cast<const __nv_bfloat16 *>(Q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(K.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(V.data_ptr<at::BFloat16>()),
        O_partial.data_ptr<float>(), m_partial.data_ptr<float>(),
        l_partial.data_ptr<float>(), B, Hq, Hkv, N, num_splits, seg, scale);

    flash_decode_stage2_kernel<HEAD_DIM><<<grid2, block>>>(
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
    throw std::runtime_error("flash-decoding splitkv: only d=64/128 supported");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  namespace py = pybind11;
  m.def("solve", &solve, "Flash-Decoding (CUDA Core, split-KV, GQA, bf16)",
        py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("O"), py::arg("B"),
        py::arg("Hq"), py::arg("Hkv"), py::arg("N"), py::arg("d"));
}
