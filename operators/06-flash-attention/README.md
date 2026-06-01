# flashattention优化（fp32 / CUDA Core）

本目录只保留 **fp32 的 CUDA Core 实现**，作为 FlashAttention v1 / v2 的算法代际参考。
bf16 + Tensor Core 的实现（CUDA mma + Triton）见 `09-flash-attention-tensorcore`。

## CUDA

- `cuda/flashattentionv1.cu`
  纯 V1 实现：grid 切 Q 行，kernel 内**外层 KV、内层 Q**；每轮 KV 都从 HBM 读出 `O_i / l_i / m_i`，按 V1 的归一化公式 `O ← (l_i*α*O + β*P·V) / l_new` 更新后再写回 HBM。HBM 流量大，但严格对应论文 Algorithm 1。

- `cuda/flashattentionv2.cu`
  纯 V2 实现：**Q 由 grid 并行**，kernel 内只剩 KV 内循环；`O / l / m` 全程驻寄存器，每轮只做 `O ← α*O + p*V` 的延迟更新，最后写回前一次性 `O / l`。HBM 流量从 V1 的 O(seq²) 降到 O(seq)。

两份都只用标准数学（`expf`、`*`、`/`），没有依赖特定架构的 intrinsic（除 warp 归约用的 `__shfl_xor_sync`），全程 fp32、纯 CUDA Core，不走 Tensor Core。

## 相关参考

- [ZeroKernel798/Learning-CUDA](https://github.com/ZeroKernel798/Learning-CUDA.git)：包含在国产平台上实现的 **FP32 FlashAttention v2** 相关内容，可作为本目录纯 fp32 / 非 Tensor Core 路线的对照参考。
