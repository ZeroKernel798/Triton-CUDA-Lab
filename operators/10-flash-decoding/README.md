# Flash-Decoding（bf16，含 GQA）

针对 **decode 阶段**(自回归推理，query 长度=1)的注意力。形状：
`Q[B,Hq,d]`、`K[B,Hkv,N,d]`、`V[B,Hkv,N,d]`、`O[B,Hq,d]`，bf16 输入 / fp32 累加。

## 为什么需要 Flash-Decoding
decode 时每个 head 只有 1 个 query token，`Q@Kᵀ`、`P@V` 退化成 **gemv**（向量×矩阵）。
普通 flash attention 按 Q 分块并行，decode 只有 `B*Hq` 个 query → 并行度严重不足，而 KV cache 很长。
**核心优化 = 沿 KV 维 split（split-KV）**：多个 block 各算一段 KV 的局部 `(m,l,O)`，再两阶段规约合并
（与 04/08 的 split-K 两阶段同思路）。

## GQA
`group = Hq/Hkv`，`kv_head = q_head / group`（MHA: `Hkv=Hq`；MQA: `Hkv=1`）。在 kernel 里只是 KV head 的索引映射，MHA/MQA 都是特例，**内置在同一 kernel**，不单开算子。

## 两条路线
- **A（CUDA Core + split-KV，gemv 形态）**：通用，group 任意。`splitkv.cu` / `splitkv.py`。
- **B（Tensor Core，group-as-M）**：GQA 把共享同一 KV head 的 `group` 个 query 沿 M 维**堆叠**，`Q[group,d]@Kᵀ` 恢复成 gemm → 用 bf16 mma。`group` 越大越划算（mma M=16，group<16 会 pad）。`splitkv_mma.cu` / `splitkv_mma.py`。

## cuda
- `cuda/splitkv.cu`（A，已实现）：两阶段。阶段1 `grid(B*Hq, num_splits)`，block=d 线程，每个 (head,split) 沿一段 KV 串行 online softmax，per-key 用 block-reduce 算 `q·k`；写 `O_partial/m_partial/l_partial`。阶段2 `grid(B*Hq)` 沿 splits 规约（rescale + 合并）。`num_splits` 按 N 自动选（每段 ~256 key，上限 64）。d ∈ {64,128}。
- `cuda/splitkv_mma.cu`（B，已实现）：group-as-M，单 warp(block=32) 处理 16 行 query，bf16 mma m16n8k16（基于 09 split-Q 移植）+ 段内 KV 循环，写 partial；阶段2 同 A 规约。约束 `group<=16`（更大的 MQA 需 M 维 tiling）。

## triton
- `triton/splitkv.py`（A，已实现）：同两阶段，`tl.sum(k*q, axis=1)` 做 `q·k`（gemv），online softmax 累加 `acc[d]`，阶段2 规约。
- `triton/splitkv_mma.py`（B，已实现）：group-as-M，`grid(B*Hkv, num_splits)` 把 group 个 query 堆成 `[BLOCK_M,d]`，`tl.dot` 自动走 TC；阶段2 同 A 规约。

## 验证
- 设备 RTX 4090 / sm_89，bf16。基准用 fp32 SDPA（KV 按 group `repeat_interleave` 到 Hq），容差 `2e-2`。
- 先验 A（triton → cuda），跑通后再做 B。cuda 第一版可能需在 4090 上按报错迭代。
