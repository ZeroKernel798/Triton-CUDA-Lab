# FlashAttention（bf16 + Tensor Core）

本目录是 06-flash-attention 的 **bf16 + Tensor Core** 版本：CUDA 用 `mma.sync` PTX 手写，Triton 用 `tl.dot`（自动走 TC）。
fp32 / CUDA Core 的算法代际参考见 `06-flash-attention`。

- 数据类型：Q/K/V/O 为 bf16；矩阵乘累加全程 fp32（bf16 没有 bf16 累加的 mma）。
- 形状：单 batch/head，2D `Q[M,d] K[N,d] V[N,d] O[M,d]`。
- 基准：验证用 fp32 `F.scaled_dot_product_attention`（升精度避免 bf16 基准掩盖误差），容差 `atol=rtol=2e-2`。

## cuda（mma.sync，FA2 split-Q）
两版都是 **FlashAttention-2** 算法（Q 由 grid 并行、KV 内循环、O/l/m 驻寄存器、末尾归一化），区别在优化程度——
因为 Tensor Core 与 V1 的「O/l/m HBM 往返」相冲突，业界 TC 实现（LeetCUDA、flash-attention、cutlass）都只做 FA2。

- `cuda/flashattentionv1.cu`：**基础版**。`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`，`ldmatrix` 取 Q/K/V fragment，寄存器内 online softmax（warp shuffle 求 row max/sum），softmax 后把 P 转 bf16 压回 `R_S` 复用为 P@V 的 A。`kStage=1`（无 cp.async 双缓冲）、`kPad=0`（无 swizzle）。
- `cuda/flashattentionv2.cu`：**优化版**。在 v1 之上加 `kPad=8`（smem padding 错开 bank，降低 bank conflict）和 `kStage=2`（K tile 用 `cp.async` 双缓冲，计算当前 tile 时预取下一 tile K，隐藏 g2s 延迟）。

布局：MMA m16n8k16，Br=Bc=64，4 warps，d ∈ {64,128}。约束 `M%64==0`、`N%64==0`（tile 内不做尾部 mask）。
对照：LeetCUDA `kernels/flash-attn/mma/basic/flash_attn_mma_split_q.cu` 与 `flash_attn_mma_tiling_qk_F32F16F16F32.cu`（fp32 累加路径）。

## triton（tl.dot 自动走 Tensor Core）
- `triton/flashattentionv1.py`：真 V1，与 cuda 06 的 V1 算法对称——**KV 外层**（host 循环，每个 KV 块 launch 一次 kernel），Q 由 grid 并行，O/l/m 走 HBM 往返、每轮归一化。bf16 输入经 `tl.dot` 走 TC（fp32 累加）；O/l/m 的 round-trip buffer 用 fp32 保精度，最后转 bf16 写回。
- `triton/flashattentionv2.py`：V2，grid 只切 M，`exp2`（scale 预乘 log2(e)），延迟归一化，`tl.dot(P_bf16, V, acc)` 三操作数累加把矩阵乘与 acc 更新融合。

> Triton 的 `tl.dot` 在 bf16 输入下自动编译成 Tensor Core 的 mma，无需手写 PTX——这也是 06 的 triton 本就在用 TC、而把它归到 09 的原因。

## 验证
- 设备：RTX 4090 / sm_89（`compiler.py` 自动按设备探测 arch；bf16 mma 需 sm_80+）。
- 与 fp32 SDPA 基准在 `2e-2` 内通过；`ncu` 看 `sm__pipe_tensor_op_*` 确认走 Tensor Core；与 06 的 CUDA Core 版同尺寸对比吞吐。
- 注：cuda 两版是把已验证的 LeetCUDA mma kernel 移植到 bf16/2D 的**第一版**，可能需在 4090 上按编译/数值报错迭代。
