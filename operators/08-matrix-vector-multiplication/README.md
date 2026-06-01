# GEMV 算子优化

计算 `y = x @ A`，左为向量 `x` `(1, K)`、右为矩阵 `A` `(K, N)`、输出 `y` `(1, N)`，数据类型为 fp32。
其中 `K` 是收缩维度（向量长度 / A 行数），`N` 是输出维度（A 列数）。

## cuda 实现
1. native.cu：朴素实现。一个线程负责一个输出列 `j`，沿 `K` 行串行累加 `x[i] * A[i*N + j]` 后写回 `y[j]`。由于同一 warp 内相邻线程访问 `A[i*N + j]` 是连续地址，全局读 `A` 天然 coalesced。已加 `const __restrict__` 与 `#pragma unroll 4`。`BLOCK_X` 控制每个 block 的线程数，在 `test_cfg.py` 中 sweep `32/64/128/256/512/1024`。
2. vector.cu：float4 向量化读写。一个线程负责 4 个相邻输出列，沿 `K` 方向一次 `float4` 读 `x[i..i+3]`、对应 4 行各 `float4` 读 `A`，最后 `float4` 写回 `y[col..col+3]`。warp 内仍连续访问，coalesced 不变，但 load/store 指令数减少、内存事务更大、MLP 更高。假设 `K`、`N` 均为 4 的倍数（对齐数据），验证用例已按此约束设置。
3. smem.cu：在 vector 基础上把 `x` 分块缓存进 shared memory。同一 block 的线程负责一组相邻列但共享同一条 `x`，每个 tile 的 `x` 只从 global 读一次进 smem，block 内复用，省掉 `x` 的重复读（vector 版本中 `x` 的读流量约为 `A` 的 1/4）。`TILE_K` 控制每个 tile 缓存的 `x` 长度，在 `test_cfg.py` 中 sweep `256/512/1024`。
4. splitk.cu：smem + 两阶段 split-K，针对 `N` 很小、`K` 很大时 vector 版并行度（`N/4`）不足、喂不满 GPU 的长条场景。
   - 阶段 1（`splitk_partial_kernel`）：沿 `K` 维切成 `SPLIT_K` 段，2D grid（`gridDim.y = SPLIT_K`），每个 split 段算自己那段的部分和并写到临时 buffer `partial[SPLIT_K, N]` 的对应行——各 split 写各自行、互不重叠，因此**不需要 atomicAdd**。block 内用 `TILE_K` 分块缓存本段 `x` 进 shared memory 复用。
   - 阶段 2（`splitk_reduce_kernel`）：沿 `SPLIT_K` 维把 `partial` 各行规约相加写回 `y`。
   - 相比单阶段 atomicAdd 版本，用一块临时显存和一次额外 kernel 换掉了跨 block 的 atomic 争用。`SPLIT_K` sweep `2/4/8/16`，`TILE_K` sweep `256/512`。

## triton 实现
Triton 以 program（block）为单位操作张量块，`tl.load`/`tl.store` 自动向量化与 coalesce（等价 CUDA 的 float4），`num_stages` 自动用 shared memory 做预取流水（等价 CUDA 的 smem 缓存 x + double buffer）。因此 CUDA 里手写的 float4 / 显式 smem 在 Triton 里塌缩成「一个基础 kernel + autotune」，只有 split-K 这种算法层策略需要单独写。

1. native.py：一个 program 负责 `BLOCK_N` 个输出列，沿 `K` 以 `BLOCK_K` 为步长循环，载入 `x` 段 `[BLOCK_K]` 和 `A` tile `[BLOCK_K, BLOCK_N]`，用 `tl.sum(A * x[:, None], axis=0)` 沿 `K` 规约累加。autotune 旋钮：`BLOCK_N`（≈ thread coarsening）、`BLOCK_K`、`num_warps`、`num_stages`（≈ smem 预取）。边界由 mask 处理，无需假设 `K/N` 对齐。不用 `tl.dot`：GEMV 的 `M=1` 维会被 pad 到 16，浪费 Tensor Core。
2. splitk.py：smem + 两阶段 split-K，与 cuda 版对称。阶段 1（`splitk_partial_kernel`）2D grid `(cdiv(N, BLOCK_N), SPLIT_K)`，每个 program 算自己那段 `K` 的部分和写到 `partial[SPLIT_K, N]` 的对应行（各 split 写各自行、无竞争，避免 atomic）；阶段 2（`splitk_reduce_kernel`）沿 `SPLIT_K` 规约 `partial` 写回 `y`。autotune 旋钮在 native 基础上加 `SPLIT_K`。
