# 矩阵乘法优化（fp32 / CUDA Core）

计算 `C = A @ B`，`A(M,K)`、`B(K,N)`、`C(M,N)`，fp32、纯 CUDA Core（不走 Tensor Core）。
这是一条从 naive 到逼近 cuBLAS 的逐瓶颈优化链；Tensor Core（TF32 mma）版本见
`11-matrix-multiplication-tensorcore`。

## cuda 实现（优化链）

1. `native.cu` -> 朴素实现
   一个线程算一个 C 元素，沿 K 直接累加 global 内存。无任何复用，访存量巨大，带宽瓶颈。
   16x16 线程块。

2. `smem.cu` -> 共享内存分块
   每个 block 把 A/B 的一个 tile 搬进 shared memory，在 BK 内循环复用，global 访存量降到
   约 1/BK，计算访存比上升。`BM=BN=BK=32`，仍是 1 输出/线程、标量内积。

3. `smem_vector.cu` -> float4 向量化 + 1x4 寄存器块
   引入 float4 向量化读写（128-bit 访存，减少 load/store 指令、增大内存事务），每线程持 4 个
   累加器、一次广播 `a_val` 乘 4 个 B 列。访存合并、带宽利用率提升。

4. `smem_outer8x8.cu` -> 8x8 寄存器外积
   每线程改算 8x8 的 C 寄存器块（`float sum[8][8]`），把内积重排为外积：一次 shared load 复用
   8 次，计算访存比大幅提升，kernel 由 memory-bound 转向 compute-bound。`BM=BN=128, BK=16`，
   256 线程。代价是寄存器压力上升。

5. `smem_outer8x8_at.cu` -> A 转置入 smem（at = A-transpose）
   把 A tile 转置写入 shared（`sA[BK][BM]`），使内层能用 float4 一次读 A 的 8 个值，替代之前
   8 次跨步标量读，降低 LSU 压力、减少流水线气泡。

6. `smem_outer8x8_at_swizzling.cu` -> XOR swizzle 消 bank conflict
   转置写/读在 shared 上引入了 bank conflict，用 XOR swizzle 重映射列号消除：
   `#define SWIZZLE_A(row, col) ((col) ^ ((row >> 2) << 3))`，A 的写入与 float4 读取都套用。

7. `swizzling_mem_coalesced.cu` -> C 写回访存合并
   把每线程 8 宽的 B/C 跨度拆成相距 64 列的两条 4 宽 strip（`tile_col_0 = tid%16*4`、
   `tile_col_1 = +64`），使一个 warp-row 的 16 个线程写回 global C 时落在连续的 64-float 段上，
   修复之前单条 8 宽 strip 留下的写回间隙、提升 C 写合并。

8. `double_buffer.cu` -> 双缓冲流水线
   shared 开两份（`sA[2][BK][BM]`、`sB[2][BK][BN]`），用 `read/write_stage ^= 1` ping-pong：
   计算当前 tile 的同时把下一 K-tile 预取到寄存器再写入另一 buffer，重叠计算与搬运、隐藏访存
   延迟，并省掉每轮一次 `__syncthreads`。代价是 smem 翻倍，可能压低 SM 活跃度。

9. `cpasync.cu` -> Ampere cp.async
   B 矩阵用 `__pipeline_memcpy_async`（`cp.async`）从 global 直达 shared、绕过寄存器/L1，越界
   走 zfill；`__pipeline_commit` / `__pipeline_wait_prior(0)` 做软件流水。A 仍走寄存器 staging。
   降低搬运过程中的寄存器/LSU 占用，进一步隐藏延迟。

## triton 实现

1. `tile.py` -> 基础 2D 分块
   grid `(cdiv(M,BM), cdiv(N,BN))`，沿 K 以 `BLOCK_SIZE_K` 累加，`tl.dot` 计算，mask 处理边界。
   用 `input_precision="ieee"` 强制 fp32 FMA（关闭 TF32），与 CUDA Core 口径一致。
   调优旋钮：`BLOCK_SIZE_M/N/K`、`num_warps`、`num_stages`。

2. `swizzle.py` -> block 调度 swizzle（L2 复用）
   在 tile 基础上把 grid 摊平成 1D，按 M-major 分组重映射 `pid -> (pid_m, pid_n)`
   （`GROUP_SIZE_M` 个连续行块一组），让并发调度的 block 共享同一段 A/B 行、提升 L2 命中率。
   额外旋钮 `GROUP_SIZE_M`（默认 4），矩阵越大收益越明显。
