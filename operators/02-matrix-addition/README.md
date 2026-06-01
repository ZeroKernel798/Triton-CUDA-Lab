# 矩阵加法优化

计算 `C = A + B`，`A/B/C` 为 `N x N` fp32 矩阵。与向量加法同为**纯带宽瓶颈**算子；由于逐元素
运算与内存布局无关，关键技巧是把 2D 矩阵摊平成 1D 连续访存再向量化。

## cuda 实现

1. `native.cu` -> 朴素 2D 实现
   2D grid + 2D block，一个线程处理一个 `(x, y)` 元素 `C[y*N+x] = A[y*N+x] + B[y*N+x]`。
   行优先下 warp 内访问连续、天然合并，但每线程只搬一个 float、指令开销大。`BLOCK_X/Y` 可调。

2. `flattened_float4.cu` -> 1D 摊平 + float4 向量化
   把矩阵当作 `N*N` 的一维数组处理，一个线程搬一个 float4（128-bit），load/store 指令与内存
   事务降到 1/4、更逼近峰值带宽；尾部不足 4 个元素退化为标量。

## triton 实现

1. `native.py` -> 1D 摊平分块
   同样以 `n_elements = N*N` 的一维视角处理，一个 program 负责 `BLOCK_SIZE` 个元素，
   `tl.load`/`tl.store` 自动向量化合并、mask 处理边界。调优旋钮 `BLOCK_SIZE`、`num_warps`。
