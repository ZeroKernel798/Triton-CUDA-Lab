# 向量加法优化

计算 `C = A + B`，fp32。计算密度极低（每个元素 1 次加法、3 次访存），是典型的
**纯带宽瓶颈（memory-bound）**算子，优化核心就是把访存吃满。

## cuda 实现

1. `native.cu` -> 朴素实现
   一个线程处理一个元素，`C[tid] = A[tid] + B[tid]`，输入指针用 `__restrict__` 提示编译器。
   warp 内访问天然合并，但每线程只搬一个 float、load/store 指令多。`BLOCK_SIZE` 可调。

2. `float4.cu` -> float4 向量化
   一个线程处理一个 float4（128-bit 访存），把 load/store 指令数和内存事务数压到 1/4，更接近
   峰值带宽；尾部不足 4 个元素的部分退化为标量逐元素处理。

## triton 实现

1. `native.py` -> 基础 1D 分块
   一个 program 处理 `BLOCK_SIZE` 个元素，`tl.load`/`tl.store` 自动向量化与合并访存（等价手写
   float4），mask 处理边界。调优旋钮 `BLOCK_SIZE`、`num_warps`。
