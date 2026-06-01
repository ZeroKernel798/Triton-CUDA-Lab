# Reduction（求和规约）优化

计算一维数组的全局和 `output = sum(input)`，fp32。规约是 memory-bound 算子，优化核心是
**多级规约**（warp → block → grid）以减少同步与原子争用，并用 grid-stride 让每个线程承担多份
数据、跑满 SM。

## cuda 实现

1. `testv1.cu` -> grid-stride + warp shuffle + 原子合并
   四级规约：
   - **Grid-stride loop**：每个线程沿 `stride = gridDim.x * blockDim.x` 累加多个元素，兼容非
     2 的幂次与超大规模输入；
   - **Warp 级规约**：`__shfl_down_sync` 蝶形归约，无需 shared memory，极快；
   - **Block 级规约**：各 warp 的 leader 写入 shared，再由第一个 warp 做最后一轮 shuffle 规约；
   - **Global 级合并**：每个 block 只发一次 `atomicAdd` 写回，最小化原子争用。

   block 数按 `min(cdiv(N, BLOCK_SIZE), SM_count * 8)` 动态选取以平衡负载、掩盖访存延迟；
   `BLOCK_SIZE` 由框架编译期注入宏，shared 大小据此静态分配。

## triton 实现

1. `testv1.py` -> grid-stride + 块内规约 + 原子合并
   与 cuda 版同思路：每个 program 用 grid-stride loop 累加多个 `BLOCK_SIZE` 段，`tl.sum` 做块内
   规约，最后 `tl.atomic_add` 把局部和累加到全局（program 数上限 1024）。调优旋钮
   `BLOCK_SIZE`、`num_warps`、`num_stages`。
