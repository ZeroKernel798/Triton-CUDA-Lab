# RMSNorm 算子优化

计算 `Y = X / sqrt(mean(X^2) + eps) * gamma`，`X/Y` 为 `M x N` 的 bf16，`gamma` 为 `(N,)`，
平方和与缩放在 fp32 累加。按行做规约，是 memory-bound 算子，优化围绕**规约方式、向量化访存、
shared memory 复用、跨行调度**展开。

## cuda 实现（优化链）

1. `native.cu` -> warp-per-row
   一个 warp 负责一整行：lane 以步长 32 跨步读 `X` 累加平方和，`__shfl_xor_sync` 蝶形规约让全
   warp 拿到完整平方和，再跨步写回 `Y`。简单，但每行仅 32 线程、`X` 被读两次（规约 + 输出）。

2. `rowblock.cu` -> block-per-row + 两级块内规约
   一个 block 负责一行，用更多线程并行扫一行：先 warp 内 `__shfl_down_sync` 规约，各 warp 的
   partial sum 经 shared 由第一个 warp 做第二级规约，scale 经 shared 广播。提升单行并行度。

3. `rowblockvec.cu` -> + bf16 向量化读写
   在 rowblock 基础上，按 4 个 bf16 一组（`uint2` 打包）向量化读写 `X`/`Y`，减少 load/store 指令、
   增大内存事务、提升带宽利用（`N % 4 == 0` 时启用，否则回退标量）。

4. `rowblockvec_smem.cu` -> 整行缓存进 shared
   把整行 `X` 缓存到 shared memory：规约阶段读一次 global `X`，输出阶段从 shared 复用，省掉对
   `X` 的第二次 global 读取（约一半 `X` 读流量）。

5. `rowblockvec_smem_gridstride.cu` -> grid-stride 跨行 + 缓存 gamma
   block 数限制为 `min(M, SM * GRID_STRIDE_SM_MULT)`（`GRID_STRIDE_SM_MULT` 为编译宏，
   sweep `2/4/6/8`），每个 block 用 grid-stride row loop 处理多行，`M` 很大时减少过量 block 调度
   开销；进入 row loop 前把 `gamma` 缓存到 shared，避免多行重复读权重；该配置用 8 个 bf16
   向量化读写。

## triton 实现

1. `native.py` -> **占位（TODO）**
   已预留 kernel 与 `solve` 接口（`BLOCK_SIZE`/`num_warps`/`num_stages` 旋钮），kernel 体尚未实现
   （`mean(x^2)` 规约 + `rsqrt` 归一化 + 逐元素缩放待补）。
