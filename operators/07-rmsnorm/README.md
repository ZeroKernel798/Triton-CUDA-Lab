# RMSNorm 算子优化

## cuda 实现
1. native.cu 占位版本：预留标准 RMSNorm kernel 接口（输入 `X`、权重 `gamma`、输出 `Y`，参数 `M/N/eps`），后续可基于 block 内归约实现每行 RMS 计算，并加入向量化读写与 warp-level reduce 优化。
2. rowblockvec_smem.cu：在 rowblockvec 的 4 个 bf16 向量化读写基础上，每个 block 负责一行，并把整行 `X` 缓存在 shared memory 中；归约阶段读一次全局 `X`，输出阶段从 shared memory 复用 `X`，减少一次全局内存读取。
3. rowblockvec_smem_gridstride.cu：在 rowblockvec_smem 基础上使用 grid-stride row loop。launcher 通过 `cudaGetDeviceProperties` 获取 SM 数，并将实际 block 数限制为 `SM * GRID_STRIDE_SM_MULT`（不超过 `M`）；`GRID_STRIDE_SM_MULT` 是编译宏，当前在 `test_cfg.py` 中 sweep `2/4/6/8` 倍，适合 `M` 很大时减少过量 block 调度开销。

## triton 实现
1. native.py 占位版本：预留 Triton kernel 启动与参数接口，后续可按行分块完成 `mean(x^2)` 归约、`rsqrt` 归一化和逐元素缩放，支持通过 `BLOCK_SIZE`、`num_warps`、`num_stages` 做 autotune。
