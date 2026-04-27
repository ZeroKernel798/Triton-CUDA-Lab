# RMSNorm 算子优化

## cuda 实现
1. native.cu 占位版本：预留标准 RMSNorm kernel 接口（输入 `X`、权重 `gamma`、输出 `Y`，参数 `M/N/eps`），后续可基于 block 内归约实现每行 RMS 计算，并加入向量化读写与 warp-level reduce 优化。

## triton 实现
1. native.py 占位版本：预留 Triton kernel 启动与参数接口，后续可按行分块完成 `mean(x^2)` 归约、`rsqrt` 归一化和逐元素缩放，支持通过 `BLOCK_SIZE`、`num_warps`、`num_stages` 做 autotune。
