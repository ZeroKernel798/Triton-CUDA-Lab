# 矩阵乘法 TF32 Tensor Core 优化

手搓 TF32 Tensor Core SGEMM 的逐版本优化链，目标是逼近并反超 cuBLAS。与
`04-matrix-multiplication`（fp32 SIMT / CUDA core）分目录共存，本目录聚焦 `mma.m16n8k8`
(tf32) 路径。输入仍是 fp32 张量，Tensor Core 内部按 TF32（10-bit 尾数）计算，故与 fp32
基准对比时放宽容差（`atol=rtol=1e-2`）。

公共结构：block tiling 128x128，BK=16，256 线程；2x4 warp tiling，一个 warp 负责 64x32 的
C 块，对应 4x4 个 `m16n8k8`。约束 `M,N % 128 == 0`、`K % 16 == 0`。

## 版本演进（瓶颈 → 手段 → 指标）

`bt.cu` -> 基础版（B transposed）
A 用 `cp.async` 直达 smem，B 用 LDG 手动转置写入 smem，再 `ldmatrix` 装寄存器、`mma` 计算。
先把 Tensor Core 跑起来，暂不管 bank conflict / 访存合并，作为优化起点。

`bt_swizzle.cu` -> + XOR swizzle 消 bank conflict
对 As/Bs 的 K 维列索引按行索引的有效 bit 做 XOR 扰动，使 float4 的落点 {0,4,8,12} 错开
bank。As 的 ldmatrix 读取做到 0 冲突；Bs 读取无冲突、转置写入降到 ~8-way（受 16-wide +
float4 对齐的硬约束）。

`swizzle_bcf.cu` -> bank-conflict-free（逆推 cuBLAS）
Bs 不再转置（改 16x128 布局），用 `cp.async` 直接搬运，读取时放弃 ldmatrix、改为手动
shared load + 坐标映射拼 b fragment，从而 As/Bs 读写均 0 冲突。`SWIZZLE_B_F2` 只扰动 col
的 bit3~5，保证 float4 写入与逐元素读取自洽。

`swizzle_bcf_dbf.cu` -> 终版：+ grid swizzle + double buffer
- grid swizzle：把 block 的 global tile 访问轨迹折叠成宽 8 的竖条，相邻 block 复用同一批
  B tile，把 L2 命中率拉满（TC 算力强、访存时延凸显时收益明显）；
- double buffer：As/Bs 各 2 份 smem，`cp.async` 预取下一 K tile 与当前 tile 的
  ldmatrix/mma 计算重叠，隐藏 global->smem 延迟。

## 运行

```bash
# 编译本算子全部 CUDA 版本
python3 build.py --op 11 -j 8

# 功能校验 + 指定规模性能（vs torch fp32，开 TF32 口径一致）
python3 run.py --op 11 --mode tuning --metrics all

# 全尺度性能演进曲线
python3 run.py --op 11 --mode scaling --metrics all
```

## 备注

后三版（swizzle / bcf / bcf_dbf）的 swizzle 公式与双缓冲流水线依据原始优化笔记的**原理**
重建，已在远端 GPU 上做数值校验与性能对照；具体证据见 `.remote-logs/`。
