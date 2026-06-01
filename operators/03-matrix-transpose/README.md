# transpose 算子优化

## cuda 实现
1. native.cu 实现，有着访存合并的问题，读合并必然面临着写不合并，反之亦然，对于访存是瓶颈的算子来说，影响很大。

2. smem.cu 实现，通过引入 shared memory + padding，既解决了读写操作不能同时访存合并的问题，也解决了银行冲突导致的 shared memory 读写的问题。会有额外的 shared memory 损耗，且未做向量化读写，未能充分利用显存带宽。

3. smem_float4.cu 实现，在上一个版本的基础上，进一步引入了 float4 向量读写，在读取相同数据时，减少了线程数量，同时减轻了指令压力，提高了带宽利用率，但仍然有额外的 shared memory 损耗。但注意平衡单线程的寄存器使用，以及 SM 活跃度的问题。

4. smem_float4_swizzle.cu 实现，在上一个版本的基础上，将 shared memory 的 padding（`tile[32][33]`）替换为 XOR Swizzle（`tile[32][32]`）。Swizzle 以 float4 为粒度对物理列号进行重映射：`phys_f4_col = logical_f4_col XOR (row & 7)`，使同一 warp 中不同行的线程访问不同的 float4 列，从而将 Bank Conflict 从 32-way 降低至 4-way。相比 padding，Swizzle 消除了每行 1 个 float 的额外内存开销（32×32 vs 32×33），在 shared memory 压力较大时有机会多并发一个 block/SM，但实际收益需结合具体矩阵规模与 GPU 型号实测对比。

## triton 实现
1. native.py 实现，直接使用 `tl.load` + `tl.trans` + `tl.store` 完成转置，逻辑极简。Triton 编译器会自动为 shared memory 选择合适的布局以消除 bank conflict，读写合并也由编译器保证。核心调优参数是 `BLOCK_ROW` 和 `BLOCK_COL`；block 越大每次搬运的数据越多、指令开销摊薄，但寄存器压力和 shared memory 占用也随之上升，需在 block size 与 SM 占用率之间取平衡。

2. swizzle.py 实现，在 native.py 基础上引入 **block 调度 Swizzle** 以提升 L2 缓存利用率。native 使用 2D 网格，硬件按 row-major 调度；转置中 block `(pid_row, pid_col)` 写到输出矩阵的 `[pid_col*BC, pid_row*BR]`，row-major 下并发 SM 写到截然不同的输出行，导致 L2 写缓存频繁被驱逐。Swizzle 改为 **1D 网格**，将连续 `GROUP_SIZE` 个 `pid_row` 与同一 `pid_col` 组成一组优先执行，使并发 SM 集中写到同一段输出列条带，显著提升 L2 写命中率。额外引入调优参数 `GROUP_SIZE`（推荐 4~16），矩阵越大、SM 越多时收益越明显；`GROUP_SIZE` 过大会导致 SM 负载不均，需实测权衡。