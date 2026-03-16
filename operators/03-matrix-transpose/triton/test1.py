import torch
import triton
import triton.language as tl


@triton.jit
def matrix_transpose_kernel(
    input_ptr, output_ptr, 
    rows, cols, 
    stride_i_row, stride_i_col, 
    stride_o_row, stride_o_col, 
    BLOCK_ROW: tl.constexpr, 
    BLOCK_COL: tl.constexpr
):
    # 获取块的索引 针对输入
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    # 生成具体块对应的向量
    rm = pid_row * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    rn = pid_col * BLOCK_COL + tl.arange(0, BLOCK_COL)

    # 构建二位矩阵
    in_ptrs = input_ptr + rm[:, None] * stride_i_row + rn[None, :] * stride_i_col
    
    # mask处理
    mask = (rm[:, None] < rows) & (rn[None, :] < cols)

    # 数据加载过程
    tile = tl.load(in_ptrs, mask=mask)

    # 调用矩阵转置
    transposed_tile = tl.trans(tile)

    # 输出数据
    out_ptrs = output_ptr + rn[:, None] * stride_o_row + rm[None, :] * stride_o_col
    out_mask = tl.trans(mask)
    tl.store(out_ptrs, transposed_tile, mask=out_mask)


def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int, **kwargs):
    BLOCK_ROW = kwargs.get("BLOCK_ROW", 32)
    BLOCK_COL = kwargs.get("BLOCK_COL", 32)
    num_warps = kwargs.get("num_warps", 4)

    # 计算 Grid
    grid = (triton.cdiv(rows, BLOCK_ROW), triton.cdiv(cols, BLOCK_COL))

    matrix_transpose_kernel[grid](
        input, output, 
        rows, cols,
        input.stride(0), input.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_ROW=BLOCK_ROW, BLOCK_COL=BLOCK_COL,
        num_warps=num_warps
    )
