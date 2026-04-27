import torch
import triton
import triton.language as tl


@triton.jit
def matrix_transpose_swizzle_kernel(
    input_ptr, output_ptr,
    rows, cols,
    stride_i_row, stride_i_col,
    stride_o_row, stride_o_col,
    BLOCK_ROW: tl.constexpr,
    BLOCK_COL: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    # ---- Swizzle 调度原理 ----
    # native.py 用 2D 网格，硬件按 row-major 调度：(0,0)(0,1)(0,2)...
    # 转置中 block(pid_row, pid_col) 写到 output[pid_col*BC, pid_row*BR]
    # → row-major 下并发 SMs 写到截然不同的 output 行 → L2 写缓存被反复驱逐
    #
    # 这里改为 1D 网格 + GROUP_SIZE 重排：
    # 相邻 GROUP_SIZE 个 pid_row 搭配同一个 pid_col 先跑完，
    # 使并发 SMs 都写到同一段 output 列条带 → L2 写命中率大幅提升
    #
    # 映射公式（与 Triton matmul tutorial 的 swizzle 相同）:
    #   group_id       = pid // (GROUP_SIZE * num_pid_cols)
    #   first_pid_row  = group_id * GROUP_SIZE
    #   group_size_adj = min(num_pid_rows - first_pid_row, GROUP_SIZE)  # 末尾修正
    #   pid_row = first_pid_row + pid % group_size_adj
    #   pid_col = (pid % (GROUP_SIZE * num_pid_cols)) // group_size_adj

    pid = tl.program_id(0)
    num_pid_rows = tl.cdiv(rows, BLOCK_ROW)
    num_pid_cols = tl.cdiv(cols, BLOCK_COL)

    num_pid_in_group = GROUP_SIZE * num_pid_cols
    group_id        = pid // num_pid_in_group
    first_pid_row   = group_id * GROUP_SIZE
    group_size_adj  = tl.minimum(num_pid_rows - first_pid_row, GROUP_SIZE)
    pid_row = first_pid_row + (pid % group_size_adj)
    pid_col = (pid % num_pid_in_group) // group_size_adj

    # ---- 以下与 native.py 完全相同 ----
    rm = pid_row * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    rn = pid_col * BLOCK_COL + tl.arange(0, BLOCK_COL)

    in_ptrs = input_ptr + rm[:, None] * stride_i_row + rn[None, :] * stride_i_col
    mask    = (rm[:, None] < rows) & (rn[None, :] < cols)

    tile            = tl.load(in_ptrs, mask=mask)
    transposed_tile = tl.trans(tile)

    out_ptrs = output_ptr + rn[:, None] * stride_o_row + rm[None, :] * stride_o_col
    out_mask = tl.trans(mask)
    tl.store(out_ptrs, transposed_tile, mask=out_mask)


def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int, **kwargs):
    BLOCK_ROW  = kwargs.get("BLOCK_ROW",  32)
    BLOCK_COL  = kwargs.get("BLOCK_COL",  32)
    num_warps  = kwargs.get("num_warps",   4)
    GROUP_SIZE = kwargs.get("GROUP_SIZE",  8)

    num_pid_rows = triton.cdiv(rows, BLOCK_ROW)
    num_pid_cols = triton.cdiv(cols, BLOCK_COL)

    matrix_transpose_swizzle_kernel[num_pid_rows * num_pid_cols,](
        input, output,
        rows, cols,
        input.stride(0),  input.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_ROW=BLOCK_ROW,
        BLOCK_COL=BLOCK_COL,
        GROUP_SIZE=GROUP_SIZE,
        num_warps=num_warps,
    )
