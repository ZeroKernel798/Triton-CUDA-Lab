import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # 使用 triton 实现向量加法核函数
    bid = tl.program_id(0)

    # 通过基准地址 + 处理长度来精准找到需要处理的向量数据
    block_start = bid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 获取数据 然后执行加法运算
    a_data = tl.load(a_ptr + offsets, mask=mask)
    b_data = tl.load(b_ptr + offsets, mask=mask)
    output = a_data + b_data

    # 将数据存回去
    tl.store(c_ptr + offsets, output, mask=mask)
    

def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int, **kwargs):
    # 通过动态参数来获取 kernel 调优参数
    BLOCK_SIZE = kwargs.get("BLOCK_SIZE", 1024)
    num_warps = kwargs.get("num_warps", 4)
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    # 启动核函数
    vector_add_kernel[grid](A, B, C, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)


