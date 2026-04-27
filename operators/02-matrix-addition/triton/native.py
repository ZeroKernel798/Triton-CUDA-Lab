import torch
import triton
import triton.language as tl

@triton.jit
def kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # 我们用一维视角去处理 简化处理逻辑
    bid = tl.program_id(0)
    block_start = bid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 数据掩码 防止访问奇怪的地址
    mask = offsets < n_elements
    
    # 下载数据和执行运算
    a_data = tl.load(a_ptr + offsets, mask=mask)
    b_data = tl.load(b_ptr + offsets, mask=mask)
    
    c_data = a_data + b_data
    
    # 数据存储
    tl.store(c_ptr + offsets, c_data, mask=mask)


def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int, **kwargs):
    # 获取核函数配置参数
    BLOCK_SIZE = kwargs.get("BLOCK_SIZE", 1024)
    num_warps = kwargs.get("num_warps", 4)
    
    # 转换成 1D 的视角逻辑 简化处理
    n_elements = N * N
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # 调用 kernel 时，显式传递这些调优参数
    kernel[grid](
        A, B, C,
        n_elements=n_elements, 
        BLOCK_SIZE=BLOCK_SIZE, 
        num_warps=num_warps  
    )
