import torch
import triton
import triton.language as tl

@triton.jit
def kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    bid = tl.program_id(axis=0)
    block_start = bid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a_data = tl.load(a_ptr + offsets, mask=mask) 
    b_data = tl.load(b_ptr + offsets, mask=mask) 
    c_data = a_data + b_data
    tl.store(c_ptr + offsets, c_data, mask=mask)

# 增加 **kwargs 来接收来自框架的 BLOCK_SIZE, num_warps 等
def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int, **kwargs):
    BLOCK_SIZE = kwargs.get("BLOCK_SIZE", 1024)
    num_warps = kwargs.get("num_warps", 4)
    
    # 这里的 n_elements 是 N*N，逻辑保持不变
    n_elements = N * N
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # 调用 kernel 时，显式传递这些调优参数
    kernel[grid](
        a_ptr=A, 
        b_ptr=B, 
        c_ptr=C, 
        n_elements=n_elements, 
        BLOCK_SIZE=BLOCK_SIZE, # 传给 tl.constexpr
        num_warps=num_warps    # 传给 Triton 编译器
    )
