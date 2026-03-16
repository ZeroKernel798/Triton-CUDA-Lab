import torch
import triton
import triton.language as tl

@triton.jit
def kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # 获取程序 ID (相当于 CUDA 的 blockIdx)
    pid = tl.program_id(axis=0)
    
    # 算出当前块处理的范围
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 边界掩码
    mask = offsets < n_elements
    
    # 加载数据
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    
    # 计算
    output = a + b
    
    # 写回
    tl.store(c_ptr + offsets, output, mask=mask)


def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int, **kwargs):
    # 设置核函数的参数 通过外部传入以便后续调优
    BLOCK_SIZE = kwargs.get("BLOCK_SIZE", 1024)
    num_warps = kwargs.get("num_warps", 4)
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    kernel[grid](A, B, C, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)


