import torch
import triton
import triton.language as tl

@triton.jit
def reduce(input: torch.Tensor, output: torch.Tensor, N: int, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = off < N

    ins = tl.load(input + off, mask, other=0.0)
    sum = tl.sum(ins, axis=0)
    tl.atomic_add(output, sum)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    bz =1024

    nums_blocks = triton.cdiv(N, bz)
    grid = (nums_blocks,)

    reduce[grid](input, output, N , bz)