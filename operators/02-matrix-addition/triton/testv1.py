import torch
import triton
import triton.language as tl

@triton.jit
def kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    bid = tl.program_id(axis=0)
    block_start = bid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a_data = tl.load(a_ptr + offsets, mask=mask) # shape (BLOCK_SIZE)
    b_data = tl.load(b_ptr + offsets, mask=mask) # shape (BLOCK_SIZE)
    c_data = a_data + b_data
    tl.store(c_ptr + offsets, c_data, mask=mask)

# a, b, c are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    n_elements = N * N
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    # 内部调用 kernel 时，变量名跟着改就行
    kernel[grid](A, B, C, n_elements, BLOCK_SIZE)
