import torch
import triton
import triton.language as tl


@triton.jit
def splitk_partial_kernel(
    x_ptr,
    A_ptr,
    partial_ptr,
    K,
    N,
    k_per_split,
    stride_ak,
    stride_an,
    stride_ps,
    stride_pn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 阶段 1: grid = (cdiv(N, BLOCK_N), SPLIT_K)。
    # 每个 program 算自己那段 K 的部分和，写到 partial[pid_s, :]，各 split 写各自行、无竞争。
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    k_start = pid_s * k_per_split
    k_end = tl.minimum(k_start + k_per_split, K)

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k0 in range(k_start, k_end, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < k_end

        x_blk = tl.load(x_ptr + offs_k, mask=mask_k, other=0.0)
        a_ptrs = A_ptr + offs_k[:, None] * stride_ak + offs_n[None, :] * stride_an
        a_mask = mask_k[:, None] & (offs_n[None, :] < N)
        a_blk = tl.load(a_ptrs, mask=a_mask, other=0.0)

        acc += tl.sum(a_blk * x_blk[:, None], axis=0)

    p_ptrs = partial_ptr + pid_s * stride_ps + offs_n * stride_pn
    tl.store(p_ptrs, acc, mask=offs_n < N)


@triton.jit
def splitk_reduce_kernel(
    partial_ptr,
    y_ptr,
    N,
    stride_ps,
    stride_pn,
    SPLIT_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # 阶段 2: 沿 SPLIT_K 维把 partial[0..SPLIT_K-1, :] 规约相加写回 y。
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for s in range(0, SPLIT_K):
        p = tl.load(partial_ptr + s * stride_ps + offs_n * stride_pn, mask=mask_n, other=0.0)
        acc += p

    tl.store(y_ptr + offs_n, acc, mask=mask_n)


def solve(
    x: torch.Tensor,
    A: torch.Tensor,
    y: torch.Tensor,
    K: int,
    N: int,
    **kwargs,
):
    BLOCK_N = kwargs.get("BLOCK_N", 128)
    BLOCK_K = kwargs.get("BLOCK_K", 64)
    SPLIT_K = kwargs.get("SPLIT_K", 4)
    num_warps = kwargs.get("num_warps", 4)
    num_stages = kwargs.get("num_stages", 2)

    # 临时 buffer 存各 split 的部分和，[SPLIT_K, N]，fp32 累加。
    partial = torch.empty((SPLIT_K, N), device=x.device, dtype=torch.float32)
    k_per_split = triton.cdiv(K, SPLIT_K)

    # 阶段 1: 各 split 段算部分和。
    grid_partial = (triton.cdiv(N, BLOCK_N), SPLIT_K)
    splitk_partial_kernel[grid_partial](
        x, A, partial,
        K, N, k_per_split,
        N, 1,  # A: stride_ak = N, stride_an = 1
        N, 1,  # partial: stride_ps = N, stride_pn = 1
        BLOCK_N,
        BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    # 阶段 2: 沿 SPLIT_K 规约 partial -> y。
    grid_reduce = (triton.cdiv(N, BLOCK_N),)
    splitk_reduce_kernel[grid_reduce](
        partial, y,
        N,
        N, 1,
        SPLIT_K,
        BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
