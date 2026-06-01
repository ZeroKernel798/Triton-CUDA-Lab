import torch
import triton
import triton.language as tl


@triton.jit
def native_gemv_kernel(
    x_ptr,
    A_ptr,
    y_ptr,
    K,
    N,
    stride_ak,
    stride_an,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # y = x @ A, x (K,), A (K, N), y (N,)
    # 一个 program 负责 BLOCK_N 个输出列，沿 K 以 BLOCK_K 为步长循环累加。
    # tl.load 自动向量化/coalesce（等价 CUDA 的 float4），num_stages 自动用 smem 做预取流水
    # （等价 CUDA 的 smem 缓存 x + double buffer），所以这里不需要手写这些底层优化。
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # x 的一段 [BLOCK_K]，被本 program 负责的 BLOCK_N 列复用。
        x_blk = tl.load(x_ptr + offs_k, mask=mask_k, other=0.0)
        # A 的一个 tile [BLOCK_K, BLOCK_N]。
        a_ptrs = A_ptr + offs_k[:, None] * stride_ak + offs_n[None, :] * stride_an
        a_mask = mask_k[:, None] & (offs_n[None, :] < N)
        a_blk = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # 沿 K 方向规约: sum_k x[k] * A[k, n]
        acc += tl.sum(a_blk * x_blk[:, None], axis=0)

    tl.store(y_ptr + offs_n, acc, mask=offs_n < N)


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
    num_warps = kwargs.get("num_warps", 4)
    num_stages = kwargs.get("num_stages", 2)

    grid = (triton.cdiv(N, BLOCK_N),)
    native_gemv_kernel[grid](
        x, A, y,
        K, N,
        N, 1,  # A 行优先: stride_ak = N, stride_an = 1
        BLOCK_N,
        BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
