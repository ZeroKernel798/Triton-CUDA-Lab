import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_native_kernel(
    X_ptr,
    gamma_ptr,
    Y_ptr,
    M,
    N,
    eps,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    BLOCK_SIZE: tl.constexpr,
):
    # TODO: 实现 Triton RMSNorm kernel
    pass


def solve(
    X: torch.Tensor,
    gamma: torch.Tensor,
    Y: torch.Tensor,
    M: int,
    N: int,
    eps: float,
    **kwargs,
):
    BLOCK_SIZE = kwargs.get("BLOCK_SIZE", 256)
    num_warps = kwargs.get("num_warps", 4)
    num_stages = kwargs.get("num_stages", 2)

    # TODO: 设置 grid 并 launch _rmsnorm_kernel
    _ = (X, gamma, Y, M, N, eps, BLOCK_SIZE, num_warps, num_stages)
