import torch
import triton
import triton.language as tl

@triton.jit
def application(
    ninetoothed_ninetoothed_tensor_0_pointer, ninetoothed_ninetoothed_tensor_1_pointer, ninetoothed_ninetoothed_tensor_2_pointer,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Swizzling 坐标计算 
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    # 直接计算绝对索引，让 Mask 逻辑生效
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # 指针初始化
    a_ptrs = ninetoothed_ninetoothed_tensor_0_pointer + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = ninetoothed_ninetoothed_tensor_1_pointer + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # 累加计算
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # 正确的 Mask：当索引 >= M 或 >= K 时，Mask 为 False，加载 0.0
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        b_mask = (offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N)
        
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # accumulator = tl.dot(a_tile, b_tile, accumulator)
        # 关闭tf32
        accumulator = tl.dot(a_tile, b_tile, accumulator, allow_tf32=False)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # 写回结果
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = ninetoothed_ninetoothed_tensor_2_pointer + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, M: int, N: int, K: int, **kwargs):
    # 获取调优参数，如果没有则给定默认高性能参数
    BLOCK_SIZE_M = kwargs.get("BLOCK_SIZE_M", 128)
    BLOCK_SIZE_N = kwargs.get("BLOCK_SIZE_N", 256)
    BLOCK_SIZE_K = kwargs.get("BLOCK_SIZE_K", 64)
    GROUP_SIZE_M = kwargs.get("GROUP_SIZE_M", 8)
    num_warps = kwargs.get("num_warps", 8)
    num_stages = kwargs.get("num_stages", 3)

    # 由于使用了 Swizzling 逻辑，Grid 通常是一个 1D 的任务池
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), )

    application[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=num_warps,
        num_stages=num_stages
    )

