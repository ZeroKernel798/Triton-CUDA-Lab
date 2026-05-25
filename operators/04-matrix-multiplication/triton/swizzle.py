import torch
import triton
import triton.language as tl


@triton.jit
def swizzle_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    # 优化版本，采用 swizzling 技术来进行优化，提高 L2 Cache 的利用率
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(GROUP_SIZE_M, num_pid_m - first_pid_m)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # 可以进行编译器暗示，帮助优化指令
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    
    # 下面进行标准 tile matmul 过程
    # 计算 tile 对应的偏移向量
    offset_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offset_k = tl.arange(0, BLOCK_SIZE_K)
    
    # 计算块的基础地址矩阵
    a_ptr = a_ptr + offset_am[:, None] * stride_am + offset_k[None, :] * stride_ak
    b_ptr = b_ptr + offset_k[:, None] * stride_bk + offset_bn[None, :] * stride_bn
    
    # 累加器累加器
    sum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # 循环分块加载数据和计算
    for k in range(0, K, BLOCK_SIZE_K):
        # 计算当前要 load 的掩码
        a_mask = (offset_am[:, None] < M) & (offset_k[None, :] < K - k)
        b_mask = (offset_k[:, None] < K - k) & (offset_bn[None, :] < N)
        
        # load 数据
        a_data = tl.load(a_ptr, mask=a_mask, other=0.0)
        b_data = tl.load(b_ptr, mask=b_mask, other=0.0)
        
        # 做矩阵运算
        sum += tl.dot(a_data, b_data)
        
        # 指针继续往后偏移
        a_ptr += BLOCK_SIZE_K * stride_ak
        b_ptr += BLOCK_SIZE_K * stride_bk
        
    # 把数据写入到矩阵 c
    c_ptr = c_ptr + offset_am[:, None] * stride_cm + offset_bn[None, :] * stride_cn
    c_mask = (offset_am[:, None] < M) & (offset_bn[None, :] < N)
    tl.store(c_ptr, sum, mask=c_mask)
    

# host侧函数
def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int, M : int, K : int, **kwargs):
    BLOCK_SIZE_M = kwargs.get("BLOCK_SIZE_M", 32)
    BLOCK_SIZE_N = kwargs.get("BLOCK_SIZE_N", 32)
    BLOCK_SIZE_K = kwargs.get("BLOCK_SIZE_K", 8)
    GROUP_SIZE_M = kwargs.get("GROUP_SIZE_M", 4)
    num_warps = kwargs.get("num_warps", 4)
    num_stages = kwargs.get("num_stages", 2)
    
    # 计算线程分块参数
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    
    # 调用 kernel 时，显式传递这些调优参数
    swizzle_matmul_kernel[grid](
        A, B, C,
        M, N, K,
        K, 1,
        N, 1,
        N, 1,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE_M,
        num_warps=num_warps,
        num_stages=num_stages
    )
