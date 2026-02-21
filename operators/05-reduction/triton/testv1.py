import torch
import triton
import triton.language as tl

@triton.jit
def reduce_kernel(
    input_ptr, 
    output_ptr, 
    N, 
    BLOCK_SIZE: tl.constexpr
):
    # 1. 计算当前 Program 负责的起始位置
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(0)
    
    # 每个 Program 内部先进行累加，减少原子操作次数
    total_sum = 0.0
    
    # Grid-stride Loop：模仿 CUDA 的 for 循环
    # 这样每个 Program 可以处理多个 BLOCK_SIZE 的数据
    step = num_jobs * BLOCK_SIZE
    for i in range(pid * BLOCK_SIZE, N, step):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        ins = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        total_sum += tl.sum(ins, axis=0)

    # 2. 最后将本 Program 的局部和，原子累加到全局
    tl.atomic_add(output_ptr + 0, total_sum)

def solve(input: torch.Tensor, output: torch.Tensor, N: int, **kwargs):
    # 确保输出清零
    output.zero_()

    # 获取参数
    BLOCK_SIZE = kwargs.get("BLOCK_SIZE", 1024)
    num_warps = kwargs.get("num_warps", 4)
    num_stages = kwargs.get("num_stages", 2)

    # 这里的 grid 逻辑会自动使用上面获取到的 BLOCK_SIZE
    grid = lambda META: (min(triton.cdiv(N, META['BLOCK_SIZE']), 1024),)

    reduce_kernel[grid](
        input_ptr=input, 
        output_ptr=output, 
        N=N, 
        BLOCK_SIZE=BLOCK_SIZE,      # 传入 tl.constexpr 参数
        num_warps=num_warps,        # 传入 Triton 编译配置
        num_stages=num_stages       # 传入 Triton 编译配置
    )