import torch
import triton
import triton.language as tl


@triton.jit
def softmax_attention(Q_ptr, K_ptr, V_ptr, OUT_ptr,
                      M, N, d,
                      Q_stride_M, Q_stride_d,
                      K_stride_N, K_stride_d,
                      V_stride_N, V_stride_d,
                      OUT_stride_M, OUT_stride_d,
                      BLOCKSIZE_M: tl.constexpr,
                      BLOCKSIZE_N: tl.constexpr,
                      BLOCKSIZE_d: tl.constexpr):
    
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    # 这个是Q矩阵M维度的偏移
    offset_M = pid0 * BLOCKSIZE_M + tl.arange(0, BLOCKSIZE_M)
    # 这是大家D维度的偏移
    offset_d = pid1 * BLOCKSIZE_d + tl.arange(0, BLOCKSIZE_d)
    # N维度需要滑动　所以不给初始值
    offset_N = tl.arange(0, BLOCKSIZE_N)

    # 当前块 只负责这个Q_offset 
    Q_offset = offset_M[:, None] * Q_stride_M + offset_d[None, :] * Q_stride_d
    Q_mask = (offset_M[:, None] < M) & (offset_d[None, :] < d)
    
    Q_data = tl.load(Q_ptr + Q_offset, mask=Q_mask)

    # 分块矩阵应该是blk_m * d   d * blk_n  blk_n * d最后得到的是blk_m * d
    accumulator = tl.zeros((BLOCKSIZE_M, BLOCKSIZE_d), dtype=tl.float32)
    # 这个应该是每一行分母指数和 每一行一个 恰好跟列的维度对应
    softmax_running_sum = tl.zeros([BLOCKSIZE_M], dtype=tl.float32)
    # 这个是每一行的局部最大值 会动态更新 每一行一个 恰好跟列维度对应
    softmax_current_max = tl.full([BLOCKSIZE_M], float("-inf"), dtype=tl.float32)
    # 这个是d那个玩意
    attention_logits_scale = tl.sqrt(d + 0.0)

    # 分块操作
    for current_index in range(0, N, BLOCKSIZE_N):
        current_k_offset = current_index + offset_N
        current_v_offset = current_k_offset

        K_offset = current_k_offset[:, None] * K_stride_N + offset_d[None, :] * K_stride_d
        V_offset = current_v_offset[:, None] * V_stride_N + offset_d[None, :] * V_stride_d

        K_mask = (current_k_offset[:, None] < N) & (offset_d[None, :] < d)
        V_mask = (current_v_offset[:, None] < N) & (offset_d[None, :] < d)

        K_data = tl.load(K_ptr + K_offset, mask=K_mask)
        V_data = tl.load(V_ptr + V_offset, mask=V_mask)

        attention_logits = tl.dot(Q_data, tl.trans(K_data)) / attention_logits_scale

        attention_logits_mask = (offset_M[:, None] < M) & (current_k_offset[None, :] < N)
        attention_logits = tl.where(attention_logits_mask, attention_logits, float("-inf"))

        # 找每一列的最大值
        current_block_max = tl.max(attention_logits, axis=-1)
        # 跟之前的最大值比较 然后更新
        max_value = tl.maximum(current_block_max, softmax_current_max)
        # 补差价
        alpha = tl.exp(softmax_current_max - max_value)
        softmax_current_max = max_value

        # 做安全softmax
        attention_logits_shift = attention_logits - max_value[:, None]

        # 当前这组值 
        softmax_nom = tl.exp(attention_logits_shift)
        # 当前的局部分母指数和
        softmax_denom = tl.sum(softmax_nom, axis=1)
        # 补差价再加上末尾的值
        softmax_running_sum = tl.fma(softmax_running_sum, alpha, softmax_denom)
        # 同上
        accumulator = tl.fma(accumulator, alpha[:, None], tl.dot(softmax_nom, V_data))

    accumulator /= softmax_running_sum[:, None]
    
    OUT_offset = offset_M[:, None] * OUT_stride_M + offset_d[None, :] * OUT_stride_d
    OUT_mask = (offset_M[:, None] < M) & (offset_d[None, :] < d)

    tl.store(OUT_ptr + OUT_offset, accumulator.to(tl.float16), mask=OUT_mask)


# Q, K, V, output are tensors on the GPU
def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, N: int, d: int):    
    BLOCKSIZE_M = 32
    BLOCKSIZE_d = 64
    BLOCKSIZE_N = 64
    
    grid = (triton.cdiv(M, BLOCKSIZE_M), triton.cdiv(d, BLOCKSIZE_d))
    
    Q_stride_M, Q_stride_d = Q.stride()
    K_stride_N, K_stride_d = K.stride()
    V_stride_N, V_stride_d = V.stride()
    OUT_stride_M, OUT_stride_d = output.stride()
    
    softmax_attention[grid](
        Q_ptr=Q,
        K_ptr=K, 
        V_ptr=V,
        OUT_ptr=output,
        
        M=M,
        N=N,
        d=d,
        
        Q_stride_M=Q_stride_M,
        Q_stride_d=Q_stride_d,
        K_stride_N=K_stride_N,
        K_stride_d=K_stride_d,
        V_stride_N=V_stride_N,
        V_stride_d=V_stride_d,

        OUT_stride_M=OUT_stride_M,
        OUT_stride_d=OUT_stride_d,
        
        BLOCKSIZE_M=BLOCKSIZE_M,
        BLOCKSIZE_d=BLOCKSIZE_d,
        BLOCKSIZE_N=BLOCKSIZE_N,
        num_warps=4
    )