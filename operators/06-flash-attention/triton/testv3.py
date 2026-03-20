import torch
import triton
import triton.language as tl


@triton.jit
def softmax_attention(Q_ptr, K_ptr, V_ptr, OUT_ptr,
                      M, N, d,
                      Q_stride_M,
                      K_stride_N,
                      V_stride_N,
                      OUT_stride_M,
                      BLOCKSIZE_M: tl.constexpr,
                      BLOCKSIZE_N: tl.constexpr,
                      BLOCKSIZE_d: tl.constexpr):
    
    # 先找出块的索引ID  d这个维度没有
    pid0 = tl.program_id(0)

    # 找出几个维度的索引偏移 还未对应到地址
    offset_M = pid0 * BLOCKSIZE_M + tl.arange(0, BLOCKSIZE_M)
    offset_d = tl.arange(0, BLOCKSIZE_d)
    offset_N = tl.arange(0, BLOCKSIZE_N)

    # 先找出当前Q对应的位置
    Q_offset = offset_M[:, None] * Q_stride_M + offset_d[None,:] * 1
    Q_mask = (offset_M[:, None] < M) & (offset_d[None,:] < d)
    Q_data = tl.load(Q_ptr + Q_offset, mask=Q_mask, other=0.0)

    # 先生成要用的局部值
    acc = tl.zeros((BLOCKSIZE_M, BLOCKSIZE_d), dtype=tl.float32)
    softmax_current_sum = tl.zeros([BLOCKSIZE_M], dtype=tl.float32)
    softmax_current_max = tl.full([BLOCKSIZE_M], float("-inf"), dtype=tl.float32)
    # 这个可以将e^x变成2^x 提升速度
    attention_scale = tl.math.rsqrt(d.to(tl.float32)) * 1.44269504

    # 接下来开始搞K和V
    for current_idx in range(0, N, BLOCKSIZE_N):
        current_offset_N = current_idx + offset_N

        K_offset = current_offset_N[:, None] * K_stride_N + offset_d[None,:] * 1
        V_offset = current_offset_N[:, None] * V_stride_N + offset_d[None,:] * 1

        K_mask = (current_offset_N[:, None] < N) & (offset_d[None,:] < d)
        V_mask = (current_offset_N[:, None] < N) & (offset_d[None,:] < d)

        K_data = tl.load(K_ptr + K_offset, mask=K_mask)
        V_data = tl.load(V_ptr + V_offset, mask=V_mask)

        # 这里是半精度操作 然后输出为fp32
        mat_data = tl.dot(Q_data, tl.trans(K_data), out_dtype=tl.float32)
        mat_data *= attention_scale
        mat_mask = (current_offset_N[None, :] < N) & (offset_M[:, None] < M)
        # 加上mask 不影响后面的求最大值
        mat_data = tl.where(mat_mask, mat_data, float("-inf"))

        # 先处理最大值逻辑
        current_block_max = tl.max(mat_data, axis=-1)
        max_value = tl.maximum(current_block_max, softmax_current_max)
        # 计算差值
        alpha = tl.exp2(softmax_current_max - max_value)
        softmax_current_max = max_value

        # 下面对当前组的数据做safe softmax
        mat_data_shift = mat_data - max_value[:,None]
        # 这个是当前这组数据的safe softmax
        softmax_data = tl.exp2(mat_data_shift)
        #  当前的局部分母指数和
        softmax_denom = tl.sum(softmax_data, axis=1)
        # 补差价再加上末尾的值
        softmax_current_sum = tl.fma(softmax_current_sum, alpha, softmax_denom)
        # 把之前的矩阵结果补差价 再加上末尾的值
        # acc = tl.fma(acc, alpha[:, None], tl.dot(softmax_data, V_data))

        # fp16提速
        softmax_data_fp16 = softmax_data.to(tl.float16)
        # 补齐之前的 acc 差价（这一步依然是 FP32 标量乘法）
        acc = acc * alpha[:, None]
        # 使用 tl.dot 的累加模式：acc = softmax_data_fp16 * V_data + acc
        # 这样不仅快，而且利用了现成的 acc 作为累加器
        acc = tl.dot(softmax_data_fp16, V_data, acc, out_dtype=tl.float32)
    
    # 这个地方要处理分母指数和的除法
    acc /= softmax_current_sum[:, None]

    OUT_offset = offset_M[:, None] * OUT_stride_M + offset_d[None,:] * 1
    OUT_mask = (offset_M[:, None] < M) & (offset_d[None,:] < d)

    tl.store(OUT_ptr + OUT_offset, acc.to(tl.float16), mask=OUT_mask)


# Q, K, V, output are tensors on the GPU
def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, O: torch.Tensor, M: int, N: int, d: int): 
    #  确保输入是半精度
    Q, K, V = Q.to(torch.float16), K.to(torch.float16), V.to(torch.float16)  
    
    # 设置block大小 
    BLOCKSIZE_M = 32
    BLOCKSIZE_N = 32
    BLOCKSIZE_d = 32 if d <= 32 else (64 if d <= 64 else 128)
    
    # 不应该对维度d分块 否则无法同步整行的最大值求出softmax
    grid = (triton.cdiv(M, BLOCKSIZE_M),)
    
    Q_stride_M, _ = Q.stride()
    K_stride_N, _ = K.stride()
    V_stride_N, _ = V.stride()
    OUT_stride_M, _ = O.stride()
    
    softmax_attention[grid](
        Q_ptr=Q,
        K_ptr=K, 
        V_ptr=V,
        OUT_ptr=O,
        
        M=M,
        N=N,
        d=d,
        
        Q_stride_M=Q_stride_M,
        K_stride_N=K_stride_N,
        V_stride_N=V_stride_N,

        OUT_stride_M=OUT_stride_M,
  
        BLOCKSIZE_M=BLOCKSIZE_M,
        BLOCKSIZE_N=BLOCKSIZE_N,
        BLOCKSIZE_d=BLOCKSIZE_d,
        num_warps=4
    )