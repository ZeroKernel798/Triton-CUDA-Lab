import torch
import triton
import triton.language as tl


# FlashAttention V1（bf16 + Tensor Core，与 cuda/flashattentionv1.cu 对称）
# 关键特征：KV 在外层（由 host 循环驱动，每个 KV 块 launch 一次 kernel）；Q 由 grid 并行；
# O / l / m 存在 HBM，每个 KV 块都「读出 -> 在线 softmax 更新 -> 写回」，体现 V1 的 HBM 往返。
# 每轮都做 1/l 归一化（V1 公式），对照 V2 的延迟归一化。
# bf16: Q/K/V 为 bf16，tl.dot 走 Tensor Core（fp32 累加）；O/l/m 的 round-trip buffer 用 fp32
#       保精度（bf16 反复读写会掉精度），最后由 host 转 bf16 写入真正的 O。
@triton.jit
def flash_attn_v1_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, M_ptr,
    M, N, d, j_start, scale,
    stride_qm, stride_km, stride_vm, stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)          # d 不分块：BLOCK_D >= d，整行点积才正确
    offs_n = j_start + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    q_mask = m_mask[:, None] & (offs_d[None, :] < d)
    Q = tl.load(Q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :], mask=q_mask, other=0.0)

    kv_mask = (offs_n[:, None] < N) & (offs_d[None, :] < d)
    Kj = tl.load(K_ptr + offs_n[:, None] * stride_km + offs_d[None, :], mask=kv_mask, other=0.0)
    Vj = tl.load(V_ptr + offs_n[:, None] * stride_vm + offs_d[None, :], mask=kv_mask, other=0.0)

    # V1 关键步骤：从 HBM 读回上一轮的 (m_i, l_i, O_i)
    m_i = tl.load(M_ptr + offs_m, mask=m_mask, other=-float("inf"))
    l_i = tl.load(L_ptr + offs_m, mask=m_mask, other=0.0)
    o_i = tl.load(O_ptr + offs_m[:, None] * stride_om + offs_d[None, :], mask=q_mask, other=0.0)

    # S_ij = Q_i K_j^T * scale，对越界列填 -inf（bf16 输入，fp32 累加，走 TC）
    S = tl.dot(Q, tl.trans(Kj), out_dtype=tl.float32) * scale
    s_mask = m_mask[:, None] & (offs_n[None, :] < N)
    S = tl.where(s_mask, S, -float("inf"))

    # 当前 tile 的局部 softmax 统计量
    m_tilde = tl.max(S, axis=1)
    P = tl.where(s_mask, tl.exp(S - m_tilde[:, None]), 0.0)
    l_tilde = tl.sum(P, axis=1)

    # 在线 softmax 合并新旧统计量
    m_new = tl.maximum(m_i, m_tilde)
    alpha = tl.exp(m_i - m_new)
    beta = tl.exp(m_tilde - m_new)
    l_new = alpha * l_i + beta * l_tilde
    inv_l = 1.0 / l_new

    # V1 归一化公式：O_i <- (l_i * alpha * O_i_prev + beta * (P @ V_j)) / l_new
    # P 转 bf16 喂第二个 dot（走 TC），fp32 累加；o_i 来自 fp32 round-trip buffer
    pv = tl.dot(P.to(tl.bfloat16), Vj, out_dtype=tl.float32)
    o_new = (l_i[:, None] * alpha[:, None] * o_i + beta[:, None] * pv) * inv_l[:, None]

    # 写回 HBM
    tl.store(O_ptr + offs_m[:, None] * stride_om + offs_d[None, :], o_new, mask=q_mask)
    tl.store(L_ptr + offs_m, l_new, mask=m_mask)
    tl.store(M_ptr + offs_m, m_new, mask=m_mask)


def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, O: torch.Tensor,
          M: int, N: int, d: int, **kwargs):
    BLOCK_M = 32
    BLOCK_N = 32
    # d 不分块：单块必须覆盖整个 d，且满足 tl.dot 对收缩维 >= 16 的要求
    BLOCK_D = 32 if d <= 32 else (64 if d <= 64 else 128)

    # Q/K/V 走 bf16（Tensor Core）
    Q = Q.to(torch.bfloat16)
    K = K.to(torch.bfloat16)
    V = V.to(torch.bfloat16)

    # V1 的 round-trip buffer 用 fp32 保精度：O_acc=0, l=0, m=-inf
    O_acc = torch.zeros((M, d), device=Q.device, dtype=torch.float32)
    L = torch.zeros((M,), device=Q.device, dtype=torch.float32)
    Max = torch.full((M,), -float("inf"), device=Q.device, dtype=torch.float32)

    scale = 1.0 / (d ** 0.5)
    grid = (triton.cdiv(M, BLOCK_M),)

    # KV 外层循环：每个 KV 块 launch 一次 kernel，跨 launch 的 HBM 读写由 stream 顺序保证一致。
    for j in range(0, N, BLOCK_N):
        flash_attn_v1_kernel[grid](
            Q, K, V, O_acc, L, Max,
            M, N, d, j, scale,
            Q.stride(0), K.stride(0), V.stride(0), O_acc.stride(0),
            BLOCK_M, BLOCK_N, BLOCK_D,
            num_warps=4,
        )

    # 最终把 fp32 累加结果转 bf16 写入真正的输出 O
    O.copy_(O_acc.to(O.dtype))
