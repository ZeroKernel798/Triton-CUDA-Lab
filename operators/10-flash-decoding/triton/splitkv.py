import torch
import triton
import triton.language as tl


# Flash-Decoding (A 版): split-KV 两阶段, gemv 形态 (decode seqlen_q=1)。
# 阶段1: grid(B*Hq, num_splits), 每个 program 算该 head 对一段 KV 的局部 (m,l,acc[d])。
# 阶段2: grid(B*Hq), 沿 num_splits 规约。
# GQA: kv_head = q_head // (Hq/Hkv)。形状 Q[B,Hq,d] K[B,Hkv,N,d] V[B,Hkv,N,d] O[B,Hq,d]。
# QK^T / PV 是向量×矩阵, 用 tl.sum 做 (不走 tl.dot/TC)。


@triton.jit
def flash_decode_stage1(Q, K, V, Op, Mp, Lp, Hq, Hkv, N, num_splits, seg, scale,
                        D: tl.constexpr, BLOCK_N: tl.constexpr):
    bh = tl.program_id(0)
    split = tl.program_id(1)
    b = bh // Hq
    hq = bh % Hq
    hkv = hq // (Hq // Hkv)

    offs_d = tl.arange(0, D)
    q = tl.load(Q + (b * Hq + hq) * D + offs_d).to(tl.float32)  # [D]

    kv_start = split * seg
    kv_end = tl.minimum(kv_start + seg, N)
    kv_base = (b * Hkv + hkv) * N  # 行起点(以 key 为单位), 列再 *D

    m = -float("inf")
    l = 0.0
    acc = tl.zeros([D], dtype=tl.float32)

    for n0 in range(kv_start, kv_end, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < kv_end
        kptr = K + (kv_base + offs_n)[:, None] * D + offs_d[None, :]
        vptr = V + (kv_base + offs_n)[:, None] * D + offs_d[None, :]
        k = tl.load(kptr, mask=mask_n[:, None], other=0.0).to(tl.float32)  # [BLOCK_N, D]
        v = tl.load(vptr, mask=mask_n[:, None], other=0.0).to(tl.float32)

        s = tl.sum(k * q[None, :], axis=1) * scale          # [BLOCK_N] = q·k
        s = tl.where(mask_n, s, -float("inf"))

        m_new = tl.maximum(m, tl.max(s, axis=0))
        alpha = tl.exp(m - m_new)
        p = tl.exp(s - m_new)                                # [BLOCK_N]
        l = l * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)   # [D]
        m = m_new

    tl.store(Op + (bh * num_splits + split) * D + offs_d, acc)
    tl.store(Mp + bh * num_splits + split, m)
    tl.store(Lp + bh * num_splits + split, l)


@triton.jit
def flash_decode_stage2(Op, Mp, Lp, O, num_splits, D: tl.constexpr):
    bh = tl.program_id(0)
    offs_d = tl.arange(0, D)

    global_m = -float("inf")
    for s in range(0, num_splits):
        global_m = tl.maximum(global_m, tl.load(Mp + bh * num_splits + s))

    acc = tl.zeros([D], dtype=tl.float32)
    l = 0.0
    for s in range(0, num_splits):
        ms = tl.load(Mp + bh * num_splits + s)
        ls = tl.load(Lp + bh * num_splits + s)
        sc = tl.exp(ms - global_m)  # 空段 ms=-inf -> 0
        acc += sc * tl.load(Op + (bh * num_splits + s) * D + offs_d)
        l += sc * ls

    tl.store(O + bh * D + offs_d, (acc / l).to(O.dtype.element_ty))


def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, O: torch.Tensor,
          B: int, Hq: int, Hkv: int, N: int, d: int, **kwargs):
    Q = Q.to(torch.bfloat16).contiguous()
    K = K.to(torch.bfloat16).contiguous()
    V = V.to(torch.bfloat16).contiguous()

    scale = 1.0 / (d ** 0.5)
    num_splits = min(64, max(1, (N + 255) // 256))
    seg = (N + num_splits - 1) // num_splits
    BLOCK_N = 64

    Op = torch.empty((B * Hq * num_splits, d), device=Q.device, dtype=torch.float32)
    Mp = torch.empty((B * Hq * num_splits,), device=Q.device, dtype=torch.float32)
    Lp = torch.empty((B * Hq * num_splits,), device=Q.device, dtype=torch.float32)

    flash_decode_stage1[(B * Hq, num_splits)](
        Q, K, V, Op, Mp, Lp, Hq, Hkv, N, num_splits, seg, scale,
        D=d, BLOCK_N=BLOCK_N, num_warps=4,
    )
    flash_decode_stage2[(B * Hq,)](
        Op, Mp, Lp, O, num_splits, D=d, num_warps=4,
    )
