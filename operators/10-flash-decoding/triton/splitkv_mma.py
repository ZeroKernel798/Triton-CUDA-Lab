import torch
import triton
import triton.language as tl


# Flash-Decoding (B 版): GQA group-as-M + Tensor Core。
# 把共享同一 KV head 的 group 个 query 沿 M 维堆成 [BLOCK_M, d]，对同一段 KV 做 attention，
# Q@K^T / P@V 恢复成 gemm，tl.dot 自动走 Tensor Core (bf16, fp32 累加)。
# 阶段1: grid(B*Hkv, num_splits)，每个 program 处理一个 (b,kv_head) 的 group 个 query 对一段 KV。
# 阶段2: 与 A 相同，沿 num_splits 做 per-query 规约。
# group 越大越划算 (mma M=16，group<16 会 pad)。形状同 A。


@triton.jit
def flash_decode_mma_stage1(Q, K, V, Op, Mp, Lp, Hq, Hkv, N, group, num_splits,
                            seg, scale, D: tl.constexpr, BLOCK_M: tl.constexpr,
                            BLOCK_N: tl.constexpr):
    pid_bkv = tl.program_id(0)  # b*Hkv + hkv
    split = tl.program_id(1)
    b = pid_bkv // Hkv
    hkv = pid_bkv % Hkv
    q_head0 = hkv * group  # 该 kv_head 对应的首个 q_head

    offs_m = tl.arange(0, BLOCK_M)  # group 内 query 索引
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m < group

    # 堆叠 group 个 query: Q 行 = b*Hq + q_head0 + offs_m
    q_rows = b * Hq + q_head0 + offs_m
    q = tl.load(Q + q_rows[:, None] * D + offs_d[None, :], mask=mask_m[:, None],
                other=0.0)  # [BLOCK_M, D] bf16

    kv_base = (b * Hkv + hkv) * N
    kv_start = split * seg
    kv_end = tl.minimum(kv_start + seg, N)

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    for n0 in range(kv_start, kv_end, BLOCK_N):
        cn = n0 + offs_n
        mask_n = cn < kv_end
        k = tl.load(K + (kv_base + cn)[:, None] * D + offs_d[None, :],
                    mask=mask_n[:, None], other=0.0)  # [BLOCK_N, D]
        v = tl.load(V + (kv_base + cn)[:, None] * D + offs_d[None, :],
                    mask=mask_n[:, None], other=0.0)

        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale  # [BLOCK_M, BLOCK_N]
        s = tl.where(mask_n[None, :], s, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])                  # [BLOCK_M, BLOCK_N]
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v, out_dtype=tl.float32)
        m_i = m_new

    bh = b * Hq + q_head0 + offs_m
    tl.store(Op + (bh * num_splits + split)[:, None] * D + offs_d[None, :], acc,
             mask=mask_m[:, None])
    tl.store(Mp + bh * num_splits + split, m_i, mask=mask_m)
    tl.store(Lp + bh * num_splits + split, l_i, mask=mask_m)


@triton.jit
def flash_decode_mma_stage2(Op, Mp, Lp, O, num_splits, D: tl.constexpr):
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
        sc = tl.exp(ms - global_m)
        acc += sc * tl.load(Op + (bh * num_splits + s) * D + offs_d)
        l += sc * ls

    tl.store(O + bh * D + offs_d, (acc / l).to(O.dtype.element_ty))


def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, O: torch.Tensor,
          B: int, Hq: int, Hkv: int, N: int, d: int, **kwargs):
    Q = Q.to(torch.bfloat16).contiguous()
    K = K.to(torch.bfloat16).contiguous()
    V = V.to(torch.bfloat16).contiguous()

    group = Hq // Hkv
    scale = 1.0 / (d ** 0.5)
    num_splits = min(64, max(1, (N + 255) // 256))
    seg = (N + num_splits - 1) // num_splits
    BLOCK_M = max(16, triton.next_power_of_2(group))  # mma M >= 16
    BLOCK_N = 64

    Op = torch.empty((B * Hq * num_splits, d), device=Q.device, dtype=torch.float32)
    Mp = torch.empty((B * Hq * num_splits,), device=Q.device, dtype=torch.float32)
    Lp = torch.empty((B * Hq * num_splits,), device=Q.device, dtype=torch.float32)

    flash_decode_mma_stage1[(B * Hkv, num_splits)](
        Q, K, V, Op, Mp, Lp, Hq, Hkv, N, group, num_splits, seg, scale,
        D=d, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4,
    )
    flash_decode_mma_stage2[(B * Hq,)](
        Op, Mp, Lp, O, num_splits, D=d, num_warps=4,
    )
