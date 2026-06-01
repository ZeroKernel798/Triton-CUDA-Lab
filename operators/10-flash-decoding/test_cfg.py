import torch
import torch.nn.functional as F
from typing import Any, Dict, List

from core.spec import BaseOperatorSpec


class FlashDecodingSpec(BaseOperatorSpec):
    def __init__(self):
        super().__init__()
        # 1. 基础元数据与精度 (decode, bf16)
        self.name = "Flash-Decoding (bf16, GQA)"
        self.output_name = "O"
        self.atol = 2e-2
        self.rtol = 2e-2

        # 2. 测试规模: decode 阶段 seqlen_q=1。
        #    形状 Q[B,Hq,d] K[B,Hkv,N,d] V[B,Hkv,N,d] O[B,Hq,d]。
        #    GQA: group=Hq/Hkv (MHA: Hkv=Hq; MQA: Hkv=1)。d ∈ {64,128}。
        self.x_vals = [
            {"B": 1, "Hq": 32, "Hkv": 8, "N": 1024, "d": 128},   # GQA group4
            {"B": 4, "Hq": 32, "Hkv": 8, "N": 2048, "d": 128},
            {"B": 8, "Hq": 32, "Hkv": 8, "N": 4096, "d": 128},
            {"B": 16, "Hq": 16, "Hkv": 16, "N": 1024, "d": 64},  # MHA
        ]

        # 3. 配置 (cuda splitkv 无可调宏; splitkv_mma 待定; triton 同名)
        self.cuda_tuning_configs = self.make_configs([{}], ["splitkv", "splitkv_mma"])
        self.tuning_configs = self.make_configs([{}], ["splitkv", "splitkv_mma"])

    def get_macros(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    # 2. 物理指标
    def get_bytes_accessed(self, case: Dict[str, Any]) -> int:
        B, Hq, Hkv, N, d = case["B"], case["Hq"], case["Hkv"], case["N"], case["d"]
        # 读 Q(B*Hq*d) + K/V(B*Hkv*N*d each) + 写 O(B*Hq*d), bf16=2 bytes
        return (2 * B * Hq * d + 2 * B * Hkv * N * d) * 2

    def get_total_flops(self, case: Dict[str, Any]) -> int:
        B, Hq, N, d = case["B"], case["Hq"], case["N"], case["d"]
        return 4 * B * Hq * N * d  # QK^T (2) + PV (2)

    # 3. 验证: fp32 SDPA, KV 按 group repeat 到 Hq
    def reference_impl(self, Q, K, V, O, B, Hq, Hkv, N, d, **kwargs):
        group = Hq // Hkv
        q = Q.float().view(B, Hq, 1, d)
        k = K.float().view(B, Hkv, N, d).repeat_interleave(group, dim=1)
        v = V.float().view(B, Hkv, N, d).repeat_interleave(group, dim=1)
        res = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                             dropout_p=0.0, is_causal=False)
        O.copy_(res.view(B, Hq, d).to(O.dtype))

    def _rand(self, B, Hq, Hkv, N, d):
        dt = torch.bfloat16
        return {
            "Q": torch.empty((B, Hq, d), device="cuda", dtype=dt).uniform_(-0.1, 0.1),
            "K": torch.empty((B, Hkv, N, d), device="cuda", dtype=dt).uniform_(-0.1, 0.1),
            "V": torch.empty((B, Hkv, N, d), device="cuda", dtype=dt).uniform_(-0.1, 0.1),
            "O": torch.empty((B, Hq, d), device="cuda", dtype=dt),
            "B": B, "Hq": Hq, "Hkv": Hkv, "N": N, "d": d,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        return self._rand(1, 8, 2, 128, 64)  # GQA group4

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        return [
            self._rand(1, 8, 8, 128, 64),    # MHA
            self._rand(1, 8, 2, 128, 64),    # GQA group4
            self._rand(1, 8, 1, 256, 128),   # MQA
            self._rand(2, 16, 4, 512, 64),   # GQA group4, batch
        ]

    def generate_performance_test(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        return self._rand(cfg.get("B", 4), cfg.get("Hq", 32), cfg.get("Hkv", 8),
                          cfg.get("N", 2048), cfg.get("d", 128))
