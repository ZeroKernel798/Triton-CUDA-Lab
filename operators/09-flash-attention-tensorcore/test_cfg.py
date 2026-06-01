import torch
import torch.nn.functional as F
from typing import Any, Dict, List

from core.spec import BaseOperatorSpec


class FlashAttentionTCSpec(BaseOperatorSpec):
    def __init__(self):
        super().__init__()
        # 1. 基础元数据与精度（bf16 + Tensor Core）
        self.name = "Flash Attention (Tensor Core, bf16)"
        self.output_name = "O"
        # bf16 + TC 累加误差较大，容差放宽（不过再放宽）
        self.atol = 2e-2
        self.rtol = 2e-2

        # 2. 测试规模（TC 友好：M=N，且都是 64 的倍数；d ∈ {64,128}）
        #    cuda 版要求 M%64==0、N%64==0、d∈{64,128}；triton 版用 mask 限制更松。
        self.x_vals = [
            {"M": 512, "N": 512, "d": 64},
            {"M": 1024, "N": 1024, "d": 128},
            {"M": 2048, "N": 2048, "d": 128},
            {"M": 4096, "N": 4096, "d": 64},
        ]

        # 3. cuda 版无可调编译宏（Br/Bc/stage 在 kernel 内 constexpr 写死），
        #    每个 version 一个占位 config 让框架跑验证/压测即可。
        self.cuda_tuning_configs = self.make_configs(
            [{}], ["flashattentionv1", "flashattentionv2"]
        )
        # triton 版 solve 内部固定 BLOCK（未读 config），同样用占位 config。
        self.tuning_configs = self.make_configs(
            [{}], ["flashattentionv1", "flashattentionv2"]
        )

    # 1. 无编译宏
    def get_macros(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    # 2. 物理指标
    def get_bytes_accessed(self, case: Dict[str, Any]) -> int:
        M, N, d = case["M"], case["N"], case["d"]
        # 读 Q(Md)+K(Nd)+V(Nd) + 写 O(Md)，bf16 = 2 bytes
        return (M * d + N * d + N * d + M * d) * 2

    def get_total_flops(self, case: Dict[str, Any]) -> int:
        M, N, d = case["M"], case["N"], case["d"]
        return 4 * M * N * d  # QK^T (2MNd) + PV (2MNd)

    # 3. 验证：用 fp32 SDPA 作基准（升精度避免 bf16 基准掩盖误差）
    def reference_impl(self, Q, K, V, O, M, N, d, **kwargs):
        q = Q.float().view(1, 1, M, d)
        k = K.float().view(1, 1, N, d)
        v = V.float().view(1, 1, N, d)
        res = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                             dropout_p=0.0, is_causal=False)
        O.copy_(res.view(M, d).to(O.dtype))

    def _rand(self, m, n, d):
        dtype = torch.bfloat16
        return {
            "Q": torch.empty((m, d), device="cuda", dtype=dtype).uniform_(-0.1, 0.1),
            "K": torch.empty((n, d), device="cuda", dtype=dtype).uniform_(-0.1, 0.1),
            "V": torch.empty((n, d), device="cuda", dtype=dtype).uniform_(-0.1, 0.1),
            "O": torch.empty((m, d), device="cuda", dtype=dtype),
            "M": m, "N": n, "d": d,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        return self._rand(64, 64, 64)

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        # M%64==0, N%64==0, d∈{64,128}
        return [
            self._rand(64, 64, 64),
            self._rand(128, 128, 64),
            self._rand(128, 128, 128),
            self._rand(256, 128, 64),
        ]

    def generate_performance_test(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        M = cfg.get("M", 2048)
        N = cfg.get("N", 2048)
        d = cfg.get("d", 128)
        return self._rand(M, N, d)
