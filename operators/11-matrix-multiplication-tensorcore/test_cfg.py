import torch
from typing import Any, Dict, List

from core.spec import BaseOperatorSpec

# 本算子是 TF32 Tensor Core kernel；让 PyTorch 基线的 matmul 也走 TF32 张量核，
# 保证性能对比是 TF32 vs TF32 的公平口径（而非 TF32 kernel 对 FP32 cuBLAS）。
torch.backends.cuda.matmul.allow_tf32 = True

# TF32 GEMM 与 fp32 基准对比时，"抵消后接近 0"的输出元素绝对误差天然偏大
# （误差来自被累加的大中间乘积，而非最终小值），逐元素 atol 会误判。
# 因此用"最大绝对误差 / 参考矩阵峰值"这一相对峰值判据，符合 GEMM 数值校验惯例。
_TC_REL_PEAK_TOL = 2e-2


class MatrixMultiplicationTCSpec(BaseOperatorSpec):
    """TF32 Tensor Core SGEMM 的逐版本优化链。

    与 04-matrix-multiplication（fp32 SIMT）分目录共存，参考 09-flash-attention-tensorcore
    的 Tensor Core 算子组织方式。kernel 用 mma.m16n8k8 (tf32) 计算，输入仍是 fp32 张量，
    Tensor Core 内部按 TF32 截断尾数，因此对 fp32 基准放宽容差。
    """

    # block tiling 128x128 / BK=16；这些 version 都要求 shape 对齐到 tile。
    VERSIONS = ["bt", "bt_swizzle", "swizzle_bcf", "swizzle_bcf_dbf"]

    def __init__(self):
        super().__init__()
        self.name = "Matrix multiplication (Tensor Core, tf32)"
        self.output_name = "C"
        # TF32 仅 10-bit 尾数，大 K 累加后与 fp32 基准的相对误差偏大，放宽容差。
        self.atol = 1e-2
        self.rtol = 1e-2

        # 性能 scaling 规模（M,N % 128 == 0，K % 16 == 0）。
        self.x_vals = [
            {"M": 512, "K": 512, "N": 512},
            {"M": 1024, "K": 1024, "N": 1024},
            {"M": 2048, "K": 2048, "N": 2048},
            {"M": 4096, "K": 4096, "N": 4096},
            {"M": 8192, "K": 4096, "N": 8192},
        ]

        # kernel 内 tile 形态 constexpr 写死，无需注入宏：每个 version 一个占位 config。
        self.cuda_tuning_configs = self.make_configs([{}], self.VERSIONS)
        # 本算子聚焦 CUDA Tensor Core 演进，不提供 triton 版。
        self.tuning_configs = []

    # 相对峰值误差判据（覆盖基类的逐元素 allclose）
    def validate(self, outputs: Dict[str, torch.Tensor], reference: Dict[str, torch.Tensor]) -> bool:
        out = outputs.get(self.output_name)
        ref = reference.get(self.output_name)
        if out is None or ref is None:
            raise KeyError(f"找不到输出键名: {self.output_name}")
        denom = ref.abs().max().clamp_min(1e-6)
        rel = (out - ref).abs().max() / denom
        return bool(rel.item() < _TC_REL_PEAK_TOL)

    # 1. 无编译宏
    def get_macros(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    # 2. 物理指标（与 04 一致：fp32 4B；2*M*N*K flop）
    def get_bytes_accessed(self, case: Dict[str, Any]) -> int:
        m, k, n = case["M"], case["K"], case["N"]
        return (m * k + k * n + m * n) * 4

    def get_total_flops(self, case: Dict[str, Any]) -> int:
        m, k, n = case["M"], case["K"], case["N"]
        return 2 * m * n * k

    # 3. 基准：fp32 torch.matmul（与被测 TF32 kernel 的差异由相对峰值判据吸收）
    def reference_impl(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                       M: int, N: int, K: int, **kwargs):
        assert A.shape == (M, K)
        assert B.shape == (K, N)
        assert C.shape == (M, N)
        torch.matmul(A, B, out=C)

    def _make_case(self, M: int, K: int, N: int) -> Dict[str, Any]:
        dtype = torch.float32
        return {
            "A": torch.empty(M, K, device="cuda", dtype=dtype).uniform_(-2.0, 2.0),
            "B": torch.empty(K, N, device="cuda", dtype=dtype).uniform_(-2.0, 2.0),
            "C": torch.empty(M, N, device="cuda", dtype=dtype),
            "M": M, "N": N, "K": K,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        return self._make_case(128, 128, 128)

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        # 仅对齐 shape：M,N % 128 == 0，K % 16 == 0。
        return [self._make_case(M, K, N) for M, K, N in [
            (128, 128, 128),
            (256, 256, 256),
            (256, 128, 512),
            (512, 512, 256),
        ]]

    def generate_performance_test(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        M, N, K = cfg.get("M", 4096), cfg.get("N", 4096), cfg.get("K", 4096)
        dtype = torch.float32
        return {
            "A": torch.empty(M, K, device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
            "B": torch.empty(K, N, device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
            "C": torch.empty(M, N, device="cuda", dtype=dtype),
            "M": M, "N": N, "K": K,
        }
