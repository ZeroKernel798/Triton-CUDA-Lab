import itertools
from typing import Any, Dict, List

import torch

from core.spec import BaseOperatorSpec


class RMSNormSpec(BaseOperatorSpec):
    def __init__(self):
        super().__init__()
        # 1. 基础元数据与精度标准
        self.name = "RMSNorm"
        self.output_name = "Y"
        self.atol = 1e-5
        self.rtol = 1e-5
        self.eps = 1e-5

        # 2. 测试规模定义 (M: token/batch 维度, N: hidden 维度)
        self.x_vals = [
            {"M": 128, "N": 256},
            {"M": 512, "N": 1024},
            {"M": 1024, "N": 2048},
            {"M": 4096, "N": 4096},
            {"M": 8192, "N": 4096},
        ]
        self.perf_input = {"M": 4096, "N": 4096}

        # 3. Triton 配置生成
        triton_params = []
        for block_size, stages, warps in itertools.product(
            [128, 256, 512, 1024],
            [2, 3, 4],
            [4, 8],
        ):
            triton_params.append(
                {
                    "BLOCK_SIZE": block_size,
                    "num_stages": stages,
                    "num_warps": warps,
                }
            )
        self.tuning_configs = self.make_configs(triton_params, ["native"])

        # 4. CUDA 配置生成
        cuda_params = []
        for block_size, vec in itertools.product([128, 256, 512], [1, 2, 4]):
            cuda_params.append({"block_size": block_size, "vec": vec})
        self.cuda_tuning_configs = self.make_configs(cuda_params, ["native"])

    # 1. 将 Python Config 转换为 C++ 宏
    def get_macros(self, config: Dict[str, Any]) -> Dict[str, Any]:
        macros = {
            "BLOCK_SIZE": config.get("block_size"),
            "VEC": config.get("vec"),
        }
        return {k: v for k, v in macros.items() if v is not None}

    # 2. 物理指标计算
    def get_bytes_accessed(self, case: Dict[str, Any]) -> int:
        m = case.get("M", self.perf_input["M"])
        n = case.get("N", self.perf_input["N"])
        # 读 X (M*N) + 读 gamma (N) + 写 Y (M*N), float32 = 4 bytes
        return (m * n + n + m * n) * 4

    def get_total_flops(self, case: Dict[str, Any]) -> int:
        m = case.get("M", self.perf_input["M"])
        n = case.get("N", self.perf_input["N"])
        # 粗略估计: x^2 + reduce add + scale/rsqrt + 乘 gamma，按 ~5 FLOPs/element
        return m * n * 5

    # 3. 验证与测试实现
    def reference_impl(
        self,
        X: torch.Tensor,
        gamma: torch.Tensor,
        Y: torch.Tensor,
        M: int,
        N: int,
        eps: float,
        **kwargs,
    ):
        assert X.shape == (M, N)
        assert gamma.shape == (N,)
        assert Y.shape == (M, N)

        x_fp32 = X.float()
        rms = torch.rsqrt(torch.mean(x_fp32 * x_fp32, dim=1, keepdim=True) + eps)
        out = (x_fp32 * rms) * gamma.float()
        Y.copy_(out.to(X.dtype))

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        m, n = 2, 4
        return {
            "X": torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 2.0, 1.0]], device="cuda", dtype=dtype),
            "gamma": torch.ones(n, device="cuda", dtype=dtype),
            "Y": torch.empty(m, n, device="cuda", dtype=dtype),
            "M": m,
            "N": n,
            "eps": self.eps,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests: List[Dict[str, Any]] = []

        # 基础固定用例
        tests.append(
            {
                "X": torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 2.0, 1.0]], device="cuda", dtype=dtype),
                "gamma": torch.tensor([1.0, 1.0, 1.0, 1.0], device="cuda", dtype=dtype),
                "Y": torch.empty(2, 4, device="cuda", dtype=dtype),
                "M": 2,
                "N": 4,
                "eps": self.eps,
            }
        )
        tests.append(
            {
                "X": torch.zeros((3, 8), device="cuda", dtype=dtype),
                "gamma": torch.ones(8, device="cuda", dtype=dtype),
                "Y": torch.empty(3, 8, device="cuda", dtype=dtype),
                "M": 3,
                "N": 8,
                "eps": self.eps,
            }
        )
        tests.append(
            {
                "X": torch.empty((8, 16), device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                "gamma": torch.empty(16, device="cuda", dtype=dtype).uniform_(0.5, 1.5),
                "Y": torch.empty(8, 16, device="cuda", dtype=dtype),
                "M": 8,
                "N": 16,
                "eps": self.eps,
            }
        )

        # 边界尺寸
        for m, n in [(1, 1), (1, 1024), (1024, 256)]:
            tests.append(
                {
                    "X": torch.empty((m, n), device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                    "gamma": torch.ones(n, device="cuda", dtype=dtype),
                    "Y": torch.empty(m, n, device="cuda", dtype=dtype),
                    "M": m,
                    "N": n,
                    "eps": self.eps,
                }
            )

        return tests

    def generate_performance_test(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        dtype = torch.float32
        m = cfg.get("M", self.perf_input["M"])
        n = cfg.get("N", self.perf_input["N"])

        return {
            "X": torch.empty((m, n), device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
            "gamma": torch.empty(n, device="cuda", dtype=dtype).uniform_(0.5, 1.5),
            "Y": torch.empty(m, n, device="cuda", dtype=dtype),
            "M": m,
            "N": n,
            "eps": self.eps,
        }
