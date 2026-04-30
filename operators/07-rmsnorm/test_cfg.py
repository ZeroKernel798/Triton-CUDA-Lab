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
        self.atol = 1e-2
        self.rtol = 1e-2
        self.eps = 1e-5

        # 2. 测试规模定义 (M: token/batch 维度, N: hidden 维度)
        self.x_vals = [
            {"M": 8192, "N": 128}, # 测试下 N 较小的时候，限制 block 数量是否有优势
            {"M": 512, "N": 1024},
            {"M": 1024, "N": 2048},
            {"M": 4096, "N": 4096},
            {"M": 8192, "N": 4096},
            {"M": 8192, "N": 8192},
        ]
        self.perf_input = {"M": 8192, "N": 8192}

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
        # native CUDA 实现是一行一个 warp，因此 block_x 固定为 32。
        cuda_native_params = [
            {"block_x": 32, "block_y": 1},
            {"block_x": 32, "block_y": 2},
            {"block_x": 32, "block_y": 4},
            {"block_x": 32, "block_y": 8},
            {"block_x": 32, "block_y": 16},
        ]
        # rowblock CUDA 实现是一个 block 负责一行，因此只调每行线程数。
        cuda_rowblock_params = [
            {"block_x": 32},
            {"block_x": 64},
            {"block_x": 128},
            {"block_x": 256},
            {"block_x": 512},
        ]
        # rowblockvec 在 rowblock 基础上每个线程向量化处理 4 个 bf16 元素。
        cuda_rowblockvec_params = [
            {"block_x": 32, "vec_size": 4},
            {"block_x": 64, "vec_size": 4},
            {"block_x": 128, "vec_size": 4},
            {"block_x": 256, "vec_size": 4},
            {"block_x": 512, "vec_size": 4},
        ]
        # rowblockvec_smem 在 rowblockvec 基础上用 shared memory 缓存每行 X，减少一次全局读 X。
        cuda_rowblockvec_smem_params = [
            {"block_x": 32, "vec_size": 4},
            {"block_x": 64, "vec_size": 4},
            {"block_x": 128, "vec_size": 4},
            {"block_x": 256, "vec_size": 4},
            {"block_x": 512, "vec_size": 4},
        ]
        # rowblockvec_smem_gridstride 固定 block 数为 SM 数的倍数，kernel 内用 grid-stride loop 覆盖 M 行。
        cuda_rowblockvec_smem_gridstride_params = [
            {"block_x": block_x, "vec_size": 4, "grid_stride_sm_mult": sm_mult}
            for block_x, sm_mult in itertools.product([32, 64, 128, 256, 512], [2, 4, 6, 8])
        ]
        self.cuda_tuning_configs = self.make_configs(cuda_native_params, ["native"])
        self.cuda_tuning_configs.extend(self.make_configs(cuda_rowblock_params, ["rowblock"]))
        self.cuda_tuning_configs.extend(self.make_configs(cuda_rowblockvec_params, ["rowblockvec"]))
        self.cuda_tuning_configs.extend(self.make_configs(cuda_rowblockvec_smem_params, ["rowblockvec_smem"]))
        self.cuda_tuning_configs.extend(
            self.make_configs(cuda_rowblockvec_smem_gridstride_params, ["rowblockvec_smem_gridstride"])
        )

    # 1. 将 Python Config 转换为 C++ 宏
    def get_macros(self, config: Dict[str, Any]) -> Dict[str, Any]:
        macros = {
            "BLOCK_X": config.get("block_x"),
            "BLOCK_Y": config.get("block_y"),
            "VEC_SIZE": config.get("vec_size"),
            "GRID_STRIDE_SM_MULT": config.get("grid_stride_sm_mult"),
        }
        return {k: v for k, v in macros.items() if v is not None}

    # 2. 物理指标计算
    def get_bytes_accessed(self, case: Dict[str, Any]) -> int:
        m = case.get("M", self.perf_input["M"])
        n = case.get("N", self.perf_input["N"])
        config = case.get("_config", {})
        version = config.get("version", "") if isinstance(config, dict) else ""

        # smem 版本第一遍读 X 后缓存到 shared memory，第二阶段从 smem 读 X，
        # 因此 global memory traffic 是读 X 一次 + 读 gamma 一次 + 写 Y 一次。
        if version in {"rowblockvec_smem", "rowblockvec_smem_gridstride"}:
            return (m * n + m * n + m * n) * 2

        # 其它 CUDA/Triton 实现是 two-pass：第一遍读 X 算 sum，第二遍读 X/gamma 并写 Y。
        # 读 X 两次 (2*M*N) + 读 gamma (M*N) + 写 Y (M*N), bfloat16 = 2 bytes
        return (2 * m * n + m * n + m * n) * 2

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
        return ((x_fp32 * rms) * gamma.float()).to(X.dtype)

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.bfloat16
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
        dtype = torch.bfloat16
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
        dtype = torch.bfloat16
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
