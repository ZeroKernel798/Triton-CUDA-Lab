import torch
from typing import Any, Dict, List
from core.spec import BaseOperatorSpec

class MatrixMultiplicationSpec(BaseOperatorSpec):
    VECTOR_LOAD_VERSIONS = {
        "smem_vector",
        "smem_outer8x8",
        "smem_outer8x8_at",
        "smem_outer8x8_at_swizzling",
        "swizzling_mem_coalesced",
        "double_buffer",
        "cpasync"
    }

    def __init__(self):
        super().__init__()
        # 1. 基础元数据与精度标准
        self.name = "Matrix multiplication"
        self.output_name = "C"
        self.atol = 1e-02
        self.rtol = 1e-02
        
        # 2. 数据规模测试 (保持原有的 x_vals)
        self.x_vals = [
            {"M": 128, "K": 128, "N": 128},
            {"M": 512, "K": 1024, "N": 512},
            {"M": 4096, "K": 128, "N": 8192},
            {"M": 4096, "K": 4096, "N": 4096},
            {"M": 8192, "K": 4096, "N": 8192},
        ]

        # Triton 配置：只覆盖对齐 tile 形态，避免把搜索空间扩得过大。
        tile_configs = [
            {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "num_warps": 4, "num_stages": 3},
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 32, "num_warps": 4, "num_stages": 3},
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "num_warps": 4, "num_stages": 3},
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "num_warps": 4, "num_stages": 3},
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "num_warps": 4, "num_stages": 3},
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "num_warps": 8, "num_stages": 3},
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "num_warps": 4, "num_stages": 4},
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64, "num_warps": 4, "num_stages": 4},
        ]
        self.tuning_configs = self.make_configs(tile_configs, ["tile"])

        swizzle_configs = [
            {**cfg, "GROUP_SIZE_M": group_size}
            for cfg in tile_configs
            for group_size in (4, 8)
        ]
        swizzle_configs.extend([
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 16, "num_warps": 4, "num_stages": 3},
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 16, "num_warps": 4, "num_stages": 3},
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 3},
        ])
        self.tuning_configs.extend(self.make_configs(swizzle_configs, ["swizzle"]))
        self.cuda_tuning_configs = self.make_configs([{}], [
            "native",
            "smem",
            "smem_vector",
            "smem_outer8x8",
            "smem_outer8x8_at",
            "smem_outer8x8_at_swizzling",
            "swizzling_mem_coalesced",
            "double_buffer",
            "cpasync"
        ])

    def _make_case(self, M: int, K: int, N: int, dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
        return {
            "A": torch.empty(M, K, device="cuda", dtype=dtype).uniform_(-2.0, 2.0),
            "B": torch.empty(K, N, device="cuda", dtype=dtype).uniform_(-2.0, 2.0),
            "C": torch.empty(M, N, device="cuda", dtype=dtype),
            "M": M,
            "N": N,
            "K": K,
        }

    def _aligned_shapes(self, version: str) -> List[tuple[int, int, int]]:
        if version in self.VECTOR_LOAD_VERSIONS:
            # float4 路径只验证满足 16B 对齐假设的 shape，避免把 fallback 逻辑塞回 kernel。
            return [
                (32, 32, 32),
                (64, 64, 64),
                (128, 128, 128),
                (128, 256, 64),
            ]
        return [
            (4, 4, 4),
            (8, 10, 6),
            (16, 20, 12),
            (32, 16, 8),
        ]

    # 1. 将 Python Config 转换为 C++ 宏
    def get_macros(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # 矩阵乘法当前只测试输入 shape，不从 Python 注入 kernel 超参数。
        return {}

    # 2. 物理指标计算 (对应原有的 get_throughput 和 get_flops 逻辑)
    def get_bytes_accessed(self, case: Dict[str, Any]) -> int:
        m, k, n = case["M"], case["K"], case["N"]
        # 理论最小访存量：读 A + 读 B + 写 C (float32 = 4 bytes)
        return (m * k + k * n + m * n) * 4

    def get_total_flops(self, case: Dict[str, Any]) -> int:
        m, k, n = case["M"], case["K"], case["N"]
        # 总操作数: 2 * M * N * K
        return 2 * m * n * k

    # 3. 验证与测试实现
    def reference_impl(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, M: int, N: int, K: int, **kwargs):
        """标准实现"""
        assert A.shape == (M, K)
        assert B.shape == (K, N)
        assert C.shape == (M, N)
        torch.matmul(A, B, out=C)

    def generate_example_test(self) -> Dict[str, Any]:
        """最小闭环验证 """
        return self._make_case(4, 4, 4)

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        """全量功能回归测试"""
        return [self._make_case(M, K, N) for M, K, N in [
            (4, 4, 4),
            (8, 10, 6),
            (16, 20, 12),
            (32, 16, 8),
        ]]

    def get_validation_cases(self, version: str, config: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        if version in self.VECTOR_LOAD_VERSIONS or version in {"swizzling_mem_coalesced", "double_buffer"}:
            return [self._make_case(M, K, N) for M, K, N in self._aligned_shapes(version)]
        return [self.generate_example_test()] + self.generate_functional_test()

    def generate_performance_test(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """压测数据生成"""
        dtype = torch.float32
        M, N, K = cfg.get("M", 8192), cfg.get("N", 4096), cfg.get("K", 8192)
        return {
            "A": torch.empty(M, K, device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
            "B": torch.empty(K, N, device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
            "C": torch.empty(M, N, device="cuda", dtype=dtype),
            "M": M, "N": N, "K": K,
        }
