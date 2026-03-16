import torch
from typing import Any, Dict, List
from core.spec import BaseOperatorSpec

class VectorAdditionSpec(BaseOperatorSpec):
    def __init__(self):
        super().__init__()
        # 基础元数据
        self.name = "Vector Addition"
        self.output_name = "C"
        self.atol = 1e-05
        self.rtol = 1e-05
        
        # 测试规模定义
        self.perf_input = 1024 * 1024 * 16
        self.x_vals = [{"N": 2**i} for i in range(12, 16)]

        # 1. Triton 配置 
        triton_params = [
            {"BLOCK_SIZE": 32, "num_warps": 2},
            {"BLOCK_SIZE": 64, "num_warps": 4},
            {"BLOCK_SIZE": 128, "num_warps": 4},
            {"BLOCK_SIZE": 256, "num_warps": 8},
        ]
        self.tuning_configs = self.make_configs(triton_params, ["main"])

        # 2. CUDA 配置
        cuda_1d_params = [{"block_size": bs} for bs in [32, 128, 256, 512, 1024]]
        # native 和 float4 版本共享这套 1D Block 配置 添加cublas是为了能够参与编译
        self.cuda_tuning_configs = self.make_configs(cuda_1d_params, ["native", "float4"])
        self.cuda_tuning_configs.append({"version": "official"})

    # 1. 将 Python Config 转换为 C++ 宏
    def get_macros(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # 统一将配置中的 block_size 映射为 C++ 侧的宏
        return {
            "BLOCK_SIZE": config.get("block_size", 256)
        }

    # 2. 物理指标计算
    def get_bytes_accessed(self, case: Dict[str, Any]) -> int:
        n = case.get("N", self.perf_input)
        # 读 A, B + 写 C (float32 = 4 bytes)
        return n * 4 * 3

    def get_total_flops(self, case: Dict[str, Any]) -> int:
        n = case.get("N", self.perf_input)
        # 向量加法，每个位置 1 次计算
        return n

    # 3. 验证与测试
    def reference_impl(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int, **kwargs):
        """标准 PyTorch 实现"""
        assert A.shape == B.shape == C.shape
        assert A.dtype == B.dtype == C.dtype
        assert A.device == B.device == C.device
        torch.add(A, B, out=C)

    def generate_example_test(self) -> Dict[str, Any]:
        """最小闭环逻辑验证"""
        dtype = torch.float32
        N = 4
        return {
            "A": torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda", dtype=dtype),
            "B": torch.tensor([5.0, 6.0, 7.0, 8.0], device="cuda", dtype=dtype),
            "C": torch.empty(N, device="cuda", dtype=dtype),
            "N": N,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        """全量功能回归测试"""
        dtype = torch.float32
        test_specs = [
            ("scalar_tail_1", [1.0], [2.0]),
            ("scalar_tail_2", [1.0, 2.0], [3.0, 4.0]),
            ("scalar_tail_3", [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
            ("basic_small", [1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]),
            ("all_zeros", [0.0] * 16, [0.0] * 16),
            ("non_power_of_two", [1.0] * 30, [2.0] * 30),
            ("negative_numbers", [-1.0, -2.0, -3.0, -4.0], [-5.0, -6.0, -7.0, -8.0]),
            ("mixed_positive_negative", [1.0, -2.0, 3.0, -4.0], [-1.0, 2.0, -3.0, 4.0]),
            ("very_small_numbers", [1e-6, 1e-7, 1e-8, 1e-9], [1e-6, 1e-7, 1e-8, 1e-9]),
            ("large_numbers", [1e6, 1e7, -1e6, -1e7], [1e6, -1e7, -1e6, 1e7]),
        ]

        test_cases = []
        for _, a_vals, b_vals in test_specs:
            n = len(a_vals)
            test_cases.append({
                "A": torch.tensor(a_vals, device="cuda", dtype=dtype),
                "B": torch.tensor(b_vals, device="cuda", dtype=dtype),
                "C": torch.zeros(n, device="cuda", dtype=dtype),
                "N": n,
            })

        # 随机规模测试
        for _, size, a_range, b_range in [
            ("powers_of_two_size", 32, (0.0, 32.0), (0.0, 64.0)),
            ("medium_sized_vector", 1000, (0.0, 7.0), (0.0, 5.0)),
            ("large_vector", 10000, (0.0, 1.0), (0.0, 1.0)),
        ]:
            test_cases.append({
                "A": torch.empty(size, device="cuda", dtype=dtype).uniform_(*a_range),
                "B": torch.empty(size, device="cuda", dtype=dtype).uniform_(*b_range),
                "C": torch.zeros(size, device="cuda", dtype=dtype),
                "N": size,
            })
        return test_cases

    def generate_performance_test(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """压测数据生成"""
        dtype = torch.float32
        N = cfg.get("N", self.perf_input)
        return {
            "A": torch.empty(N, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
            "B": torch.empty(N, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
            "C": torch.zeros(N, device="cuda", dtype=dtype),
            "N": N,
        }