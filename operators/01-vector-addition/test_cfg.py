import torch
import ctypes
from typing import Any, Dict, List

class OperatorSpec:
    def __init__(self):
        # 设置算子和输出名字
        self.name = "Vector Addition"
        self.output_name = "C"
        # 设置精度要求
        self.atol = 1e-05
        self.rtol = 1e-05
        # 数据规模测试
        self.perf_input = 1024 * 1024 * 16
        self.x_vals = [2**i for i in range(12, 25)]
        self.base_cfg = {
            "block_size": 256,    # 1D 用的
            "BLOCK_SIZE": 1024,   # Triton 用的
            "num_warps": 4        # Triton 用的
        }
        # 核函数参数调优
        # 针对triton
        self.tuning_configs = [
            {"BLOCK_SIZE": 32, "num_warps": 2},
            {"BLOCK_SIZE": 64, "num_warps": 4},
            {"BLOCK_SIZE": 128, "num_warps": 4},
            {"BLOCK_SIZE": 256, "num_warps": 8},
        ]
        # 针对cuda
        self.cuda_tuning_configs = [
            {"block_size": 32},   
            {"block_size": 128},
            {"block_size": 256},  
            {"block_size": 512},
            {"block_size": 1024}, 
        ]

    def get_throughput(self, n = None, ms = None):
        # 向量加法吞吐量计算
        if n is None:
            n = self.perf_input
        return (n * 4 * 3) / 1e9 / (ms / 1000)

    # pytorch标准
    def reference_impl(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int, **kwargs):
        assert A.shape == B.shape == C.shape
        assert A.dtype == B.dtype == C.dtype
        assert A.device == B.device == C.device

        torch.add(A, B, out=C)

    # 生成最简单的测试案例 最小闭环逻辑验证
    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 4
        A = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda", dtype=dtype)
        B = torch.tensor([5.0, 6.0, 7.0, 8.0], device="cuda", dtype=dtype)
        C = torch.empty(N, device="cuda", dtype=dtype)
        return {
            "A": A,
            "B": B,
            "C": C,
            "N": N,
            **self.base_cfg 
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
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
            (
                "very_small_numbers",
                [0.000001, 0.0000001, 0.00000001, 0.000000001],
                [0.000001, 0.0000001, 0.00000001, 0.000000001],
            ),
            (
                "large_numbers",
                [1000000.0, 10000000.0, -1000000.0, -10000000.0],
                [1000000.0, -10000000.0, -1000000.0, 10000000.0],
            ),
        ]

        test_cases = []
        for _, a_vals, b_vals in test_specs:
            n = len(a_vals)
            test_cases.append(
                {
                    "A": torch.tensor(a_vals, device="cuda", dtype=dtype),
                    "B": torch.tensor(b_vals, device="cuda", dtype=dtype),
                    "C": torch.zeros(n, device="cuda", dtype=dtype),
                    "N": n,
                    **self.base_cfg 
                }
            )

        # Random test cases
        for _, size, a_range, b_range in [
            ("powers_of_two_size", 32, (0.0, 32.0), (0.0, 64.0)),
            ("medium_sized_vector", 1000, (0.0, 7.0), (0.0, 5.0)),
            ("large_vector", 10000, (0.0, 1.0), (0.0, 1.0)),
        ]:
            test_cases.append(
                {
                    "A": torch.empty(size, device="cuda", dtype=dtype).uniform_(*a_range),
                    "B": torch.empty(size, device="cuda", dtype=dtype).uniform_(*b_range),
                    "C": torch.zeros(size, device="cuda", dtype=dtype),
                    "N": size,
                    **self.base_cfg  
                }
            )

        return test_cases

    def generate_performance_test(self, N=None) -> Dict[str, Any]:
        if N is None:
            N = self.perf_input
        dtype = torch.float32
        return {
            "A": torch.empty(N, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
            "B": torch.empty(N, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
            "C": torch.zeros(N, device="cuda", dtype=dtype),
            "N": N,
            "block_size": 256, # 默认值，会被 tuning 模式覆盖
            "BLOCK_SIZE": 256, # Triton 默认值
            **self.base_cfg 
        }