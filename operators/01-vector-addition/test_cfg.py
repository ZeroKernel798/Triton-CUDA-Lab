import torch
import ctypes
from typing import Any, Dict, List

class OperatorSpec:
    def __init__(self):
        self.name = "Vector Addition"
        self.atol = 1e-05
        self.rtol = 1e-05
        self.dtype = torch.float32

    def reference(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int):
        """PyTorch 标准实现用于精度对比"""
        torch.add(A, B, out=C)
        return C

    def get_cuda_args(self, test_case: Dict[str, Any]):
        """
        适配 run_lab.py 的 CUDA 参数传递
        test_case 是从下面的 generate_xxx 拿到的字典
        """
        A, B, C, N = test_case["A"], test_case["B"], test_case["C"], test_case["N"]
        return [A, B, C, N], [None, None, None, ctypes.c_size_t]

    def get_triton_args(self, test_case: Dict[str, Any]):
        """适配 run_lab.py 的 Triton 参数传递"""
        return {
            "a_ptr": test_case["A"],
            "b_ptr": test_case["B"],
            "c_ptr": test_case["C"],
            "n_elements": test_case["N"],
            "BLOCK_SIZE": 1024
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
                }
            )

        return test_cases

    def generate_performance_test(self) -> Dict[str, Any]:
        """性能测试：2500万级别的大规模向量，榨干带宽"""
        N = 25000000
        return {
            "name": "large_vector_bench",
            "A": torch.empty(N, device="cuda", dtype=self.dtype).uniform_(-100.0, 100.0),
            "B": torch.empty(N, device="cuda", dtype=self.dtype).uniform_(-100.0, 100.0),
            "C": torch.zeros(N, device="cuda", dtype=self.dtype),
            "N": N,
        }