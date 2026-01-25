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
        assert A.shape == (N, N)
        assert B.shape == (N, N)
        assert C.shape == (N, N)
        assert A.dtype == B.dtype == C.dtype
        assert A.device == B.device == C.device
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
        # 算出一共有多少个 float32 元素
        total_elements = test_case["N"] * test_case["N"] 
        return {
            "a_ptr": test_case["A"],
            "b_ptr": test_case["B"],
            "c_ptr": test_case["C"],
            "n_elements": total_elements, # 传总数给 Triton 的 mask
            "BLOCK_SIZE": 1024
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        test_cases = []

        # basic_2x2
        test_cases.append(
            {
                "A": torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=dtype),
                "B": torch.tensor([[5.0, 6.0], [7.0, 8.0]], device="cuda", dtype=dtype),
                "C": torch.zeros((2, 2), device="cuda", dtype=dtype),
                "N": 2,
            }
        )

        # all_zeros_4x4
        test_cases.append(
            {
                "A": torch.zeros((4, 4), device="cuda", dtype=dtype),
                "B": torch.zeros((4, 4), device="cuda", dtype=dtype),
                "C": torch.zeros((4, 4), device="cuda", dtype=dtype),
                "N": 4,
            }
        )

        # identity_plus_identity_3x3
        test_cases.append(
            {
                "A": torch.tensor(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device="cuda", dtype=dtype
                ),
                "B": torch.tensor(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device="cuda", dtype=dtype
                ),
                "C": torch.zeros((3, 3), device="cuda", dtype=dtype),
                "N": 3,
            }
        )

        # negative_values_2x2
        test_cases.append(
            {
                "A": torch.tensor([[-1.0, -2.0], [-3.0, -4.0]], device="cuda", dtype=dtype),
                "B": torch.tensor([[-5.0, -6.0], [-7.0, -8.0]], device="cuda", dtype=dtype),
                "C": torch.zeros((2, 2), device="cuda", dtype=dtype),
                "N": 2,
            }
        )

        # mixed_positive_negative_2x2
        test_cases.append(
            {
                "A": torch.tensor([[1.0, -2.0], [-3.0, 4.0]], device="cuda", dtype=dtype),
                "B": torch.tensor([[-1.0, 2.0], [3.0, -4.0]], device="cuda", dtype=dtype),
                "C": torch.zeros((2, 2), device="cuda", dtype=dtype),
                "N": 2,
            }
        )

        # single_element_1x1
        test_cases.append(
            {
                "A": torch.tensor([[42.0]], device="cuda", dtype=dtype),
                "B": torch.tensor([[8.0]], device="cuda", dtype=dtype),
                "C": torch.zeros((1, 1), device="cuda", dtype=dtype),
                "N": 1,
            }
        )

        # large_N_16x16
        test_cases.append(
            {
                "A": torch.empty((16, 16), device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
                "B": torch.empty((16, 16), device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
                "C": torch.zeros((16, 16), device="cuda", dtype=dtype),
                "N": 16,
            }
        )

        # very_small_numbers
        test_cases.append(
            {
                "A": torch.tensor(
                    [[0.000001, 0.0000001], [0.00000001, 0.000000001]], device="cuda", dtype=dtype
                ),
                "B": torch.tensor(
                    [[0.000001, 0.0000001], [0.00000001, 0.000000001]], device="cuda", dtype=dtype
                ),
                "C": torch.zeros((2, 2), device="cuda", dtype=dtype),
                "N": 2,
            }
        )

        # large_numbers
        test_cases.append(
            {
                "A": torch.tensor(
                    [[1000000.0, 10000000.0], [-1000000.0, -10000000.0]], device="cuda", dtype=dtype
                ),
                "B": torch.tensor(
                    [[1000000.0, -10000000.0], [-1000000.0, 10000000.0]], device="cuda", dtype=dtype
                ),
                "C": torch.zeros((2, 2), device="cuda", dtype=dtype),
                "N": 2,
            }
        )

        # non_power_of_two_size_7x7
        test_cases.append(
            {
                "A": torch.empty((7, 7), device="cuda", dtype=dtype).uniform_(-5.0, 5.0),
                "B": torch.empty((7, 7), device="cuda", dtype=dtype).uniform_(-5.0, 5.0),
                "C": torch.zeros((7, 7), device="cuda", dtype=dtype),
                "N": 7,
            }
        )

        # medium_size_32x32
        test_cases.append(
            {
                "A": torch.empty((32, 32), device="cuda", dtype=dtype).uniform_(-100.0, 100.0),
                "B": torch.empty((32, 32), device="cuda", dtype=dtype).uniform_(-100.0, 100.0),
                "C": torch.zeros((32, 32), device="cuda", dtype=dtype),
                "N": 32,
            }
        )

        return test_cases

    def generate_performance_test(self) -> Dict[str, Any]:
        """性能测试"""
        dtype = torch.float32
        N = 4096
        return {
            "A": torch.empty(N, N, device="cuda", dtype=dtype).uniform_(-1000.0, 1000.0),
            "B": torch.empty(N, N, device="cuda", dtype=dtype).uniform_(-1000.0, 1000.0),
            "C": torch.zeros(N, N, device="cuda", dtype=dtype),
            "N": N,
        }