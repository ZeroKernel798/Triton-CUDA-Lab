import torch
from typing import Any, Dict, List

class OperatorSpec:
    def __init__(self):
        # 设置算子名字以及输出名字
        self.name = "Matrix Addition"
        self.output_name = "C"  
        # 设置精度标准
        self.atol = 1e-05
        self.rtol = 1e-05
        # 数据规模测试
        self.x_vals = [{"N": 2**i} for i in range(8, 14)]
        self.perf_input = 8192
        # 默认配置 给简单测试使用
        self.base_cfg = {
            "block_size": 256,    # 1D 用的
            "bx": 16,             # 2D 用的 x
            "by": 16,             # 2D 用的 y
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
            {"block_size": 64},   
            {"block_size": 256},  
            {"block_size": 1024}, 
            {"bx": 32, "by": 8},  
            {"bx": 32, "by": 8},  
            {"bx": 16, "by": 16},  
            {"bx": 8, "by": 32},
        ]
    
    def get_throughput(self, case: Dict[str, Any], ms = None):
        if ms is None or ms == 0: return 0
        n = case.get("N", self.perf_input)
        total_elements = n * n
        return (total_elements * 4 * 3) / 1e9 / (ms / 1000)

    def reference_impl(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int, **kwargs):
        assert A.shape == (N, N)
        assert B.shape == (N, N)
        assert C.shape == (N, N)
        assert A.dtype == B.dtype == C.dtype
        assert A.device == B.device == C.device

        torch.add(A, B, out=C)

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 2
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=dtype)
        B = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device="cuda", dtype=dtype)
        C = torch.empty(N, N, device="cuda", dtype=dtype)
        return {
            "A": A,
            "B": B,
            "C": C,
            "N": N,
            **self.base_cfg 
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        test_cases = []

        # basic_2x2
        test_cases.append({
            "A": torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=dtype),
            "B": torch.tensor([[5.0, 6.0], [7.0, 8.0]], device="cuda", dtype=dtype),
            "C": torch.zeros((2, 2), device="cuda", dtype=dtype),
            "N": 2,
            **self.base_cfg
        })

        # all_zeros_4x4
        test_cases.append({
            "A": torch.zeros((4, 4), device="cuda", dtype=dtype),
            "B": torch.zeros((4, 4), device="cuda", dtype=dtype),
            "C": torch.zeros((4, 4), device="cuda", dtype=dtype),
            "N": 4,
            **self.base_cfg 
        })

        # identity_plus_identity_3x3
        test_cases.append({
            "A": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device="cuda", dtype=dtype),
            "B": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device="cuda", dtype=dtype),
            "C": torch.zeros((3, 3), device="cuda", dtype=dtype),
            "N": 3,
            **self.base_cfg 
        })

        # negative_values_2x2
        test_cases.append({
            "A": torch.tensor([[-1.0, -2.0], [-3.0, -4.0]], device="cuda", dtype=dtype),
            "B": torch.tensor([[-5.0, -6.0], [-7.0, -8.0]], device="cuda", dtype=dtype),
            "C": torch.zeros((2, 2), device="cuda", dtype=dtype),
            "N": 2,
            **self.base_cfg 
        })

        # mixed_positive_negative_2x2
        test_cases.append({
            "A": torch.tensor([[1.0, -2.0], [-3.0, 4.0]], device="cuda", dtype=dtype),
            "B": torch.tensor([[-1.0, 2.0], [3.0, -4.0]], device="cuda", dtype=dtype),
            "C": torch.zeros((2, 2), device="cuda", dtype=dtype),
            "N": 2,
            **self.base_cfg 
        })

        # single_element_1x1
        test_cases.append({
            "A": torch.tensor([[42.0]], device="cuda", dtype=dtype),
            "B": torch.tensor([[8.0]], device="cuda", dtype=dtype),
            "C": torch.zeros((1, 1), device="cuda", dtype=dtype),
            "N": 1,
            **self.base_cfg 
        })

        # large_N_16x16
        test_cases.append({
            "A": torch.empty((16, 16), device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
            "B": torch.empty((16, 16), device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
            "C": torch.zeros((16, 16), device="cuda", dtype=dtype),
            "N": 16,
            **self.base_cfg 
        })

        # very_small_numbers
        test_cases.append({
            "A": torch.tensor([[0.000001, 0.0000001], [0.00000001, 0.000000001]], device="cuda", dtype=dtype),
            "B": torch.tensor([[0.000001, 0.0000001], [0.00000001, 0.000000001]], device="cuda", dtype=dtype),
            "C": torch.zeros((2, 2), device="cuda", dtype=dtype),
            "N": 2,
            **self.base_cfg 
        })

        # large_numbers
        test_cases.append({
            "A": torch.tensor([[1000000.0, 10000000.0], [-1000000.0, -10000000.0]], device="cuda", dtype=dtype),
            "B": torch.tensor([[1000000.0, -10000000.0], [-1000000.0, 10000000.0]], device="cuda", dtype=dtype),
            "C": torch.zeros((2, 2), device="cuda", dtype=dtype),
            "N": 2,
            **self.base_cfg 
        })

        # non_power_of_two_size_7x7
        test_cases.append({
            "A": torch.empty((7, 7), device="cuda", dtype=dtype).uniform_(-5.0, 5.0),
            "B": torch.empty((7, 7), device="cuda", dtype=dtype).uniform_(-5.0, 5.0),
            "C": torch.zeros((7, 7), device="cuda", dtype=dtype),
            "N": 7,
            **self.base_cfg 
        })

        # medium_size_32x32
        test_cases.append({
            "A": torch.empty((32, 32), device="cuda", dtype=dtype).uniform_(-100.0, 100.0),
            "B": torch.empty((32, 32), device="cuda", dtype=dtype).uniform_(-100.0, 100.0),
            "C": torch.zeros((32, 32), device="cuda", dtype=dtype),
            "N": 32,
            **self.base_cfg  
        })

        return test_cases

    def generate_performance_test(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        # 如果没设置N 则使用perf_input
        dtype = torch.float32
        N = cfg.get("N", self.perf_input)
        return {
            "A": torch.empty(N, N, device="cuda", dtype=dtype).uniform_(-1000.0, 1000.0),
            "B": torch.empty(N, N, device="cuda", dtype=dtype).uniform_(-1000.0, 1000.0),
            "C": torch.zeros(N, N, device="cuda", dtype=dtype),
            "N": N,
            **self.base_cfg  
        }