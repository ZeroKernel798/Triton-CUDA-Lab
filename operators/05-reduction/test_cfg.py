import torch
from typing import Any, Dict, List

class OperatorSpec:
    def __init__(self):
        self.name = "Reduction"
        self.output_name = "output" 
        # 保留原有数值
        self.atol = 1e-05
        self.rtol = 1e-05

    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, N: int):
        # 保留所有逻辑和断言
        assert input.shape == (N,)
        assert output.shape == (1,)
        assert input.dtype == output.dtype
        assert input.device == output.device
        # 使用 double 进行中间累加保证参考实现精度
        output[0] = torch.sum(input.double()).float()

    def get_solve_signature(self) -> List[str]:
        """
        对齐 Pybind11 接口：仅返回参数名称列表。
        顺序必须与 C++ 端的 solve(input, output, N) 完全一致。
        """
        return ["input", "output", "N"]

    def generate_example_test(self) -> Dict[str, Any]:
        # 保留原有数值 N=8
        dtype = torch.float32
        input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], device="cuda", dtype=dtype)
        output = torch.empty(1, device="cuda", dtype=dtype)
        N = 8
        return {"input": input, "output": output, "N": N}

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        # 保留所有原有的测试用例和数值
        dtype = torch.float32
        tests = []
        # basic_example
        tests.append({
            "input": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 8,
        })
        # negative_numbers
        tests.append({
            "input": torch.tensor([-2.5, 1.5, -1.0, 2.0], device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 4,
        })
        # single_element
        tests.append({
            "input": torch.tensor([42.0], device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 1,
        })
        # all_zeros
        tests.append({
            "input": torch.zeros(1024, device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 1024,
        })
        # all_ones
        tests.append({
            "input": torch.ones(1024, device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 1024,
        })
        # non_power_of_two
        tests.append({
            "input": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 5,
        })
        # large_random
        tests.append({
            "input": torch.empty(10000, device="cuda", dtype=dtype).uniform_(-1000.0, 1000.0),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 10000,
        })
        # large_random_2
        tests.append({
            "input": torch.empty(15000000, device="cuda", dtype=dtype).uniform_(0.0, 1000.0),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 15000000,
        })
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        # 保留原有跑分数值 N=4,194,304
        dtype = torch.float32
        N = 4_194_304
        input = torch.empty(N, device="cuda", dtype=dtype).uniform_(0.0, 1000.0)
        output = torch.empty(1, device="cuda", dtype=dtype)
        return {"input": input, "output": output, "N": N}