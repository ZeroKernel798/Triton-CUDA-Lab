import torch
from typing import Any, Dict, List

class OperatorSpec:
    def __init__(self):
        # 设置算子名字以及输出名字
        self.name = "Matrix Transpose"
        self.output_name = "output" 
        # 设置参数列表
        self.arg_names = ["input", "output", "rows", "cols", "block_size"]
        # 设置精度标准
        self.atol = 1e-05
        self.rtol = 1e-05
        # 数据规模测试
        self.x_vals = [2**i for i in range(10, 25)] 
        self.perf_input = 2**24 # 约 16M 元素
        # 核函数参数调优
        # 针对triton
        self.tuning_configs = [
            {"BLOCK_SIZE": 1024, "num_warps": 4, "num_stages": 2},
            {"BLOCK_SIZE": 2048, "num_warps": 8, "num_stages": 3},
            {"BLOCK_SIZE": 4096, "num_warps": 16, "num_stages": 3},
        ]
        
        # CUDA 配置
        self.cuda_tuning_configs = [
            {"block_size": 128}, {"block_size": 256}, {"block_size": 512}, {"block_size": 1024}
        ]
    
    def get_throughput(self, n=None, ms=None):
        if n is None: n = self.perf_input
        if ms is None or ms == 0: return 0
        
        # 算子访问了 input (N*4 bytes) 
        # 写入了 output (1*4 bytes, 忽略不计)
        # 总流量 = N * 4 bytes
        total_gb = (n * 4) / 1e9
        seconds = ms / 1000
        return total_gb / seconds  # 单位: GB/s

    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
        assert input.shape == (rows, cols)
        assert output.shape == (cols, rows)
        assert input.dtype == output.dtype
        assert input.device == output.device

        output.copy_(input.transpose(0, 1))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return self.arg_names

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        rows, cols = 2, 3
        input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cuda", dtype=dtype)
        output_tensor = torch.empty(cols, rows, device="cuda", dtype=dtype)
        return {
            "input": input_tensor,
            "output": output_tensor,
            "rows": rows,
            "cols": cols,
            "block_size": 256, 
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        test_specs = [
            # Basic test cases
            ("basic_2x3", 2, 3, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            ("basic_3x1", 3, 1, [[1.0], [2.0], [3.0]]),
            ("square_2x2", 2, 2, [[1.0, 2.0], [3.0, 4.0]]),
            ("single_row", 1, 4, [[1.0, 2.0, 3.0, 4.0]]),
            ("single_column", 4, 1, [[1.0], [2.0], [3.0], [4.0]]),
        ]

        test_cases = []
        for _, r, c, input_vals in test_specs:
            test_cases.append(
                {
                    "input": torch.tensor(input_vals, device="cuda", dtype=dtype),
                    "output": torch.empty(c, r, device="cuda", dtype=dtype),
                    "rows": r,
                    "cols": c,
                    "block_size": 256, 
                }
            )

        # Random test cases with different sizes
        for _, rows, cols in [
            ("small_rectangular", 4, 6),
            ("medium_square", 8, 8),
            ("large_rectangular", 16, 12),
            ("tall_matrix", 32, 8),
            ("wide_matrix", 8, 32),
        ]:
            test_cases.append(
                {
                    "input": torch.empty(rows, cols, device="cuda", dtype=dtype).uniform_(
                        -10.0, 10.0
                    ),
                    "output": torch.empty(cols, rows, device="cuda", dtype=dtype),
                    "rows": rows,
                    "cols": cols,
                    "block_size": 256, 
                }
            )

        # Edge cases
        for _, rows, cols in [
            ("single_element", 1, 1),
            ("max_dimensions", 8192, 8192),
        ]:
            test_cases.append(
                {
                    "input": torch.empty(rows, cols, device="cuda", dtype=dtype).uniform_(
                        -1.0, 1.0
                    ),
                    "output": torch.empty(cols, rows, device="cuda", dtype=dtype),
                    "rows": rows,
                    "cols": cols,
                    "block_size": 256, 
                }
            )

        return test_cases

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        rows, cols = 7000, 6000
        return {
            "input": torch.empty(rows, cols, device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
            "output": torch.zeros(cols, rows, device="cuda", dtype=dtype),
            "rows": rows,
            "cols": cols,
            "block_size": 256, 
        }