import torch
from typing import Any, Dict, List

class OperatorSpec:
    def __init__(self):
        # 设置算子名字以及输出名字
        self.name = "Matrix Transpose"
        self.output_name = "output" 
        # 设置精度标准
        self.atol = 1e-05
        self.rtol = 1e-05
        # 数据规模测试
        self.x_vals = [
            {"rows": 1024, "cols": 1024},
            {"rows": 2048, "cols": 4096},
            {"rows": 8192, "cols": 8192},
            {"rows": 7000, "cols": 6000},
        ]
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
        # 核函数参数调优
        # 针对triton
        self.tuning_configs = [
            {"BLOCK_ROW": 32, "BLOCK_COL": 32, "num_warps": 2},
            {"BLOCK_ROW": 8, "BLOCK_COL": 8, "num_warps": 2},
            {"BLOCK_ROW": 16, "BLOCK_COL": 16, "num_warps": 2},
        ]
        # 针对cuda 
        self.cuda_tuning_configs = [
            {"bx": 32, "by": 8},   # 线程粗化 4x (A100 的甜点区)
            {"bx": 16, "by": 16},  # 无粗化，1:1 映射
            {"bx": 32, "by": 16},  # 线程粗化 2x
            {"bx": 32, "by": 32},  # 无粗化
        ]
    
    def get_throughput(self, case: Dict[str, Any], ms: float):
        if ms is None or ms == 0: return 0
        
        # 获取维度，如果没传就拿默认值
        rows = case.get("rows", 1)
        cols = case.get("cols", 1)
        
        # 矩阵转置：读一次 (rows*cols)，写一次 (rows*cols)
        # 总流量 = rows * cols * 4 bytes (读) + rows * cols * 4 bytes (写)
        total_bytes = (rows * cols * 4 * 2) 
        total_gb = total_bytes / 1e9
        seconds = ms / 1000
        return total_gb / seconds

    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, rows: int, cols: int, **kwargs):
        assert input.shape == (rows, cols)
        assert output.shape == (cols, rows)
        assert input.dtype == output.dtype
        assert input.device == output.device

        output.copy_(input.transpose(0, 1))

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
            **self.base_cfg 
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
                    **self.base_cfg 
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
                    **self.base_cfg 
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
                    **self.base_cfg 
                }
            )

        return test_cases


    def generate_performance_test(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        dtype = torch.float32
        # 允许外部指定 rows 和 cols，否则用默认大尺寸
        rows = cfg.get("rows", 7000)
        cols = cfg.get("cols", 6000)
        
        return {
            "input": torch.empty(rows, cols, device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
            "output": torch.zeros(cols, rows, device="cuda", dtype=dtype),
            "rows": rows,
            "cols": cols,
            **self.base_cfg 
        }