import torch
from typing import Any, Dict, List
from core.spec import BaseOperatorSpec

class MatrixTransposeSpec(BaseOperatorSpec):
    def __init__(self):
        super().__init__()
        # 基础元数据
        self.name = "Matrix Transpose"
        self.output_name = "output"
        self.atol = 1e-05
        self.rtol = 1e-05
        
        # 测试规模定义 
        self.perf_input = {"rows": 7000, "cols": 6000}
        self.x_vals = [
            {"rows": 1024, "cols": 1024},
            {"rows": 2048, "cols": 4096},
            {"rows": 4096, "cols": 2048},
            {"rows": 8192, "cols": 8192},
            {"rows": 8192, "cols": 2048},
            {"rows": 7000, "cols": 6000},
            {"rows": 16384, "cols": 16384},
        ]

        # Triton 配置
        triton_tiles = [
            {"BLOCK_ROW": 32, "BLOCK_COL": 32, "num_warps": 2},
            {"BLOCK_ROW": 8, "BLOCK_COL": 8, "num_warps": 2},
            {"BLOCK_ROW": 16, "BLOCK_COL": 16, "num_warps": 2},
        ]
        self.tuning_configs = self.make_configs(triton_tiles, ["native"])

        # CUDA 配置
        cuda_tiles = [
            {"bx": 32, "by": 8},   # 线程粗化 4x
            {"bx": 16, "by": 16},  # 无粗化
            {"bx": 32, "by": 16},  # 线程粗化 2x
            {"bx": 32, "by": 32},  # 无粗化
        ]
        self.cuda_tuning_configs = self.make_configs(
            cuda_tiles, ["shared_mm_ILP"]
        )

        cuda_base_tiles = [
            {"bx":32, "by":32},
            {"bx":16, "by":16},
            {"bx":8, "by":8},
        ]
        self.cuda_tuning_configs.extend(self.make_configs(cuda_base_tiles, ["native"]))

        cuda_smem_tiles = [
            {"bx":32, "by":32},
            {"bx":32, "by":16},
            {"bx":32, "by":8},
            {"bx":16, "by":16},
            {"bx":8, "by":8},
        ]
        self.cuda_tuning_configs.extend(self.make_configs(cuda_smem_tiles, ["shared_mm"]))


    def get_macros(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """将 Python Config 转换为 C++ 宏"""
        version = config.get("version")
        macros = {}
        
        # 统一映射逻辑
        if "BLOCK_ROW" in config:
            macros["BLOCK_ROW"] = config["BLOCK_ROW"]
            macros["BLOCK_COL"] = config["BLOCK_COL"]
        
        if "bx" in config:
            macros["BLOCK_X"] = config["bx"]
            macros["BLOCK_Y"] = config["by"]
            
        return macros

    def get_bytes_accessed(self, case: Dict[str, Any]) -> int:
        """物理指标：内存访问量计算"""
        rows = case.get("rows", self.perf_input["rows"])
        cols = case.get("cols", self.perf_input["cols"])
        # 读一次 (rows*cols*4) + 写一次 (rows*cols*4)
        return rows * cols * 4 * 2

    def get_total_flops(self, case: Dict[str, Any]) -> int:
        """物理指标：转置无数学运算，FLOPs 为 0"""
        return 0

    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, rows: int, cols: int, **kwargs):
        """标准实现"""
        assert input.shape == (rows, cols)
        assert output.shape == (cols, rows)
        assert input.dtype == output.dtype
        assert input.device == output.device
        output.copy_(input.transpose(0, 1))

    def generate_example_test(self) -> Dict[str, Any]:
        """最小闭环验证"""
        dtype = torch.float32
        rows, cols = 2, 3
        return {
            "input": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cuda", dtype=dtype),
            "output": torch.empty(cols, rows, device="cuda", dtype=dtype),
            "rows": rows,
            "cols": cols,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        """全量功能回归测试"""
        dtype = torch.float32
        test_cases = []
        
        # 1. 基础硬编码用例
        specs = [
            (2, 3, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            (3, 1, [[1.0], [2.0], [3.0]]),
            (2, 2, [[1.0, 2.0], [3.0, 4.0]]),
            (1, 4, [[1.0, 2.0, 3.0, 4.0]]),
        ]
        for r, c, vals in specs:
            test_cases.append({
                "input": torch.tensor(vals, device="cuda", dtype=dtype),
                "output": torch.empty(c, r, device="cuda", dtype=dtype),
                "rows": r, "cols": c,
            })

        # 2. 随机尺寸用例 (保持原逻辑)
        for r, c in [(4, 6), (8, 8), (32, 8), (8, 32)]:
            test_cases.append({
                "input": torch.empty(r, c, device="cuda", dtype=dtype).uniform_(-10, 10),
                "output": torch.empty(c, r, device="cuda", dtype=dtype),
                "rows": r, "cols": c,
            })

        # 3. 边界情况
        test_cases.append({
            "input": torch.empty(1, 1, device="cuda", dtype=dtype).uniform_(-1, 1),
            "output": torch.empty(1, 1, device="cuda", dtype=dtype),
            "rows": 1, "cols": 1,
        })
        
        return test_cases

    def generate_performance_test(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """压测数据生成"""
        dtype = torch.float32
        rows = cfg.get("rows", self.perf_input["rows"])
        cols = cfg.get("cols", self.perf_input["cols"])
        
        return {
            "input": torch.empty(rows, cols, device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
            "output": torch.zeros(cols, rows, device="cuda", dtype=dtype),
            "rows": rows,
            "cols": cols,
        }