import torch
from typing import Any, Dict, List
from core.spec import BaseOperatorSpec

class MatrixAdditionSpec(BaseOperatorSpec):
    def __init__(self):
        super().__init__()
        # 基础元数据
        self.name = "Matrix Addition"
        self.output_name = "C"
        self.atol = 1e-05
        self.rtol = 1e-05
        
        # 测试规模定义
        self.perf_input = 8192
        self.x_vals = [{"N": 2**i} for i in range(8, 14)]

        # 1. Triton 配置
        triton_tiles = [
            {"BLOCK_SIZE": 32, "num_warps": 2},
            {"BLOCK_SIZE": 128, "num_warps": 4},
            {"BLOCK_SIZE": 256, "num_warps": 8},
        ]
        self.tuning_configs = self.make_configs(triton_tiles, ["main"])

        # 2. CUDA 配置 (1D vs 2D)
        cuda_1d = [{"bs": 64}, {"bs": 256}, {"bs": 1024}]
        cuda_2d = [
            {"bx": 32, "by": 8}, 
            {"bx": 16, "by": 16}, 
            {"bx": 8, "by": 32}
        ]
        
        # 这里的 version 字符串必须和文件名或实现版本对应
        self.cuda_tuning_configs = self.make_configs(cuda_1d, ["flattened_float4"])
        self.cuda_tuning_configs.extend(self.make_configs(cuda_2d, ["native", "float4"]))

    # 1. 将 Python Config 转换为 C++ 宏
    def get_macros(self, config: Dict[str, Any]) -> Dict[str, Any]:
        version = config.get("version")
        macros = {}
        
        if version == "flattened_float4":
            # 映射一维参数
            macros["BLOCK_SIZE"] = config["bs"]
        else:
            # 映射二维参数 
            macros["BLOCK_X"] = config.get("bx", 16)
            macros["BLOCK_Y"] = config.get("by", 16)
            
        return macros

    # 2. 物理指标计算
    def get_bytes_accessed(self, case: Dict[str, Any]) -> int:
        n = case.get("N", self.perf_input)
        # 读 A, B + 写 C (float32 = 4 bytes)
        return (n * n) * 4 * 3

    def get_total_flops(self, case: Dict[str, Any]) -> int:
        n = case.get("N", self.perf_input)
        # 矩阵加法，每个位置 1 次加法
        return n * n

    # 3. 验证与测试
    def reference_impl(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int, **kwargs):
        """标准实现"""
        assert A.shape == (N, N)
        assert B.shape == (N, N)
        assert C.shape == (N, N)
        assert A.dtype == B.dtype == C.dtype
        assert A.device == B.device == C.device
        torch.add(A, B, out=C)

    def generate_example_test(self) -> Dict[str, Any]:
        """最小闭环验证"""
        N = 2
        return {
            "A": torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda"),
            "B": torch.tensor([[5.0, 6.0], [7.0, 8.0]], device="cuda"),
            "C": torch.empty((N, N), device="cuda"),
            "N": N,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        """全量功能回归测试"""
        dtype = torch.float32
        test_cases = []

        # basic_2x2
        test_cases.append({
            "A": torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=dtype),
            "B": torch.tensor([[5.0, 6.0], [7.0, 8.0]], device="cuda", dtype=dtype),
            "C": torch.zeros((2, 2), device="cuda", dtype=dtype), "N": 2,
        })
        # all_zeros_4x4
        test_cases.append({
            "A": torch.zeros((4, 4), device="cuda", dtype=dtype),
            "B": torch.zeros((4, 4), device="cuda", dtype=dtype),
            "C": torch.zeros((4, 4), device="cuda", dtype=dtype), "N": 4,
        })
        # identity_plus_identity_3x3
        test_cases.append({
            "A": torch.eye(3, device="cuda", dtype=dtype),
            "B": torch.eye(3, device="cuda", dtype=dtype),
            "C": torch.zeros((3, 3), device="cuda", dtype=dtype), "N": 3,
        })
        # negative_values
        test_cases.append({
            "A": torch.tensor([[-1.0, -2.0], [-3.0, -4.0]], device="cuda", dtype=dtype),
            "B": torch.tensor([[-5.0, -6.0], [-7.0, -8.0]], device="cuda", dtype=dtype),
            "C": torch.zeros((2, 2), device="cuda", dtype=dtype), "N": 2,
        })
        # single_element_1x1
        test_cases.append({
            "A": torch.tensor([[42.0]], device="cuda", dtype=dtype),
            "B": torch.tensor([[8.0]], device="cuda", dtype=dtype),
            "C": torch.zeros((1, 1), device="cuda", dtype=dtype), "N": 1,
        })
        # large_N_16x16
        test_cases.append({
            "A": torch.empty((16, 16), device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
            "B": torch.empty((16, 16), device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
            "C": torch.zeros((16, 16), device="cuda", dtype=dtype), "N": 16,
        })
        # non_power_of_two_size_7x7
        test_cases.append({
            "A": torch.empty((7, 7), device="cuda", dtype=dtype).uniform_(-5.0, 5.0),
            "B": torch.empty((7, 7), device="cuda", dtype=dtype).uniform_(-5.0, 5.0),
            "C": torch.zeros((7, 7), device="cuda", dtype=dtype), "N": 7,
        })
        # medium_size_32x32
        test_cases.append({
            "A": torch.empty((32, 32), device="cuda", dtype=dtype).uniform_(-100.0, 100.0),
            "B": torch.empty((32, 32), device="cuda", dtype=dtype).uniform_(-100.0, 100.0),
            "C": torch.zeros((32, 32), device="cuda", dtype=dtype), "N": 32,
        })

        return test_cases

    def generate_performance_test(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """压测数据生成"""
        N = cfg.get("N", self.perf_input)
        return {
            "A": torch.empty((N, N), device="cuda").uniform_(-1000, 1000),
            "B": torch.empty((N, N), device="cuda").uniform_(-1000, 1000),
            "C": torch.zeros((N, N), device="cuda"),
            "N": N,
        }