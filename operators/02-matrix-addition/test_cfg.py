import torch
from typing import Any, Dict, List

class OperatorSpec:
    @staticmethod
    def make_configs(params: List[Dict[str, Any]], versions: List[str]) -> List[Dict[str, Any]]:
        """
        组合参数池和版本名，生成带 version 标签的配置列表
        """
        return [{**p, "version": v} for v in versions for p in params]
    
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

        # 2. Triton 配置生产
        triton_tiles = [
            {"BLOCK_SIZE": 32, "num_warps": 2},
            {"BLOCK_SIZE": 64, "num_warps": 4},
            {"BLOCK_SIZE": 128, "num_warps": 4},
            {"BLOCK_SIZE": 256, "num_warps": 8},
        ]
        # 直接调用静态方法，生成针对 main 版本的调优列表
        self.tuning_configs = self.make_configs(triton_tiles, ["main"])

        # 3. CUDA 配置生产
        # 分类准备参数池（物料）
        cuda_1d = [{"block_size": bs} for bs in [64, 256, 1024]]
        cuda_2d = [
            {"bx": 32, "by": 8}, 
            {"bx": 16, "by": 16}, 
            {"bx": 8, "by": 32}
        ]

        # 批量打标：让 native 和 float4 共享 1D 参数池
        self.cuda_tuning_configs = self.make_configs(cuda_1d, ["flattened_float4"])
        
        # 2d版本
        self.cuda_tuning_configs.extend(self.make_configs(cuda_2d, ["native", "float4"]))
    
    def get_throughput(self, case: Dict[str, Any], ms = None):
        if ms is None or ms == 0: return 0
        n = case.get("N", self.perf_input)
        total_elements = n * n
        return (total_elements * 4 * 3) / 1e9 / (ms / 1000)

    def get_flops(self, case: Dict[str, Any], ms: float):
        if ms == 0 or ms is None: return 0
        n = case.get("N", self.perf_input)
        
        # 矩阵加法：每个位置执行一次加法，总共 N^2 次操作
        total_ops = float(n * n)
        
        # TFLOPS = total_ops / (秒 * 1e12)
        # ms 转秒需除以 1000，简化公式为：total_ops / (ms * 1e9)
        tflops = total_ops / (ms * 1e9)
        return tflops

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
        })

        # all_zeros_4x4
        test_cases.append({
            "A": torch.zeros((4, 4), device="cuda", dtype=dtype),
            "B": torch.zeros((4, 4), device="cuda", dtype=dtype),
            "C": torch.zeros((4, 4), device="cuda", dtype=dtype),
            "N": 4,
        })

        # identity_plus_identity_3x3
        test_cases.append({
            "A": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device="cuda", dtype=dtype),
            "B": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device="cuda", dtype=dtype),
            "C": torch.zeros((3, 3), device="cuda", dtype=dtype),
            "N": 3,
        })

        # negative_values_2x2
        test_cases.append({
            "A": torch.tensor([[-1.0, -2.0], [-3.0, -4.0]], device="cuda", dtype=dtype),
            "B": torch.tensor([[-5.0, -6.0], [-7.0, -8.0]], device="cuda", dtype=dtype),
            "C": torch.zeros((2, 2), device="cuda", dtype=dtype),
            "N": 2,
        })

        # mixed_positive_negative_2x2
        test_cases.append({
            "A": torch.tensor([[1.0, -2.0], [-3.0, 4.0]], device="cuda", dtype=dtype),
            "B": torch.tensor([[-1.0, 2.0], [3.0, -4.0]], device="cuda", dtype=dtype),
            "C": torch.zeros((2, 2), device="cuda", dtype=dtype),
            "N": 2,
        })

        # single_element_1x1
        test_cases.append({
            "A": torch.tensor([[42.0]], device="cuda", dtype=dtype),
            "B": torch.tensor([[8.0]], device="cuda", dtype=dtype),
            "C": torch.zeros((1, 1), device="cuda", dtype=dtype),
            "N": 1,
        })

        # large_N_16x16
        test_cases.append({
            "A": torch.empty((16, 16), device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
            "B": torch.empty((16, 16), device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
            "C": torch.zeros((16, 16), device="cuda", dtype=dtype),
            "N": 16,
        })

        # very_small_numbers
        test_cases.append({
            "A": torch.tensor([[0.000001, 0.0000001], [0.00000001, 0.000000001]], device="cuda", dtype=dtype),
            "B": torch.tensor([[0.000001, 0.0000001], [0.00000001, 0.000000001]], device="cuda", dtype=dtype),
            "C": torch.zeros((2, 2), device="cuda", dtype=dtype),
            "N": 2,
        })

        # large_numbers
        test_cases.append({
            "A": torch.tensor([[1000000.0, 10000000.0], [-1000000.0, -10000000.0]], device="cuda", dtype=dtype),
            "B": torch.tensor([[1000000.0, -10000000.0], [-1000000.0, 10000000.0]], device="cuda", dtype=dtype),
            "C": torch.zeros((2, 2), device="cuda", dtype=dtype),
            "N": 2,
        })

        # non_power_of_two_size_7x7
        test_cases.append({
            "A": torch.empty((7, 7), device="cuda", dtype=dtype).uniform_(-5.0, 5.0),
            "B": torch.empty((7, 7), device="cuda", dtype=dtype).uniform_(-5.0, 5.0),
            "C": torch.zeros((7, 7), device="cuda", dtype=dtype),
            "N": 7,
        })

        # medium_size_32x32
        test_cases.append({
            "A": torch.empty((32, 32), device="cuda", dtype=dtype).uniform_(-100.0, 100.0),
            "B": torch.empty((32, 32), device="cuda", dtype=dtype).uniform_(-100.0, 100.0),
            "C": torch.zeros((32, 32), device="cuda", dtype=dtype),
            "N": 32,
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
        }