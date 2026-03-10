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
        # 基础信息设置
        self.name = "Vector Addition"
        self.output_name = "C"
        self.atol, self.rtol = 1e-05, 1e-05
        self.perf_input = 1024 * 1024 * 16
        self.x_vals = [{"N": 2**i} for i in range(12, 25)]

        # CUDA 参数生产线
        cuda_1d_params = [{"block_size": bs} for bs in [32, 128, 256, 512, 1024]]
        
        # 直接批量生成：native 和 float4 共享这套参数
        self.cuda_tuning_configs = self.make_configs(cuda_1d_params, ["native", "float4"])

        # Triton 参数生产线
        triton_params = [
            {"BLOCK_SIZE": 32, "num_warps": 2},
            {"BLOCK_SIZE": 64, "num_warps": 4},
            {"BLOCK_SIZE": 128, "num_warps": 4},
            {"BLOCK_SIZE": 256, "num_warps": 8},
        ]
        
        # 批量生成：Triton 文件名如果是 Triton_main.py，这里就填 main
        self.tuning_configs = self.make_configs(triton_params, ["main"])
    

    def get_throughput(self, case: Dict[str, Any], ms = None):
        # 向量加法吞吐量计算
        if ms is None or ms == 0: return 0
        n = case.get("N", self.perf_input)
        return (n * 4 * 3) / 1e9 / (ms / 1000)
    
    def get_flops(self, case: Dict[str, Any], ms: float):
        if ms == 0 or ms is None: return 0
        n = case.get("N", self.perf_input)
        
        # 向量加法中，每个元素仅执行 1 次加法操作
        total_ops = float(n)
        
        # TFLOPS = Total Ops / (Time in seconds * 1e12)
        # 由于 ms 是毫秒，转为秒需要 / 1000
        # 简化后公式：tflops = total_ops / (ms * 1e9)
        tflops = total_ops / (ms * 1e9)
        return tflops

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

    def generate_performance_test(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        dtype = torch.float32
        N = cfg.get("N", self.perf_input)
        return {
            "A": torch.empty(N, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
            "B": torch.empty(N, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
            "C": torch.zeros(N, device="cuda", dtype=dtype),
            "N": N,
        }