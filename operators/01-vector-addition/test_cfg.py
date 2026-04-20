import torch
from typing import Any, Dict, List
from core.spec import BaseOperatorSpec

class VectorAdditionFP32Spec(BaseOperatorSpec):
    def __init__(self):
        super().__init__()
        # 基础元数据
        self.name = "Vector Addition FP32"
        self.op_name = "VectorAddition_FP32" # 对应 Executor 索引
        self.output_name = "C"
        self.atol, self.rtol = 1e-05, 1e-05
        self.dtype = torch.float32
        
        # 测试规模定义 (从 4K 元素到 16M 元素)
        self.perf_input = 1024 * 1024 * 16
        self.x_vals = [{"N": 2**i} for i in range(12, 25)]

        # Triton 配置 (BLOCK_SIZE, num_warps)
        triton_tiles = [
            {"BLOCK_SIZE": 32, "num_warps": 2},
            {"BLOCK_SIZE": 128, "num_warps": 4},
            {"BLOCK_SIZE": 256, "num_warps": 8},
            {"BLOCK_SIZE": 1024, "num_warps": 8},
        ]
        self.tuning_configs = self.make_configs(triton_tiles, ["main"])

        # CUDA 配置 (native, float4)
        cuda_params = [{"bs": 64}, {"bs": 256}, {"bs": 512}, {"bs": 1024}]
        self.cuda_tuning_configs = self.make_configs(cuda_params, ["native", "float4"])

    # --- 核心接口实现 ---

    def get_macros(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """将 Python 配置转换为 C++ 宏"""
        # 统一映射到 BLOCK_SIZE 宏
        return {"BLOCK_SIZE": config.get("bs", 256)}

    def get_bytes_accessed(self, case: Dict[str, Any]) -> int:
        """物理指标：读 A + 读 B + 写 C"""
        n = case.get("N", self.perf_input)
        return n * 4 * 3 # float32 为 4 字节

    def get_total_flops(self, case: Dict[str, Any]) -> int:
        """物理指标：加法次数"""
        return case.get("N", self.perf_input)

    def reference_impl(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int, **kwargs):
        """官方标杆"""
        assert A.shape == B.shape == C.shape == (N,)
        assert A.dtype == B.dtype == C.dtype == self.dtype
        torch.add(A, B, out=C)

    # --- 数据生成逻辑 ---

    def generate_example_test(self) -> Dict[str, Any]:
        """最小闭环验证"""
        N = 4
        return {
            "A": torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda", dtype=self.dtype),
            "B": torch.tensor([5.0, 6.0, 7.0, 8.0], device="cuda", dtype=self.dtype),
            "C": torch.empty(N, device="cuda", dtype=self.dtype),
            "N": N,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        """全量功能测试集"""
        test_cases = []
        # 边界情况：非 2 的幂次、零值、负数
        for n in [1, 7, 31, 1024, 4097]:
            test_cases.append({
                "A": torch.randn(n, device="cuda", dtype=self.dtype),
                "B": torch.randn(n, device="cuda", dtype=self.dtype),
                "C": torch.zeros(n, device="cuda", dtype=self.dtype),
                "N": n,
            })
        return test_cases

    def generate_performance_test(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """压测数据生成"""
        N = cfg.get("N", self.perf_input)
        return {
            "A": torch.empty(N, device="cuda", dtype=self.dtype).uniform_(-10, 10),
            "B": torch.empty(N, device="cuda", dtype=self.dtype).uniform_(-10, 10),
            "C": torch.zeros(N, device="cuda", dtype=self.dtype),
            "N": N,
        }