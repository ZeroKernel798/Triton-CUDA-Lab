import torch
import itertools
from typing import Any, Dict, List
from core.spec import BaseOperatorSpec

class ReductionSpec(BaseOperatorSpec):
    def __init__(self):
        super().__init__()
        # 1. 基础元数据与精度标准
        self.name = "Reduction"
        self.output_name = "output" 
        self.atol = 1e-05
        self.rtol = 1e-05
        
        # 2. 测试规模定义 (1K, 16K, 质数, 1M, 16M)
        self.x_vals = [
            {"N": 2**10}, 
            {"N": 2**14}, 
            {"N": 1000003}, 
            {"N": 2**20}, 
            {"N": 2**24}, 
        ]
        self.perf_input = 2**24

        # 3. Triton 配置生成
        triton_params = []
        for bs, stages, warps in itertools.product(
            [1024, 2048, 4096],  # BLOCK_SIZE
            [2, 3],              # num_stages
            [4, 8, 16]           # num_warps
        ):
            triton_params.append({
                "BLOCK_SIZE": bs,
                "num_stages": stages,
                "num_warps": warps
            })
        self.tuning_configs = self.make_configs(triton_params, ["testv1"])

        # 4. CUDA 配置生成
        cuda_config_basic = [
            {"block_size": 128},   
            {"block_size": 256}, 
            {"block_size": 512},
            {"block_size": 1024},
        ]
        # 统一映射到对应的实现版本标签
        self.cuda_tuning_configs = self.make_configs(cuda_config_basic, ["testv1"])

    # 1. 将 Python Config 转换为 C++ 宏
    def get_macros(self, config: Dict[str, Any]) -> Dict[str, Any]:
        macros = {}
        if "block_size" in config:
            macros["BLOCK_SIZE"] = config["block_size"]
        
        return {k: v for k, v in macros.items() if v is not None}

    # 2. 物理指标计算
    def get_bytes_accessed(self, case: Dict[str, Any]) -> int:
        n = case.get("N", self.perf_input)
        # 访存：读 input (N * 4 bytes)，写标量 output (忽略不计)
        return n * 4

    def get_total_flops(self, case: Dict[str, Any]) -> int:
        n = case.get("N", self.perf_input)
        # 规约求和：N 个数相加需要 N-1 次加法操作
        return n - 1

    # 3. 验证与测试实现
    def reference_impl(self, input: torch.Tensor, **kwargs):
        return torch.sum(input, dtype=torch.float64).to(torch.float32)

    def generate_example_test(self) -> Dict[str, Any]:
        """最小闭环验证"""
        dtype = torch.float32
        N = 8
        return {
            "input": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": N,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        """全量功能回归测试"""
        dtype = torch.float32
        test_cases = []

        # 保持你原有的测试集合：基础、负数、单元素、全0/全1、非2幂次
        test_cases.append({
            "input": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype), "N": 8,
        })
        test_cases.append({
            "input": torch.tensor([-2.5, 1.5, -1.0, 2.0], device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype), "N": 4,
        })
        test_cases.append({
            "input": torch.tensor([42.0], device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype), "N": 1,
        })
        for val, size in [(0.0, 1024), (1.0, 1024)]:
            test_cases.append({
                "input": torch.full((size,), val, device="cuda", dtype=dtype),
                "output": torch.empty(1, device="cuda", dtype=dtype), "N": size,
            })
        test_cases.append({
            "input": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype), "N": 5,
        })
        # 随机大规模测试
        for size in [10000, 15000000]:
            test_cases.append({
                "input": torch.empty(size, device="cuda", dtype=dtype).uniform_(-100.0, 100.0),
                "output": torch.empty(1, device="cuda", dtype=dtype), "N": size,
            })

        return test_cases

    def generate_performance_test(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """压测数据生成"""
        dtype = torch.float32
        N = cfg.get("N", self.perf_input)
        return {
            "input": torch.empty(N, device="cuda", dtype=dtype).uniform_(0.0, 100.0),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": N,
        }