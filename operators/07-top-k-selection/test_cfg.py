import torch
from typing import Any, Dict, List
from core.spec import BaseOperatorSpec

class TopKSelectionSpec(BaseOperatorSpec):
    def __init__(self):
        super().__init__()
        # 基础元数据
        self.name = "Top K Selection"
        self.output_name = "output"
        self.atol = 1e-04
        self.rtol = 1e-04
        
        # 测试规模定义 
        self.perf_input = {"N": 50000000, "k": 100}
        self.x_vals = [
            {"N": 1024, "k": 5},          # 基础 Debug
            {"N": 32000, "k": 1},         # Llama 词表级别
            {"N": 128000, "k": 2},        # 标准 MoE 级别
            {"N": 1000000, "k": 100},     # 百万级筛选
            {"N": 10000000, "k": 10}      # 千万级高压测试
        ]

        # CUDA 配置调优 
        cuda_params = [{"block_size": bs} for bs in [32, 128, 256, 512, 1024]]
        self.cuda_tuning_configs = self.make_configs(
            cuda_params, ["float4", "nccl_reduce_topk"]
        )
    
    def is_nccl_version(self, version: str) -> bool:
        return "nccl" in version

    def get_macros(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """将 Python Config 转换为 C++ 宏"""
        macros = {}
        if "block_size" in config:
            macros["BLOCK_SIZE"] = config["block_size"]
        return macros

    def get_bytes_accessed(self, case: Dict[str, Any]) -> int:
        """
        物理指标：内存访问量计算
        主要瓶颈在于读取全量输入 N (float32 = 4 bytes)
        """
        n = case.get("N", self.perf_input["N"])
        # 读：n * 4 bytes
        return n * 4

    def get_total_flops(self, case: Dict[str, Any]) -> int:
        """
        物理指标：计算量估算
        基于插入排序逻辑，近似为 n * k
        """
        n = case.get("N", self.perf_input["N"])
        k = case.get("k", self.perf_input["k"])
        return int(n * k)

    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, N: int, k: int, **kwargs):
        """标准实现"""
        assert input.shape == (N,)
        assert output.shape == (k,)
        assert input.dtype == output.dtype == torch.float32
        assert input.device == output.device
        topk = torch.topk(input, k, largest=True).values
        output.copy_(topk)

    def generate_example_test(self) -> Dict[str, Any]:
        """最小闭环验证"""
        dtype = torch.float32
        input = torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0], device="cuda", dtype=dtype)
        output = torch.empty(3, device="cuda", dtype=dtype)
        return {
            "input": input,
            "output": output,
            "N": 5,
            "k": 3,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        """全量功能回归测试"""
        dtype = torch.float32
        test_cases = []

        # basic_example
        test_cases.append({
            "input": torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0], device="cuda", dtype=dtype),
            "output": torch.empty(3, device="cuda", dtype=dtype),
            "N": 5, "k": 3,
        })
        # negative_numbers
        test_cases.append({
            "input": torch.tensor([-2.0, -1.0, -3.0, -4.0, -5.0, -6.0], device="cuda", dtype=dtype),
            "output": torch.empty(2, device="cuda", dtype=dtype),
            "N": 6, "k": 2,
        })
        # all_equal
        test_cases.append({
            "input": torch.tensor([7.0, 7.0, 7.0, 7.0], device="cuda", dtype=dtype),
            "output": torch.empty(3, device="cuda", dtype=dtype),
            "N": 4, "k": 3,
        })
        # single_element
        test_cases.append({
            "input": torch.tensor([42.0], device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 1, "k": 1,
        })
        # reverse_sorted
        test_cases.append({
            "input": torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0], device="cuda", dtype=dtype),
            "output": torch.empty(2, device="cuda", dtype=dtype),
            "N": 5, "k": 2,
        })
        # large_random
        N, k = 1000, 10
        test_cases.append({
            "input": torch.empty(N, device="cuda", dtype=dtype).uniform_(-1000.0, 1000.0),
            "output": torch.empty(k, device="cuda", dtype=dtype),
            "N": N, "k": k,
        })
        
        return test_cases

    def generate_performance_test(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """压测数据生成"""
        dtype = torch.float32
        N = cfg.get("N", self.perf_input["N"])
        k = cfg.get("k", self.perf_input["k"])
        return {
            "input": torch.empty(N, device="cuda", dtype=dtype).uniform_(-1e6, 1e6),
            "output": torch.empty(k, device="cuda", dtype=dtype),
            "N": N,
            "k": k,
        }