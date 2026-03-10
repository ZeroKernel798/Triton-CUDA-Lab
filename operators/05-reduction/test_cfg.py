import torch
import itertools
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
        self.name = "Reduction"
        self.output_name = "output" 
        # 设置精度标准
        self.atol = 1e-05
        self.rtol = 1e-05
        
        # 数据规模测试 (从 1K 到 16M)
        self.x_vals = [
            {"N": 2**10}, # 1K
            {"N": 2**14}, # 16K
            {"N": 1000003}, # 非 2 的幂次（质数）
            {"N": 2**20}, # 1M
            {"N": 2**24}, # 16M
        ]
        self.perf_input = 2**24

        # 核函数参数调优
        # 针对 Triton 的配置生产线
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

        # 针对 CUDA 的配置生产线
        cuda_config_basic = [
            {"block_size": 128},   
            {"block_size": 256}, 
            {"block_size": 512},
            {"block_size": 1024},
        ]
        # 将参数池分配给不同的实现版本
        self.cuda_tuning_configs = self.make_configs(cuda_config_basic, 
                                    ["testv1"])

    def get_throughput(self, case: Dict[str, Any], ms: float):
        if ms == 0: return 0
        n = case["N"]
        # 访存量：读取整个向量 (N * 4 bytes)，写入一个标量 (忽略不计)
        total_bytes = n * 4
        return (total_bytes / 1e9) / (ms / 1000)

    def get_flops(self, case: Dict[str, Any], ms: float):
        """计算 TFLOPS (Tera Floating-point Operations Per Second)"""
        if ms == 0: return 0
        n = case["N"]
        # 规约求和的操作数：N-1 次加法
        total_ops = float(n - 1)
        # ms 转秒需 / 1000，TFLOPS 需 / 1e12，合并为 / 1e9
        tflops = total_ops / (ms * 1e9) 
        return tflops

    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, N: int, **kwargs):
        assert input.shape == (N,)
        assert output.shape == (1,)
        assert input.dtype == output.dtype
        assert input.device == output.device
        # 使用 double 进行中间累加保证参考实现精度，再转回 float
        output[0] = torch.sum(input.double()).float()

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 8
        input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], device="cuda", dtype=dtype)
        output_tensor = torch.empty(1, device="cuda", dtype=dtype)
        return {
            "input": input_tensor,
            "output": output_tensor,
            "N": N,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        test_cases = []

        # 1. 基础案例
        test_cases.append({
            "input": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 8,
        })

        # 2. 负数案例
        test_cases.append({
            "input": torch.tensor([-2.5, 1.5, -1.0, 2.0], device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 4,
        })

        # 3. 边界：单元素
        test_cases.append({
            "input": torch.tensor([42.0], device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 1,
        })

        # 4. 特殊数值：全 0 和全 1
        for val, size in [(0.0, 1024), (1.0, 1024)]:
            test_cases.append({
                "input": torch.full((size,), val, device="cuda", dtype=dtype),
                "output": torch.empty(1, device="cuda", dtype=dtype),
                "N": size,
            })

        # 5. 非 2 的幂次规模
        test_cases.append({
            "input": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 5,
        })

        # 6. 大规模随机测试
        for size in [10000, 15000000]:
            test_cases.append({
                "input": torch.empty(size, device="cuda", dtype=dtype).uniform_(-100.0, 100.0),
                "output": torch.empty(1, device="cuda", dtype=dtype),
                "N": size,
            })

        return test_cases

    def generate_performance_test(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        dtype = torch.float32
        # 允许从 Scaling 模式的 cfg 中提取 N
        N = cfg.get("N", self.perf_input)
        return {
            "input": torch.empty(N, device="cuda", dtype=dtype).uniform_(0.0, 100.0),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": N,
        }