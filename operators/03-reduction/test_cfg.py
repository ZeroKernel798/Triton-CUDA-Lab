import torch
from typing import Any, Dict, List

class OperatorSpec:
    def __init__(self):
        # 设置算子名字以及输出名字
        self.name = "Reduction"
        self.output_name = "output" 
        # 设置参数列表
        self.arg_names = ["input", "output", "N", "block_size"]
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

    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, N: int,  **kwargs):
        # 保留所有逻辑和断言
        assert input.shape == (N,)
        assert output.shape == (1,)
        assert input.dtype == output.dtype
        assert input.device == output.device
        # 使用 double 进行中间累加保证参考实现精度
        output[0] = torch.sum(input.double()).float()

    def get_solve_signature(self) -> List[str]:
        return self.arg_names

    def generate_example_test(self) -> Dict[str, Any]:
        # 保留原有数值 N=8
        dtype = torch.float32
        input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], device="cuda", dtype=dtype)
        output = torch.empty(1, device="cuda", dtype=dtype)
        N = 8
        return {"input": input, "output": output, "N": N, "block_size": 256,}

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        # 保留所有原有的测试用例和数值
        dtype = torch.float32
        tests = []
        # basic_example
        tests.append({
            "input": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 8,
            "block_size": 256, 
        })
        # negative_numbers
        tests.append({
            "input": torch.tensor([-2.5, 1.5, -1.0, 2.0], device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 4,
            "block_size": 256, 
        })
        # single_element
        tests.append({
            "input": torch.tensor([42.0], device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 1,
            "block_size": 256, 
        })
        # all_zeros
        tests.append({
            "input": torch.zeros(1024, device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 1024,
            "block_size": 256, 
        })
        # all_ones
        tests.append({
            "input": torch.ones(1024, device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 1024,
            "block_size": 256, 
        })
        # non_power_of_two
        tests.append({
            "input": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 5,
        })
        # large_random
        tests.append({
            "input": torch.empty(10000, device="cuda", dtype=dtype).uniform_(-1000.0, 1000.0),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 10000,
            "block_size": 256, 
        })
        # large_random_2
        tests.append({
            "input": torch.empty(15000000, device="cuda", dtype=dtype).uniform_(0.0, 1000.0),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 15000000,
            "block_size": 256, 
        })
        return tests

    def generate_performance_test(self, N = None) -> Dict[str, Any]:
        if N is None:
            N = self.perf_input
        dtype = torch.float32
        input = torch.empty(N, device="cuda", dtype=dtype).uniform_(0.0, 1000.0)
        output = torch.empty(1, device="cuda", dtype=dtype)
        return {"input": input, 
                "output": output, 
                "N": N,
                "block_size": 256, # 默认值，会被 tuning 模式覆盖
                "BLOCK_SIZE": 256, # Triton 默认值
                "num_warps": 4     # Triton 默认值
                }