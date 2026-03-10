import torch
from typing import Any, Dict, List
import itertools

class OperatorSpec:
    @staticmethod
    def make_configs(params: List[Dict[str, Any]], versions: List[str]) -> List[Dict[str, Any]]:
        """
        组合参数池和版本名，生成带 version 标签的配置列表
        """
        return [{**p, "version": v} for v in versions for p in params]
    
    def __init__(self):
        self.name = "Flash Attention"
        self.output_name = "O"  
        self.atol = 1e-02 
        self.rtol = 1e-02

        self.x_vals = [
            {"M": 128, "N": 128, "d": 64},   # 基础 Debug
            {"M": 512, "N": 512, "d": 64},   # 标准尺寸
            {"M": 1024, "N": 1024, "d": 128}, # 常用尺寸
            {"M": 2048, "N": 2048, "d": 128}, # 长文本
            {"M": 4096, "N": 4096, "d": 64},  # 极限测试
        ]

        cuda_params = []
        for br, bc in itertools.product(
            [16, 32], # Br: Block Row (Q 分块)
            [16, 32]  # Bc: Block Col (KV 分块)
        ):
            cuda_params.append({
                "Br": br,
                "Bc": bc
            })
        
        self.cuda_tuning_configs = self.make_configs(cuda_params, 
                            ["flashattentionv1", "flashattentionv2_opt"])

    def get_throughput(self, case: Dict[str, Any], ms: float):
        """
        计算吞吐量 (GB/s)
        对于 Flash Attention，理论最小访存量为：读 Q + 读 K + 读 V + 写 Output
        """
        if ms == 0: return 0
        M, N, d = case["M"], case["N"], case["d"]
        
        # 假设使用的是 float32 (4 bytes)
        # 访存总量 = (Q字节数 + K字节数 + V字节数 + Output字节数)
        # Q: M * d, K: N * d, V: N * d, O: M * d
        total_bytes = (M * d + N * d + N * d + M * d) * 4
        
        # 转换为 GB/s: (Bytes / 1e9) / (ms / 1000)
        return (total_bytes / 1e9) / (ms / 1000)

    def get_flops(self, case: Dict[str, Any], ms: float):
        """计算 TFLOPS: Attention 的计算量约为 2 * M * N * (d + d)"""
        if ms == 0: return 0
        m, n, d = case["M"], case["N"], case["d"]
        # 总计约 4 * M * N * d
        total_ops = 4.0 * m * n * d
        return total_ops / (ms * 1e9)
        
    def reference_impl(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        O: torch.Tensor,
        M: int,
        N: int,
        d: int,
        **kwargs
    ):
        # 保留原有参考实现逻辑
        scale = d**0.5
        attn = torch.matmul(Q, K.t()) / scale
        attn = torch.softmax(attn, dim=1)
        torch.matmul(attn, V, out=O)

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        Q = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], device="cuda", dtype=dtype)
        K = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            device="cuda",
            dtype=dtype,
        )
        V = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
            device="cuda",
            dtype=dtype,
        )
        O = torch.empty(2, 4, device="cuda", dtype=dtype)
        return {"Q": Q, "K": K, "V": V, "O": O, "M": 2, "N": 3, "d": 4}

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        # 保留所有功能测试用例数值
        dtype = torch.float32
        tests = []

        # basic_example
        tests.append(
            {
                "Q": torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], device="cuda", dtype=dtype
                ),
                "K": torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                    device="cuda",
                    dtype=dtype,
                ),
                "V": torch.tensor(
                    [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
                    device="cuda",
                    dtype=dtype,
                ),
                "O": torch.empty(2, 4, device="cuda", dtype=dtype),
                "M": 2, "N": 3, "d": 4,
            }
        )

        # zero_matrices
        tests.append(
            {
                "Q": torch.zeros((3, 5), device="cuda", dtype=dtype),
                "K": torch.zeros((3, 5), device="cuda", dtype=dtype),
                "V": torch.zeros((3, 5), device="cuda", dtype=dtype),
                "O": torch.empty(3, 5, device="cuda", dtype=dtype),
                "M": 3, "N": 3, "d": 5,
            }
        )

        # mixed_values
        tests.append(
            {
                "Q": torch.tensor(
                    [[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0], [-7.0, 8.0, -9.0], [10.0, -11.0, 12.0]],
                    device="cuda", dtype=dtype,
                ),
                "K": torch.tensor(
                    [[2.0, -1.0, 3.0], [-4.0, 5.0, -6.0], [7.0, -8.0, 9.0], [-10.0, 11.0, -12.0]],
                    device="cuda", dtype=dtype,
                ),
                "V": torch.tensor(
                    [[1.0, 0.5, -0.5], [-1.0, 2.0, 3.0], [4.0, -2.0, 1.0], [0.0, 1.0, -1.0]],
                    device="cuda", dtype=dtype,
                ),
                "O": torch.empty(4, 3, device="cuda", dtype=dtype),
                "M": 4, "N": 4, "d": 3,
            }
        )

        # large_matrices
        tests.append(
            {
                "Q": torch.empty((64, 32), device="cuda", dtype=dtype).uniform_(-0.1, 0.1),
                "K": torch.empty((128, 32), device="cuda", dtype=dtype).uniform_(-0.1, 0.1),
                "V": torch.empty((128, 32), device="cuda", dtype=dtype).uniform_(-0.1, 0.1),
                "O": torch.empty(64, 32, device="cuda", dtype=dtype),
                "M": 64, "N": 128, "d": 32,
            }
        )

        return tests

    def generate_performance_test(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        # 保留原有的跑分规模 512, 256, 128
        dtype = torch.float32
        M = cfg.get("M", 512)
        N = cfg.get("N", 256)
        d = cfg.get("d", 128)

        Q = torch.empty((M, d), device="cuda", dtype=dtype).uniform_(-0.1, 0.1)
        K = torch.empty((N, d), device="cuda", dtype=dtype).uniform_(-0.1, 0.1)
        V = torch.empty((N, d), device="cuda", dtype=dtype).uniform_(-0.1, 0.1)
        O = torch.empty(M, d, device="cuda", dtype=dtype)
        return {"Q": Q, "K": K, "V": V, "O": O, "M": M, "N": N, "d": d}