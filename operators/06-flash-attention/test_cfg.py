import torch
from typing import Any, Dict, List

class OperatorSpec:
    def __init__(self):
        self.name = "Softmax Attention"
        self.output_name = "output"  
        # 保留原有的精度阈值
        self.atol = 1e-02 
        self.rtol = 1e-02
        
    def reference_impl(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        output: torch.Tensor,
        M: int,
        N: int,
        d: int,
    ):
        # 保留原有参考实现逻辑
        scale = d**0.5
        attn = torch.matmul(Q, K.t()) / scale
        attn = torch.softmax(attn, dim=1)
        torch.matmul(attn, V, out=output)

    def get_solve_signature(self) -> List[str]:
        """
        对齐 Pybind11 的位置参数传递逻辑。
        顺序需与 C++ 端的 solve(Q, K, V, output, M, N, d) 完全一致。
        """
        return ["Q", "K", "V", "output", "M", "N", "d"]

    def generate_example_test(self) -> Dict[str, Any]:
        # 保留原有数值
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
        output = torch.empty(2, 4, device="cuda", dtype=dtype)
        return {"Q": Q, "K": K, "V": V, "output": output, "M": 2, "N": 3, "d": 4}

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
                "output": torch.empty(2, 4, device="cuda", dtype=dtype),
                "M": 2, "N": 3, "d": 4,
            }
        )

        # zero_matrices
        tests.append(
            {
                "Q": torch.zeros((3, 5), device="cuda", dtype=dtype),
                "K": torch.zeros((3, 5), device="cuda", dtype=dtype),
                "V": torch.zeros((3, 5), device="cuda", dtype=dtype),
                "output": torch.empty(3, 5, device="cuda", dtype=dtype),
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
                "output": torch.empty(4, 3, device="cuda", dtype=dtype),
                "M": 4, "N": 4, "d": 3,
            }
        )

        # large_matrices
        tests.append(
            {
                "Q": torch.empty((64, 32), device="cuda", dtype=dtype).uniform_(-0.1, 0.1),
                "K": torch.empty((128, 32), device="cuda", dtype=dtype).uniform_(-0.1, 0.1),
                "V": torch.empty((128, 32), device="cuda", dtype=dtype).uniform_(-0.1, 0.1),
                "output": torch.empty(64, 32, device="cuda", dtype=dtype),
                "M": 64, "N": 128, "d": 32,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        # 保留原有的跑分规模 512, 256, 128
        dtype = torch.float32
        M, N, d = 512, 256, 128
        Q = torch.empty((512, 128), device="cuda", dtype=dtype).uniform_(-0.1, 0.1)
        K = torch.empty((256, 128), device="cuda", dtype=dtype).uniform_(-0.1, 0.1)
        V = torch.empty((256, 128), device="cuda", dtype=dtype).uniform_(-0.1, 0.1)
        output = torch.empty(M, d, device="cuda", dtype=dtype)
        return {"Q": Q, "K": K, "V": V, "output": output, "M": M, "N": N, "d": d}