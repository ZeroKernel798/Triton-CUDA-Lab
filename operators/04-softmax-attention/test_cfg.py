# import torch
# import ctypes
# from typing import Any, Dict, List

# class OperatorSpec:
#     def __init__(self):
#         self.name = "Flash Attention"
#         self.atol = 1e-05
#         self.rtol = 1e-05
#         self.dtype = torch.float32

#     def reference(
#         self,
#         Q: torch.Tensor,
#         K: torch.Tensor,
#         V: torch.Tensor,
#         output: torch.Tensor,
#         M: int,
#         N: int,
#         d: int,
#     ):
#         scale = d**0.5
#         attn = torch.matmul(Q, K.t()) / scale
#         attn = torch.softmax(attn, dim=1)
#         torch.matmul(attn, V, out=output)

#     def get_cuda_args(self, test_case: Dict[str, Any]):
#         """
#         适配 run_lab.py 的 CUDA 参数传递
#         test_case 是从下面的 generate_xxx 拿到的字典
#         """
#         A, B, C, N = test_case["A"], test_case["B"], test_case["C"], test_case["N"]
#         return [A, B, C, N], [None, None, None, ctypes.c_size_t]

#     def get_triton_args(self, test_case: Dict[str, Any]):
#         # 算出一共有多少个 float32 元素
#         total_elements = test_case["N"] * test_case["N"] 
#         return {
#             "a_ptr": test_case["A"],
#             "b_ptr": test_case["B"],
#             "c_ptr": test_case["C"],
#             "n_elements": total_elements, # 传总数给 Triton 的 mask
#             "BLOCK_SIZE": 1024
#         }

#     def get_solve_signature(self) -> Dict[str, tuple]:
#         return {
#             "Q": (ctypes.POINTER(ctypes.c_float), "in"),
#             "K": (ctypes.POINTER(ctypes.c_float), "in"),
#             "V": (ctypes.POINTER(ctypes.c_float), "in"),
#             "output": (ctypes.POINTER(ctypes.c_float), "out"),
#             "M": (ctypes.c_int, "in"),
#             "N": (ctypes.c_int, "in"),
#             "d": (ctypes.c_int, "in"),
#         }

#     def generate_example_test(self) -> Dict[str, Any]:
#         dtype = torch.float32
#         Q = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], device="cuda", dtype=dtype)
#         K = torch.tensor(
#             [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
#             device="cuda",
#             dtype=dtype,
#         )
#         V = torch.tensor(
#             [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
#             device="cuda",
#             dtype=dtype,
#         )
#         output = torch.empty(2, 4, device="cuda", dtype=dtype)
#         return {"Q": Q, "K": K, "V": V, "output": output, "M": 2, "N": 3, "d": 4}

#     def generate_functional_test(self) -> List[Dict[str, Any]]:
#         dtype = torch.float32
#         tests = []

#         # basic_example
#         tests.append(
#             {
#                 "Q": torch.tensor(
#                     [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], device="cuda", dtype=dtype
#                 ),
#                 "K": torch.tensor(
#                     [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
#                     device="cuda",
#                     dtype=dtype,
#                 ),
#                 "V": torch.tensor(
#                     [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
#                     device="cuda",
#                     dtype=dtype,
#                 ),
#                 "output": torch.empty(2, 4, device="cuda", dtype=dtype),
#                 "M": 2,
#                 "N": 3,
#                 "d": 4,
#             }
#         )

#         # zero_matrices
#         tests.append(
#             {
#                 "Q": torch.zeros((3, 5), device="cuda", dtype=dtype),
#                 "K": torch.zeros((3, 5), device="cuda", dtype=dtype),
#                 "V": torch.zeros((3, 5), device="cuda", dtype=dtype),
#                 "output": torch.empty(3, 5, device="cuda", dtype=dtype),
#                 "M": 3,
#                 "N": 3,
#                 "d": 5,
#             }
#         )

#         # mixed_values
#         tests.append(
#             {
#                 "Q": torch.tensor(
#                     [[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0], [-7.0, 8.0, -9.0], [10.0, -11.0, 12.0]],
#                     device="cuda",
#                     dtype=dtype,
#                 ),
#                 "K": torch.tensor(
#                     [[2.0, -1.0, 3.0], [-4.0, 5.0, -6.0], [7.0, -8.0, 9.0], [-10.0, 11.0, -12.0]],
#                     device="cuda",
#                     dtype=dtype,
#                 ),
#                 "V": torch.tensor(
#                     [[1.0, 0.5, -0.5], [-1.0, 2.0, 3.0], [4.0, -2.0, 1.0], [0.0, 1.0, -1.0]],
#                     device="cuda",
#                     dtype=dtype,
#                 ),
#                 "output": torch.empty(4, 3, device="cuda", dtype=dtype),
#                 "M": 4,
#                 "N": 4,
#                 "d": 3,
#             }
#         )

#         # large_matrices
#         tests.append(
#             {
#                 "Q": torch.empty((64, 32), device="cuda", dtype=dtype).uniform_(-0.1, 0.1),
#                 "K": torch.empty((128, 32), device="cuda", dtype=dtype).uniform_(-0.1, 0.1),
#                 "V": torch.empty((128, 32), device="cuda", dtype=dtype).uniform_(-0.1, 0.1),
#                 "output": torch.empty(64, 32, device="cuda", dtype=dtype),
#                 "M": 64,
#                 "N": 128,
#                 "d": 32,
#             }
#         )

#         return tests

#     def generate_performance_test(self) -> Dict[str, Any]:
#         dtype = torch.float32
#         M, N, d = 512, 256, 128
#         Q = torch.empty((512, 128), device="cuda", dtype=dtype).uniform_(-0.1, 0.1)
#         K = torch.empty((256, 128), device="cuda", dtype=dtype).uniform_(-0.1, 0.1)
#         V = torch.empty((256, 128), device="cuda", dtype=dtype).uniform_(-0.1, 0.1)
#         output = torch.empty(M, d, device="cuda", dtype=dtype)
#         return {"Q": Q, "K": K, "V": V, "output": output, "M": M, "N": N, "d": d}


import torch
import ctypes
import math
from typing import Any, Dict, List

class OperatorSpec:
    def __init__(self):
        self.name = "Flash Attention"
        self.atol = 1e-3
        self.rtol = 1e-3

    def reference(self, Q, K, V, d):
        # 标准 Attention 公式: Softmax(QK^T / sqrt(d)) * V
        scale = 1.0 / math.sqrt(d)
        attn = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn = torch.softmax(attn.float(), dim=-1).to(Q.dtype)
        return torch.matmul(attn, V)

    def clear_outputs(self, case):
        case["output"].zero_()

    def validate(self, case):
        ref_out = self.reference(case["Q"], case["K"], case["V"], case["d"])
        return torch.allclose(case["output"], ref_out, atol=self.atol, rtol=self.rtol)

    def get_cuda_args(self, case):
        # 对应 solve(Q, K, V, output, M, N, d)
        return [case["Q"], case["K"], case["V"], case["output"], 
                case["M"], case["N"], case["d"]], \
               [None, None, None, None, ctypes.c_int, ctypes.c_int, ctypes.c_int]

    def get_triton_args(self, case):
        return {
            "Q_ptr": case["Q"], "K_ptr": case["K"], "V_ptr": case["V"], "OUT_ptr": case["output"],
            "M": case["M"], "N": case["N"], "d": case["d"],
            "Q_stride_M": case["Q"].stride(0), "Q_stride_d": case["Q"].stride(1),
            "K_stride_N": case["K"].stride(0), "K_stride_d": case["K"].stride(1),
            "V_stride_N": case["V"].stride(0), "V_stride_d": case["V"].stride(1),
            "OUT_stride_M": case["output"].stride(0), "OUT_stride_d": case["output"].stride(1),
            "BLOCKSIZE_M": 32, "BLOCKSIZE_N": 64, "BLOCKSIZE_d": 128 # 需与Kernel一致
        }

    def get_triton_grid(self, case, meta):
        # 2D Grid: M维度和d维度
        import triton
        return (triton.cdiv(case["M"], meta['BLOCKSIZE_M']), 
                triton.cdiv(case["d"], meta['BLOCKSIZE_d']))

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        # 生成不同规模的测试用例
        tests = []
        for M, N, d in [(2, 3, 4), (16, 16, 64), (64, 128, 32)]:
            tests.append({
                "Q": torch.randn((M, d), device="cuda"),
                "K": torch.randn((N, d), device="cuda"),
                "V": torch.randn((N, d), device="cuda"),
                "output": torch.zeros((M, d), device="cuda"),
                "M": M, "N": N, "d": d
            })
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        M, N, d = 1024, 1024, 128
        return {
            "Q": torch.randn((M, d), device="cuda"),
            "K": torch.randn((N, d), device="cuda"),
            "V": torch.randn((N, d), device="cuda"),
            "output": torch.zeros((M, d), device="cuda"),
            "M": M, "N": N, "d": d
        }