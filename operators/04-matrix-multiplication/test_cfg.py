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
        self.name = "Matrix multiplication"
        self.output_name = "C" 
        # 设置精度标准
        self.atol = 1e-04
        self.rtol = 1e-04
        # 数据规模测试
        self.x_vals = [
            # 1. 经典小方块：方便 Debug 检查结果
            {"M": 128, "K": 128, "N": 128},
            # 2. 深度学习常用尺寸：典型的全连接层或 Attention 维度
            {"M": 512, "K": 1024, "N": 512},
            # 3. 非对齐/奇数维度：最容易让朴素版 Kernel 崩掉的情况（测试边界逻辑）
            {"M": 1025, "K": 511, "N": 777},
            # 4. 瘦长矩阵：模拟输入 Seq=1 时的推理场景
            {"M": 1, "K": 4096, "N": 4096},
            # 5. 宽大矩阵：
            {"M": 4096, "K": 128, "N": 8192},
            # 6. 大规模压力测试：查看 A100 的极限性能
            {"M": 8192, "K": 128, "N": 8192},
        ]
        # 添加 2^3 到 2^12 的标准方阵测试 (8, 16, ..., 4096)
        for i in range(3, 13):
            size = 2**i
            self.x_vals.append({"M": size, "K": size, "N": size})
        # 核函数参数调优
        # 针对triton
        triton_params = []
        for bm, bn, bk, stages, warps in itertools.product(
            [32, 64, 128],       # BLOCK_SIZE_M
            [32, 64, 128, 256],  # BLOCK_SIZE_N
            [32, 64],            # BLOCK_SIZE_K
            [2, 3, 4, 5],           # num_stages
            [2, 4, 8]            # num_warps
        ):
            triton_params.append({
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "num_stages": stages,
                "num_warps": warps,
                "GROUP_SIZE_M": 8 # 你的 L2 Swizzling 默认值
        })
        # 直接调用静态方法，生成针对 main 版本的调优列表
        self.tuning_configs = self.make_configs(triton_params, ["test"])
        # 针对cuda的配置
        cuda_config_native = [
            {"bx": 32, "by": 8,  "bk": 32},   
            {"bx": 32, "by": 16, "bk": 32}, 
            {"bx": 32, "by": 32, "bk": 32},
        ]
        cuda_config_tile = [
            {"bx": 128, "by": 128, "bk": 8},
            {"bx": 128, "by": 64,  "bk": 8},
            {"bx": 64, "by": 128, "bk": 8},
        ]
        self.cuda_tuning_configs = self.make_configs(cuda_config_native, ["native", "smem_tile"])
        self.cuda_tuning_configs.extend(self.make_configs(cuda_config_tile, 
                                ["thread_tile", "thread_tile_opt", "warp_tile", "warp_tile_double_buffer", "cpasync"]))


    def get_throughput(self, case: Dict[str, Any], ms: float):
        if ms == 0: return 0
        m, k, n = case["M"], case["K"], case["N"]
        # 理论最小访存量：读 A + 读 B + 写 C
        total_bytes = (m * k + k * n + m * n) * 4
        return (total_bytes / 1e9) / (ms / 1000)

    def get_flops(self, case: Dict[str, Any], ms: float):
        """计算 TFLOPS (Tera Floating-point Operations Per Second)"""
        if ms == 0: return 0
        m, k, n = case["M"], case["K"], case["N"]
        # 总操作数: 2 * M * N * K
        # Latency 是 ms，转为秒需要 / 1000
        # 结果除以 1e12 得到 TFLOPS
        total_ops = 2.0 * m * n * k
        tflops = total_ops / (ms * 1e9) 
        return tflops

    def reference_impl(
        self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, M: int, N: int, K: int, **kwargs):
        assert A.shape == (M, K)
        assert B.shape == (K, N)
        assert C.shape == (M, N)
        assert A.dtype == B.dtype == C.dtype
        assert A.device == B.device == C.device

        torch.matmul(A, B, out=C)

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        M, N, K = 2, 2, 2
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=dtype)
        B = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device="cuda", dtype=dtype)
        C = torch.empty(M, N, device="cuda", dtype=dtype)
        return {
            "A": A,
            "B": B,
            "C": C,
            "M": M,
            "N": N,
            "K": K,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        # 约定标准：(M, K) * (K, N) -> (M, N)
        # 参数顺序：name, M, K, N, A_data, B_data
        test_specs = [
            # Basic test cases
            ("basic_2x2", 2, 2, 2, [[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]),
            ("basic_1x3_3x1", 1, 3, 1, [[1.0, 2.0, 3.0]], [[4.0], [5.0], [6.0]]),
            (
                "identity_matrix",
                3,
                3,
                3,
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ),
            ("zero_matrix", 2, 2, 2, [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]),
            (
                "rectangular_matrices",
                2,
                3,
                1,
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[1.0], [2.0], [3.0]],
            ),
        ]

        test_cases = []
        for _, m, k, n, a_vals, b_vals in test_specs:
            test_cases.append(
                {
                    "A": torch.tensor(a_vals, device="cuda", dtype=dtype),
                    "B": torch.tensor(b_vals, device="cuda", dtype=dtype),
                    "C": torch.empty(m, n, device="cuda", dtype=dtype), # 结果是 M x N
                    "M": m,
                    "K": k,
                    "N": n,
                }
            )

        # 随机测试用例 (M, K, N)
        for _, m, k, n in [
            ("small_square", 4, 4, 4),
            ("medium_rectangular", 8, 10, 6),
            ("large_rectangular", 16, 20, 12),
            ("tall_matrix", 32, 16, 8),
            ("wide_matrix", 8, 32, 16),
        ]:
            test_cases.append(
                {
                    "A": torch.empty(m, k, device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
                    "B": torch.empty(k, n, device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
                    "C": torch.empty(m, n, device="cuda", dtype=dtype),
                    "M": m,
                    "K": k,
                    "N": n,
                }
            )

        # 边界情况 (M, K, N)
        for _, m, k, n in [
            ("single_element", 1, 1, 1),
            ("single_row", 1, 5, 3), # 1x5 * 5x3 -> 1x3
            ("single_column", 5, 3, 1), # 5x3 * 3x1 -> 5x1
            ("max_dimensions", 8192, 4096, 6144),
        ]:
            test_cases.append(
                {
                    "A": torch.empty(m, k, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                    "B": torch.empty(k, n, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                    "C": torch.empty(m, n, device="cuda", dtype=dtype),
                    "M": m,
                    "K": k,
                    "N": n,
                }
            )

        return test_cases


    def generate_performance_test(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        dtype = torch.float32
        M = cfg.get("M", 8192)
        N = cfg.get("N", 4096)
        K = cfg.get("K", 6144)
        return {
            "A": torch.empty(M, K, device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
            "B": torch.empty(K, N, device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
            "C": torch.empty(M, N, device="cuda", dtype=dtype),
            "M": M,
            "N": N,
            "K": K,
        }