import torch
from typing import Any, Dict, List


class OperatorSpec:
    def __init__(self):
        # 设置算子名字以及输出名字
        self.name = "Matrix multiplication"
        self.output_name = "C" 
        # 设置精度标准
        self.atol = 1e-05
        self.rtol = 1e-05
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
        # 默认配置 给简单配置使用 仅仅验证正确性
        self.base_cfg = [
            {"bx": 32,  "by": 32,  "bk": 32, "version": "native"},
            {"bx": 32,  "by": 32,  "bk": 32, "version": "smem_tile"},
            
            # 这里是重点！thread_tile 和 warp_tile 的起步 bk 设为 8
            # 彻底解决 32x32x32 导致的精度崩溃问题
            {"bx": 128, "by": 128, "bk": 8,  "version": "thread_tile"},
            {"bx": 128, "by": 128, "bk": 8,  "version": "thread_tile_opt"},
            {"bx": 128, "by": 128, "bk": 8,  "version": "warp_tile"},
            {"bx": 128, "by": 128, "bk": 32,  "version": "tf32"},
        ]
        # 核函数参数调优
        # 针对triton
        self.tuning_configs = [
            {"BLOCK_ROW": 32, "BLOCK_COL": 32, "num_warps": 2},
            {"BLOCK_ROW": 8, "BLOCK_COL": 8, "num_warps": 2},
            {"BLOCK_ROW": 16, "BLOCK_COL": 16, "num_warps": 2},
        ]
        # 针对cuda的配置
        self.cuda_tuning_configs = [
            # 1. 朴素版 (Baseline)：用于验证正确性，不考虑访存合并
            {"bx": 32, "by": 8,  "bk": 32, "version": "native"},   
            {"bx": 32, "by": 16, "bk": 32, "version": "native"}, 
            {"bx": 32, "by": 32, "bk": 32, "version": "native"},
            # 2. 分块内积版 (Tiled)：你之前的版本，解决了显存带宽问题
            {"bx": 32, "by": 8, "bk": 32, "version": "smem_tile"},
            {"bx": 32, "by": 16, "bk": 32, "version": "smem_tile"},
            {"bx": 32, "by": 32, "bk": 32, "version": "smem_tile"},
            # 3. 高性能外积版 (Outer)：解决 Smem 带宽问题
            # 注意：bx/by 必须是 8 的倍数
            {"bx": 128, "by": 128, "bk": 8, "version": "thread_tile"},
            {"bx": 128, "by": 64,  "bk": 8, "version": "thread_tile"},
            {"bx": 64, "by": 128, "bk": 8, "version": "thread_tile"},
            # 外积优化版本 解决了银行冲突 使用了向量读取
            {"bx": 128, "by": 128, "bk": 8, "version": "thread_tile_opt"},
            {"bx": 128, "by": 64,  "bk": 8, "version": "thread_tile_opt"},
            {"bx": 64, "by": 128, "bk": 8, "version": "thread_tile_opt"},
            # 4. warp tile版本 更细致的划分
            {"bx": 128, "by": 128, "bk": 8, "version": "warp_tile"},
            {"bx": 128, "by": 64,  "bk": 8, "version": "warp_tile"},
            {"bx": 64, "by": 128, "bk": 8, "version": "warp_tile"},
            # 5. warp tile 双缓冲版本
            {"bx": 128, "by": 128, "bk": 8,  "version": "warp_tile_double_buffer"},
            {"bx": 128, "by": 64,  "bk": 8,  "version": "warp_tile_double_buffer"},
            {"bx": 64,  "by": 128, "bk": 8,  "version": "warp_tile_double_buffer"},
            # 6. cpasync
            {"bx": 128, "by": 128, "bk": 8,  "version": "cpasync"},
            {"bx": 128, "by": 64,  "bk": 8,  "version": "cpasync"},
            {"bx": 64,  "by": 128, "bk": 8,  "version": "cpasync"},
            # 7. tp32
            {"bx": 128, "by": 128, "bk": 16, "version": "tf32"},
            {"bx": 128, "by": 64,  "bk": 16, "version": "tf32"},
            {"bx": 64,  "by": 128, "bk": 16, "version": "tf32"},
            {"bx": 64,  "by": 64,  "bk": 16, "version": "tf32"},
        ]

    def get_throughput(self, case: Dict[str, Any], ms: float):
        if ms == 0: return 0
        m, k, n = case["M"], case["K"], case["N"]
        # 理论最小访存量：读 A + 读 B + 写 C
        total_bytes = (m * k + k * n + m * n) * 4
        return (total_bytes / 1e9) / (ms / 1000)

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