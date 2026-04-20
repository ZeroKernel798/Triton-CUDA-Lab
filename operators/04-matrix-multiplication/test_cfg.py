import torch
import itertools
from typing import Any, Dict, List
from core.spec import BaseOperatorSpec

class MatrixMultiplicationSpec(BaseOperatorSpec):
    def __init__(self):
        super().__init__()
        # 1. 基础元数据与精度标准
        self.name = "Matrix multiplication"
        self.output_name = "C"
        self.atol = 1e-02
        self.rtol = 1e-02
        
        # 2. 数据规模测试 (保持原有的 x_vals)
        self.x_vals = [
            {"M": 128, "K": 128, "N": 128},
            {"M": 512, "K": 1024, "N": 512},
            {"M": 1025, "K": 511, "N": 777},
            {"M": 4096, "K": 128, "N": 8192},
            {"M": 8192, "K": 4096, "N": 6144},
        ]

        # 3. Triton 配置生成
        triton_params = []
        for bm, bn, bk, stages, warps in itertools.product(
            [32],       # BLOCK_SIZE_M
            [32, 256],  # BLOCK_SIZE_N
            [32],       # BLOCK_SIZE_K
            [2],        # num_stages
            [2]         # num_warps
        ):
            triton_params.append({
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "num_stages": stages,
                "num_warps": warps,
                "GROUP_SIZE_M": 8 
            })
        self.tuning_configs = self.make_configs(triton_params, ["test"])

        # 4. CUDA 配置生成
        # Native & Smem Tile
        cuda_config_native = [
            {"bx": 32, "by": 8,  "bk": 32},   
            {"bx": 32, "by": 16, "bk": 32}, 
            {"bx": 32, "by": 32, "bk": 32},
        ]
        # 高级 Tile 策略
        cuda_config_tile = [
            {"bx": 128, "by": 128, "bk": 8},
            {"bx": 128, "by": 64,  "bk": 8},
            {"bx": 64, "by": 128, "bk": 8},
        ]
        
        self.cuda_tuning_configs = self.make_configs(cuda_config_native, ["native", "smem_tile"])
        self.cuda_tuning_configs.extend(self.make_configs(cuda_config_tile, [
            "thread_tile", "thread_tile_opt", "warp_tile", 
            "warp_tile_double_buffer", "cpasync", "tc"
        ]))

    # 1. 将 Python Config 转换为 C++ 宏
    def get_macros(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # 统一映射 CUDA 的坐标配置到宏定义
        macros = {
            "BLOCK_X": config.get("bx"),
            "BLOCK_Y": config.get("by"),
            "BLOCK_K": config.get("bk"),
        }
        return {k: v for k, v in macros.items() if v is not None}

    # 2. 物理指标计算 (对应原有的 get_throughput 和 get_flops 逻辑)
    def get_bytes_accessed(self, case: Dict[str, Any]) -> int:
        m, k, n = case["M"], case["K"], case["N"]
        # 理论最小访存量：读 A + 读 B + 写 C (float32 = 4 bytes)
        return (m * k + k * n + m * n) * 4

    def get_total_flops(self, case: Dict[str, Any]) -> int:
        m, k, n = case["M"], case["K"], case["N"]
        # 总操作数: 2 * M * N * K
        return 2 * m * n * k

    # 3. 验证与测试实现
    def reference_impl(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, M: int, N: int, K: int, **kwargs):
        """标准实现"""
        assert A.shape == (M, K)
        assert B.shape == (K, N)
        assert C.shape == (M, N)
        torch.matmul(A, B, out=C)

    def generate_example_test(self) -> Dict[str, Any]:
        """最小闭环验证 (2x2)"""
        dtype = torch.float32
        M, N, K = 2, 2, 2
        return {
            "A": torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=dtype),
            "B": torch.tensor([[5.0, 6.0], [7.0, 8.0]], device="cuda", dtype=dtype),
            "C": torch.empty(M, N, device="cuda", dtype=dtype),
            "M": M, "N": N, "K": K,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        """全量功能回归测试"""
        dtype = torch.float32
        test_cases = []

        # 1. 基础固定用例
        test_specs = [
            ("basic_2x2", 2, 2, 2, [[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]),
            ("basic_1x3_3x1", 1, 3, 1, [[1.0, 2.0, 3.0]], [[4.0], [5.0], [6.0]]),
            ("identity", 3, 3, 3, torch.eye(3).tolist(), torch.eye(3).tolist()),
            ("zero", 2, 2, 2, [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]),
            ("rect", 2, 3, 1, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[1.0], [2.0], [3.0]]),
        ]

        for _, m, k, n, a_vals, b_vals in test_specs:
            test_cases.append({
                "A": torch.tensor(a_vals, device="cuda", dtype=dtype),
                "B": torch.tensor(b_vals, device="cuda", dtype=dtype),
                "C": torch.empty(m, n, device="cuda", dtype=dtype),
                "M": m, "K": k, "N": n,
            })

        # 2. 随机尺寸测试 (M, K, N)
        for m, k, n in [(4,4,4), (8,10,6), (16,20,12), (32,16,8), (8,32,16)]:
            test_cases.append({
                "A": torch.empty(m, k, device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
                "B": torch.empty(k, n, device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
                "C": torch.empty(m, n, device="cuda", dtype=dtype),
                "M": m, "K": k, "N": n,
            })

        # 3. 边界情况测试
        for m, k, n in [(1,1,1), (1,5,3), (5,3,1), (8192, 4096, 6144)]:
            test_cases.append({
                "A": torch.empty(m, k, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                "B": torch.empty(k, n, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                "C": torch.empty(m, n, device="cuda", dtype=dtype),
                "M": m, "K": k, "N": n,
            })

        return test_cases

    def generate_performance_test(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """压测数据生成"""
        dtype = torch.float32
        M, N, K = cfg.get("M", 8192), cfg.get("N", 4096), cfg.get("K", 6144)
        return {
            "A": torch.empty(M, K, device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
            "B": torch.empty(K, N, device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
            "C": torch.empty(M, N, device="cuda", dtype=dtype),
            "M": M, "N": N, "K": K,
        }