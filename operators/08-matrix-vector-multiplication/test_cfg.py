import itertools
from typing import Any, Dict, List

import torch

from core.spec import BaseOperatorSpec


class GEMVSpec(BaseOperatorSpec):
    def __init__(self):
        super().__init__()
        # 1. 基础元数据与精度标准 (fp32)
        self.name = "GEMV"
        self.output_name = "y"
        self.atol = 1e-3
        self.rtol = 1e-3

        # 2. 测试规模定义
        #    y = x @ A, 左向量 x (1, K), 右矩阵 A (K, N), 输出 y (1, N)
        #    K: 收缩维度 (向量长度 / A 行数), N: 输出维度 (A 列数)
        self.x_vals = [
            {"K": 128, "N": 8192},  # K 较小，考察单线程归约循环很短时的表现
            {"K": 1024, "N": 512},
            {"K": 2048, "N": 1024},
            {"K": 4096, "N": 4096},
            {"K": 4096, "N": 8192},
            {"K": 8192, "N": 8192},
        ]
        self.perf_input = {"K": 8192, "N": 8192}

        # 3. Triton 配置生成
        # native: program 负责 BLOCK_N 列，沿 K 以 BLOCK_K 步长累加；float4/smem 由 tl.load
        #         与 num_stages 自动覆盖，调优空间主要是这几个旋钮。
        triton_native_params = [
            {"BLOCK_N": bn, "BLOCK_K": bk, "num_warps": w, "num_stages": s}
            for bn, bk, w, s in itertools.product([128, 256], [32, 64, 128], [4, 8], [2, 3])
        ]
        self.tuning_configs = self.make_configs(triton_native_params, ["native"])

        # splitk: 两阶段 split-K（阶段1 写 partial，阶段2 沿 SPLIT_K 规约），额外调 SPLIT_K。
        triton_splitk_params = [
            {"BLOCK_N": bn, "BLOCK_K": bk, "SPLIT_K": sk, "num_warps": w, "num_stages": s}
            for bn, bk, sk, w, s in itertools.product([128, 256], [32, 64], [2, 4, 8], [4, 8], [2, 3])
        ]
        self.tuning_configs.extend(self.make_configs(triton_splitk_params, ["splitk"]))

        # 4. CUDA 配置生成
        # native/vector 都是 1D block，只调每个 block 的线程数。
        # native: 一个线程负责一个输出列；vector: 一个线程负责 4 个输出列 (float4)。
        # block_x>=512 在 scaling 中从不胜出，且大 block 编译更吃内存，故裁掉。
        native_block_params = [{"block_x": bx} for bx in [32, 64, 128, 256]]
        self.cuda_tuning_configs = self.make_configs(native_block_params, ["native"])

        # vector 实测仅 block_x=32 附近胜出，保留小线程数即可。
        vector_block_params = [{"block_x": bx} for bx in [32, 64, 128]]
        self.cuda_tuning_configs.extend(self.make_configs(vector_block_params, ["vector"]))

        # smem: 在 vector 基础上把 x 分块缓存进 shared memory，额外调 TILE_K（每个 tile 缓存的 x 长度）。
        # block_x=256、tile_k=512 从不胜出，已裁掉。
        smem_params = [
            {"block_x": bx, "tile_k": tk}
            for bx, tk in itertools.product([128, 512], [256, 1024])
        ]
        self.cuda_tuning_configs.extend(self.make_configs(smem_params, ["smem"]))

        # splitk: 沿 K 维切成 SPLIT_K 段并行求部分和（smem 缓存 x），再用第二个 kernel 沿
        # SPLIT_K 规约，避免跨 block 的 atomic 争用。适合 N 小 K 大的长条。
        # split_k=2、tile_k=512、block_x=512 从不胜出，已裁掉。
        splitk_params = [
            {"block_x": bx, "split_k": sk, "tile_k": tk}
            for bx, sk, tk in itertools.product([128, 256], [4, 8, 16], [256])
        ]
        self.cuda_tuning_configs.extend(self.make_configs(splitk_params, ["splitk"]))

    # 1. 将 Python Config 转换为 C++ 宏
    def get_macros(self, config: Dict[str, Any]) -> Dict[str, Any]:
        macros = {
            "BLOCK_X": config.get("block_x"),
            "TILE_K": config.get("tile_k"),
            "SPLIT_K": config.get("split_k"),
        }
        return {k: v for k, v in macros.items() if v is not None}

    # 2. 物理指标计算
    def get_bytes_accessed(self, case: Dict[str, Any]) -> int:
        k = case.get("K", self.perf_input["K"])
        n = case.get("N", self.perf_input["N"])
        # 读 A (K*N) + 读 x (K) + 写 y (N), fp32 = 4 bytes
        return (k * n + k + n) * 4

    def get_total_flops(self, case: Dict[str, Any]) -> int:
        k = case.get("K", self.perf_input["K"])
        n = case.get("N", self.perf_input["N"])
        # 每个输出元素做 K 次乘加，按 2 FLOPs/乘加计
        return 2 * k * n

    # 3. 验证与测试实现
    def reference_impl(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
        y: torch.Tensor,
        K: int,
        N: int,
        **kwargs,
    ):
        assert x.shape == (K,)
        assert A.shape == (K, N)
        assert y.shape == (N,)
        return (x @ A).to(A.dtype)

    def generate_example_test(self) -> Dict[str, Any]:
        # 注意: vector.cu 走 float4，要求 K、N 均为 4 的倍数。
        dtype = torch.float32
        k, n = 4, 4
        return {
            "x": torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda", dtype=dtype),
            "A": torch.tensor(
                [[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 2.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]],
                device="cuda",
                dtype=dtype,
            ),
            "y": torch.empty(n, device="cuda", dtype=dtype),
            "K": k,
            "N": n,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests: List[Dict[str, Any]] = []

        # 基础固定用例 (K、N 均为 4 的倍数)
        tests.append(
            {
                "x": torch.tensor([1.0, 0.0, 1.0, 0.0], device="cuda", dtype=dtype),
                "A": torch.tensor(
                    [[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 2.0, 1.0], [3.0, 0.0, 1.0, 2.0], [0.0, 1.0, 0.0, 1.0]],
                    device="cuda",
                    dtype=dtype,
                ),
                "y": torch.empty(4, device="cuda", dtype=dtype),
                "K": 4,
                "N": 4,
            }
        )
        tests.append(
            {
                "x": torch.ones(4, device="cuda", dtype=dtype),
                "A": torch.zeros((4, 8), device="cuda", dtype=dtype),
                "y": torch.empty(8, device="cuda", dtype=dtype),
                "K": 4,
                "N": 8,
            }
        )
        tests.append(
            {
                "x": torch.empty(8, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                "A": torch.empty((8, 16), device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                "y": torch.empty(16, device="cuda", dtype=dtype),
                "K": 8,
                "N": 16,
            }
        )

        # 边界尺寸 (K、N 均为 4 的倍数)
        for k, n in [(4, 4), (1024, 4), (256, 1024)]:
            tests.append(
                {
                    "x": torch.empty(k, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                    "A": torch.empty((k, n), device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
                    "y": torch.empty(n, device="cuda", dtype=dtype),
                    "K": k,
                    "N": n,
                }
            )

        return tests

    def generate_performance_test(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        dtype = torch.float32
        k = cfg.get("K", self.perf_input["K"])
        n = cfg.get("N", self.perf_input["N"])

        return {
            "x": torch.empty(k, device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
            "A": torch.empty((k, n), device="cuda", dtype=dtype).uniform_(-1.0, 1.0),
            "y": torch.empty(n, device="cuda", dtype=dtype),
            "K": k,
            "N": n,
        }
