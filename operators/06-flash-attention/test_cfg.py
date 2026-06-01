import torch
import itertools
from typing import Any, Dict, List
from core.spec import BaseOperatorSpec
import torch.nn.functional as F

class FlashAttentionSpec(BaseOperatorSpec):
    def __init__(self):
        super().__init__()
        # 1. 基础元数据与精度标准
        self.name = "Flash Attention"
        self.output_name = "O"
        self.atol = 1e-02
        self.rtol = 1e-02

        # 2. 测试规模定义 (M: Q行数, N: KV行数, d: 特征维度)
        self.x_vals = [
            {"M": 128, "N": 128, "d": 64},   # 基础 Debug
            {"M": 512, "N": 512, "d": 64},   # 标准尺寸
            {"M": 1024, "N": 1024, "d": 128}, # 常用尺寸
            {"M": 2048, "N": 2048, "d": 128}, # 长文本
            {"M": 4096, "N": 4096, "d": 64},  # 极限测试
        ]

        # 3. CUDA 配置生成 (Br: Q 分块, Bc: KV 分块)
        cuda_params = []
        for br, bc in itertools.product(
            [16, 32], 
            [16, 32]
        ):
            cuda_params.append({
                "Br": br,
                "Bc": bc
            })
        
        # 映射到不同的实现版本
        self.cuda_tuning_configs = self.make_configs(cuda_params, [
            "flashattentionv1",
            "flashattentionv2"
        ])

    # 1. 将 Python Config 转换为 C++ 宏
    def get_macros(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "BR": config.get("Br"),
            "BC": config.get("Bc"),
        }

    # 2. 物理指标计算
    def get_bytes_accessed(self, case: Dict[str, Any]) -> int:
        M, N, d = case["M"], case["N"], case["d"]
        # 理论最小访存：读 Q(Md) + 读 K(Nd) + 读 V(Nd) + 写 O(Md)
        # 假设 float32 (4 bytes)
        return (M * d + N * d + N * d + M * d) * 4

    def get_total_flops(self, case: Dict[str, Any]) -> int:
        M, N, d = case["M"], case["N"], case["d"]
        # Attention 计算量约为 4 * M * N * d
        # 具体为: QK^T (2MNd) + Softmax (忽略) + AV (2MNd)
        return 4 * M * N * d

    # 3. 验证与测试实现
    def reference_impl(self, Q, K, V, O, M, N, d, **kwargs):
        """最强对标实现：使用 PyTorch 原生 SDPA"""
        # 🌟 这里的 Q, K, V 通常期望是 [Batch, Head, Seq, Dim]
        # 如果你现在是 [M, d] 这种 2D 张量，需要临时升维
        q_4d = Q.view(1, 1, M, d)
        k_4d = K.view(1, 1, N, d)
        v_4d = V.view(1, 1, N, d)
        
        # 核心调用：它内部会自动处理 scale (默认就是 1/sqrt(d))
        # 它不会产生巨大的 [M, N] 中间矩阵，显存占用从 O(N^2) 降到 O(N)
        res = F.scaled_dot_product_attention(
            q_4d, k_4d, v_4d, 
            attn_mask=None, 
            dropout_p=0.0, 
            is_causal=False
        )
        
        # 拿回结果并写回 O
        O.copy_(res.view(M, d))

    def generate_example_test(self) -> Dict[str, Any]:
        """最小闭环验证 (2x4)"""
        dtype = torch.float32
        Q = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], device="cuda", dtype=dtype)
        K = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], device="cuda", dtype=dtype)
        V = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]], device="cuda", dtype=dtype)
        O = torch.empty(2, 4, device="cuda", dtype=dtype)
        return {"Q": Q, "K": K, "V": V, "O": O, "M": 2, "N": 3, "d": 4}

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        """全量功能回归测试"""
        dtype = torch.float32
        tests = []

        # basic_example
        tests.append({
            "Q": torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], device="cuda", dtype=dtype),
            "K": torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], device="cuda", dtype=dtype),
            "V": torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]], device="cuda", dtype=dtype),
            "O": torch.empty(2, 4, device="cuda", dtype=dtype),
            "M": 2, "N": 3, "d": 4,
        })

        # zero_matrices
        tests.append({
            "Q": torch.zeros((3, 5), device="cuda", dtype=dtype),
            "K": torch.zeros((3, 5), device="cuda", dtype=dtype),
            "V": torch.zeros((3, 5), device="cuda", dtype=dtype),
            "O": torch.empty(3, 5, device="cuda", dtype=dtype),
            "M": 3, "N": 3, "d": 5,
        })

        # mixed_values
        tests.append({
            "Q": torch.tensor([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0], [-7.0, 8.0, -9.0], [10.0, -11.0, 12.0]], device="cuda", dtype=dtype),
            "K": torch.tensor([[2.0, -1.0, 3.0], [-4.0, 5.0, -6.0], [7.0, -8.0, 9.0], [-10.0, 11.0, -12.0]], device="cuda", dtype=dtype),
            "V": torch.tensor([[1.0, 0.5, -0.5], [-1.0, 2.0, 3.0], [4.0, -2.0, 1.0], [0.0, 1.0, -1.0]], device="cuda", dtype=dtype),
            "O": torch.empty(4, 3, device="cuda", dtype=dtype),
            "M": 4, "N": 4, "d": 3,
        })

        # large_matrices
        tests.append({
            "Q": torch.empty((64, 32), device="cuda", dtype=dtype).uniform_(-0.1, 0.1),
            "K": torch.empty((128, 32), device="cuda", dtype=dtype).uniform_(-0.1, 0.1),
            "V": torch.empty((128, 32), device="cuda", dtype=dtype).uniform_(-0.1, 0.1),
            "O": torch.empty(64, 32, device="cuda", dtype=dtype),
            "M": 64, "N": 128, "d": 32,
        })

        return tests

    def generate_performance_test(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """压测数据生成"""
        dtype = torch.float32
        M = cfg.get("M", 512)
        N = cfg.get("N", 256)
        d = cfg.get("d", 128)

        return {
            "Q": torch.empty((M, d), device="cuda", dtype=dtype).uniform_(-0.1, 0.1),
            "K": torch.empty((N, d), device="cuda", dtype=dtype).uniform_(-0.1, 0.1),
            "V": torch.empty((N, d), device="cuda", dtype=dtype).uniform_(-0.1, 0.1),
            "O": torch.empty(M, d, device="cuda", dtype=dtype),
            "M": M, "N": N, "d": d
        }