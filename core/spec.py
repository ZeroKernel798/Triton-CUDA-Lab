import torch
import abc
from typing import Any, Dict, List

class BaseOperatorSpec(abc.ABC):
    def __init__(self):
        # 基础元数据 
        self.name = "base_operator"
        self.output_name = "out"  # 默认输出 Tensor 的键名
        self.atol = 1e-5         # 绝对误差阈值
        self.rtol = 1e-5         # 相对误差阈值
        
        # 默认测试规模 
        self.perf_input = 1024
        self.x_vals = []         # Scaling 模式下要遍历的规模
        
        # 调优配置列表 
        self.cuda_tuning_configs = []
        self.tuning_configs = [] 
    
    # 默认不需要 nccl要重构
    def is_nccl_version(self, version_name: str) -> bool:
        """
        默认所有版本都不是分布式版本。
        只有在子类（如 TopKSelectionSpec）中重写此方法并返回 True 时，
        框架才会将其视为分布式算子。
        """
        return "nccl" in version_name.lower()

    # 1. 辅助工具，生成配置参数池和查找配置
    @staticmethod
    def make_configs(params: List[Dict[str, Any]], versions: List[str]) -> List[Dict[str, Any]]:
        """将参数池与版本名组合，生成带 version 标签的配置列表"""
        return [{**p, "version": v} for v in versions for p in params]

    def get_configs_for_version(self, version: str):
        # 先在 Triton 桶找
        t_cfgs = [c for c in getattr(self, 'tuning_configs', []) if c.get('version') == version]
        if t_cfgs: return t_cfgs
        # 没找到再去 CUDA 桶找
        return [c for c in getattr(self, 'cuda_tuning_configs', []) if c.get('version') == version]

    def should_skip_validation(self, version: str, config: Dict[str, Any] | None = None) -> bool:
        """
        允许具体算子按版本选择性跳过精度验证。
        默认所有版本都参与验证。
        """
        return False

    def get_validation_cases(self, version: str, config: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        """
        允许具体算子按版本定制验证输入。
        默认沿用最小闭环 + 功能回归。
        """
        return [self.generate_example_test()] + self.generate_functional_test()

    # 2. 物理指标计算 
    def get_throughput(self, case: Dict[str, Any], ms: float) -> float:
        """计算吞吐量 (GB/s)"""
        if ms <= 0: return 0.0
        bytes_accessed = self.get_bytes_accessed(case)
        # 公式: bytes / 10^9 / (ms / 1000)
        return bytes_accessed / 1e9 / (ms / 1000)

    def get_flops(self, case: Dict[str, Any], ms: float) -> float:
        """计算计算强度 (TFLOPS)"""
        if ms <= 0: return 0.0
        total_flops = self.get_total_flops(case)
        # 公式: ops / (ms * 10^9)
        return total_flops / (ms * 1e9)

    # 3. 精度验证逻辑 
    def validate(self, outputs: Dict[str, torch.Tensor], reference: Dict[str, torch.Tensor]) -> bool:
        """通用精度校验：对比输出字典中 output_name 对应的 Tensor"""
        out = outputs.get(self.output_name)
        ref = reference.get(self.output_name)
        if out is None or ref is None:
            raise KeyError(f"找不到输出键名: {self.output_name}")
        return torch.allclose(out, ref, atol=self.atol, rtol=self.rtol)

    # 4. 抽象接口：必须由具体的算子 cfg.py 实现 
    @abc.abstractmethod
    def get_macros(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """核心：将调优配置映射为 C++ 编译宏"""
        pass

    @abc.abstractmethod
    def get_bytes_accessed(self, case: Dict[str, Any]) -> int:
        """数学属性：计算该算子的总访存量 (Bytes)"""
        pass

    @abc.abstractmethod
    def get_total_flops(self, case: Dict[str, Any]) -> int:
        """数学属性：计算该算子的总浮点运算次数"""
        pass

    @abc.abstractmethod
    def reference_impl(self, **kwargs):
        """标准答案：调用 PyTorch 官方函数"""
        pass

    @abc.abstractmethod
    def generate_example_test(self) -> Dict[str, Any]:
        """生成最小闭环测试数据"""
        pass

    @abc.abstractmethod
    def generate_functional_test(self) -> List[Dict[str, Any]]:
        """生成一组用于功能回归测试的数据"""
        pass

    @abc.abstractmethod
    def generate_performance_test(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """生成用于性能压测的大规模数据"""
        pass
    
