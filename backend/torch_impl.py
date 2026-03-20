from core.executor import BaseExecutor
import torch
import time
import logging
import warnings

torch._inductor.config.triton.cudagraphs = False
torch._logging.set_logs(cudagraphs=False, inductor=False)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="torch._inductor")

class TorchExecutor(BaseExecutor):
    def __init__(self, spec, config):
        # 统一命名为 official，内部逻辑自动切换
        super().__init__("official", None, spec, config)
        self.fastest_fn = None
        self._is_autotuned = False
        self.compiled_fn = None

    def compile(self):
        """
        满足 BaseExecutor 的抽象要求。
        对于 Torch 来说，我们在这里先准备好编译对象，但不触发实际编译（因为需要 inputs 确定 Shape）。
        """
        if self.compiled_fn is None:
            # 开启高性能模式，dynamic=True 减少因 Shape 变化导致的重编
            self.compiled_fn = torch.compile(
                self.spec.reference_impl, 
                mode="max-autotune", 
                dynamic=True
            )

    def _get_median_ms(self, func, inputs, iters=5):
        """内部测速工具"""
        # 预热并同步
        for _ in range(3):
            func(**inputs)
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(iters):
            func(**inputs)
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000 / iters

    def autotune(self, inputs):
        """自动寻优逻辑"""
        if self._is_autotuned: return
        
        # 确保编译对象已准备好
        self.compile()

        # 1. 测试 Eager
        ms_eager = self._get_median_ms(self.spec.reference_impl, inputs)
        
        # 2. 测试 Compile (第一次跑会触发真正的后台编译，耗时较长)
        ms_compile = self._get_median_ms(self.compiled_fn, inputs)

        # 3. 优胜劣汰
        if ms_compile < ms_eager:
            self.fastest_fn = self.compiled_fn
            tag = "COMPILED (Inductor)"
        else:
            self.fastest_fn = self.spec.reference_impl
            tag = "EAGER (CUB/Native)"
            
        self._is_autotuned = True

    def run(self, inputs):
        # 1. 第一次运行进行寻优
        if not self._is_autotuned:
            self.autotune(inputs)
        
        # 2. 执行计算 (不管它是 Eager 还是 Compile)
        # 我们不关心它的返回值，只关心它对 inputs 字典里 Tensor 的修改
        self.fastest_fn(**inputs)
        
        # 3. 🌟 核心修复：直接从 inputs 字典里根据 Spec 定义的名字拿结果
        # 这样即便 reference_impl 忘了写 return，我们也能拿到被修改后的 Tensor
        return inputs[self.spec.output_name]