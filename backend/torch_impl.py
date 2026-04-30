from core.executor import BaseExecutor
import torch
import logging
import warnings
from triton.testing import do_bench

torch._inductor.config.triton.cudagraphs = False
torch._logging.set_logs(cudagraphs=False, inductor=False)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="torch._inductor")

class TorchExecutor(BaseExecutor):
    def __init__(self, spec, config, mode="official_auto"):
        # torch baseline 支持 eager / compile / auto 三种对比口径。
        super().__init__(mode, None, spec, config)
        self.mode = mode
        self.fastest_fns = {}
        self.compiled_fns = {}

    def _make_signature(self, inputs):
        signature = []
        for name, value in sorted(inputs.items()):
            if isinstance(value, torch.Tensor):
                signature.append(
                    (
                        name,
                        tuple(value.shape),
                        tuple(value.stride()),
                        str(value.dtype),
                        value.device.type,
                        value.device.index,
                    )
                )
            elif isinstance(value, (int, float, bool, str, type(None))):
                signature.append((name, value))
        return tuple(signature)

    def compile(self, inputs=None):
        """
        满足 BaseExecutor 的抽象要求。
        对于 Torch 来说，我们在这里先准备好编译对象，但不触发实际编译（因为需要 inputs 确定 Shape）。
        """
        if inputs is None:
            return None

        signature = self._make_signature(inputs)
        if signature not in self.compiled_fns:
            torch._dynamo.reset()
            # 使用固定 shape 编译，并为每个输入签名单独缓存 compiled function。
            # reset 可以避免验证阶段和多 shape benchmark 撞到 Dynamo cache limit 后回退 eager。
            # RMSNorm 这类 reduction/pointwise 算子默认 Inductor 往往比 max-autotune 更快。
            self.compiled_fns[signature] = torch.compile(
                self.spec.reference_impl,
                dynamic=False,
            )
        return self.compiled_fns[signature]

    def _get_median_ms(self, func, inputs):
        """内部测速工具，和 BaseExecutor.benchmark 保持同一 do_bench 口径。"""
        def task():
            return func(**inputs)

        return do_bench(task, warmup=25, rep=100, return_mode="median")

    def autotune(self, inputs):
        """自动寻优逻辑"""
        signature = self._make_signature(inputs)
        if signature in self.fastest_fns:
            return

        if self.mode == "official_eager":
            self.fastest_fns[signature] = self.spec.reference_impl
            return

        # compile / auto 模式都需要准备 torch.compile 包装。
        compiled_fn = self.compile(inputs)

        if self.mode == "official_compile":
            self.fastest_fns[signature] = compiled_fn
            return

        # 1. 测试 Eager
        ms_eager = self._get_median_ms(self.spec.reference_impl, inputs)
        
        # 2. 测试 Compile (第一次跑会触发真正的后台编译，耗时较长)
        ms_compile = self._get_median_ms(compiled_fn, inputs)

        # 3. 优胜劣汰
        if ms_compile < ms_eager:
            self.fastest_fns[signature] = compiled_fn
        else:
            self.fastest_fns[signature] = self.spec.reference_impl

    def run(self, inputs):
        # 1. 第一次运行进行寻优
        signature = self._make_signature(inputs)
        self.autotune(inputs)
        
        # 2. 执行计算 (不管它是 Eager 还是 Compile)
        # 兼容返回式 reference_impl 和原地写输出的 reference_impl。
        out = self.fastest_fns[signature](**inputs)
        
        return out if out is not None else inputs[self.spec.output_name]
