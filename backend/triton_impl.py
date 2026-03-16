import importlib.util
import os
from core.executor import BaseExecutor

class TritonExecutor(BaseExecutor):
    def __init__(self, op_name, kernel_file, spec, config):
        super().__init__(op_name, kernel_file, spec, config)
        self.solve_fn = None

    def compile(self):
        """Triton 的'编译'：动态加载 Python 脚本"""
        module_name = os.path.basename(self.kernel_file).replace(".py", "")
        spec = importlib.util.spec_from_file_location(module_name, self.kernel_file)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        
        if hasattr(foo, 'solve'):
            self.solve_fn = foo.solve
        else:
            raise AttributeError(f"Triton 文件 {self.kernel_file} 必须包含 'solve' 函数")

    def run(self, inputs):
        if self.solve_fn is None: 
            self.compile()

        full_params = {**inputs, **self.config}
        res = self.solve_fn(**full_params)
        
        if res is None:
            res = inputs.get(self.spec.output_name)

        return res
