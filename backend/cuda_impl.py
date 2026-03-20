from core.executor import BaseExecutor
from utils.compiler import compiler
from triton.testing import do_bench 

class CudaExecutor(BaseExecutor):        
    def compile(self):
        # 从 spec 获取需要注入的宏
        macros = self.spec.get_macros(self.config)
        # 调用编译器
        self.module = compiler.compile(
            self.op_name, 
            self.kernel_file, 
            macros
        )

    def run(self, inputs):
        if self.module is None: self.compile()
        # 执行内核
        self.module.solve(**inputs)
        # 返回结果
        return inputs[self.spec.output_name]
