import abc
from triton.testing import do_bench

# 执行器 管理编译、精度验证以及性能测试等逻辑 
class BaseExecutor(abc.ABC):
    def __init__(self, op_name, kernel_file, spec, config):
        self.op_name = op_name
        self.kernel_file = kernel_file
        self.spec = spec      # 传入对应的 OperatorSpec
        self.config = config  # 当前的配置，例如 {'block_size': 256}
        self.module = None    # 编译后的库对象

    def benchmark(self, inputs, warmup=25, iters=100):
        """
        所有执行器共用这一个逻辑。
        通过 do_bench 确保统计口径（中位数）和预热逻辑完全一致。
        """
        # 定义一个包装，让计时器只测计算部分
        def task():
            return self.run(inputs)
            
        # 返回中位数，单位是毫秒
        return do_bench(task, warmup=warmup, rep=iters, return_mode="median")
    
    @abc.abstractmethod
    def compile(self):
        """将源码编译为可执行对象"""
        pass

    @abc.abstractmethod
    def run(self, inputs):
        """执行算子"""
        pass