import torch
import torch.distributed as dist

def log_master(msg):
    """只在 Rank 0 打印，防止分布式环境下日志爆炸"""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(msg)


class OperatorRunner:
    def __init__(self, spec, executor, args):
        self.spec, self.executor, self.args = spec, executor, args

    def validate(self) -> bool:
        """精度验证：最小闭环 + 功能测试"""
        cases = [self.spec.generate_example_test()] + self.spec.generate_functional_test()
        for case in cases:
            try:
                out = self.executor.run(case)
                # 深度拷贝，防止参考实现修改了原始数据
                ref_case = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in case.items()}
                ref_out = self.spec.reference_impl(**ref_case)
                if ref_out is None:
                    ref_out = ref_case[self.spec.output_name]
                if not self.spec.validate({self.spec.output_name: out}, 
                                        {self.spec.output_name: ref_out}):
                    return False
            except Exception as e:
                log_master(f"      ⚠️ 验证异常: {e}")
                return False
        return True

    def measure(self, size_cfg):
        """物理指标测量"""
        test_case = self.spec.generate_performance_test(size_cfg)
        ms = self.executor.benchmark(test_case)
        metrics_case = {**test_case, "_config": self.executor.config}
        gbps = self.spec.get_throughput(metrics_case, ms)
        tflops = self.spec.get_flops(metrics_case, ms)
        return {"ms": ms, "gbps": gbps, "tflops": tflops}
    
    def profile(self, size_cfg):
        """精准捕获：只 Profile 性能最优的那几次运行"""
        test_case = self.spec.generate_performance_test(size_cfg)
        
        # Warmup (必须要，防止初次加载和初始化干扰)
        for _ in range(10):
            self.executor.run(test_case)
        torch.cuda.synchronize()

        # 通知 nsys/ncu 开始干活
        log_master(f"📡 [Profiler] 开启采样区域...")
        torch.cuda.profiler.start()
        
        # Profile 期间执行次数不宜过多，通常 3-5 次足够看指标
        for i in range(5):
            self.executor.run(test_case)
            
        torch.cuda.synchronize()
        torch.cuda.profiler.stop() # 停止采样
        log_master(f"📡 [Profiler] 采样区域关闭。")
