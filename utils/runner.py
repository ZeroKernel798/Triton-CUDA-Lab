import torch
from triton.testing import do_bench
from .prober import KernelProber

class BenchmarkRunner:
    def __init__(self, spec, args, logger):
        self.spec = spec
        self.args = args
        self.logger = logger

    def dispatch_call(self, fn, p_case, is_cuda, params, has_kwargs):
        if is_cuda:
            # CUDA 逻辑：严格匹配 C++ 函数签名需要的参数，仅支持关键字传参
            args_to_pass = {p: p_case[p] for p in params if p in p_case}
            return fn(**args_to_pass)
        else:
            # Triton 逻辑：保持灵活性
            if has_kwargs: 
                return fn(**p_case)
            # 如果没有 **kwargs，则根据函数签名过滤参数
            filtered = {p: p_case[p] for p in params if p in p_case}
            return fn(**filtered)

    def measure_latency(self, solve_fn, p_case, is_cuda, params, has_kwargs):
        """
        使用 triton.testing.do_bench 替代手写的 Event 循环
        """
        # 1. 准备调用闭包 (Closure)
        # 因为 do_bench 接受的是一个不带参数的 lambda
        def benchmark_func():
            self.dispatch_call(solve_fn, p_case, is_cuda, params, has_kwargs)

        # 2. 调用 do_bench
        # warmup: 预热次数 (对应你之前的 self.args.warmup)
        # rep: 正式运行迭代次数 (对应你之前的 self.args.epoch)
        # return_mode: "median" 返回中位数, "max" 返回最大值, "min" 返回最小值
        ms = do_bench(
            benchmark_func, 
            warmup=self.args.warmup, 
            rep=self.args.epoch,
            return_mode="median"
        )
        
        return ms

    def run_benchmark(self, solve_fn, name, is_cuda):
        # 1. 探测与元数据准备
        params, has_kwargs = KernelProber.probe(solve_fn)
        atol = getattr(self.spec, "atol", 1e-5)
        rtol = getattr(self.spec, "rtol", 1e-5)
        out_n = getattr(self.spec, "output_name", "C")
        current_impl_name = name.split('.')[0]
        
        print("\n" + "="*80)
        print(f"📂 FILE: {name} | {'CUDA' if is_cuda else 'Triton'}")
        print("="*80)

        # 内部校验辅助函数：支持版本感知的配置
        def validate_accuracy(case, label):
            configs = self.spec.cuda_tuning_configs if is_cuda else self.spec.tuning_configs
            target_cfg = next((c for c in configs if c.get("version") == current_impl_name), {})
            pure_kernel_params = {k: v for k, v in target_cfg.items() if k != 'version'}
            case.update(pure_kernel_params)

            ref_case = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in case.items()}
            self.dispatch_call(solve_fn, case, is_cuda, params, has_kwargs)
            self.dispatch_call(self.spec.reference_impl, ref_case, False, set(), True)
            
            if not torch.allclose(case[out_n], ref_case[out_n], atol=atol, rtol=rtol):
                print(f"❌ {label} 精度检查失败！")
                clean_cfg = {k: v for k, v in case.items() if not isinstance(v, torch.Tensor)}
                print(f"   使用的配置: {clean_cfg}")
                return False
            return True

        # 2. 功能验证阶段
        if hasattr(self.spec, "generate_example_test"):
            if not validate_accuracy(self.spec.generate_example_test(), "Example Test"): return
            print(f"    ✅ 最小闭环精度校验通过")

        if hasattr(self.spec, "generate_functional_test"):
            test_cases = self.spec.generate_functional_test()
            for i, case in enumerate(test_cases):
                if not validate_accuracy(case, f"Case {i}"): return
            print(f"    ✅ 所有功能回归测试全部通过！")

        # 3. 性能测试阶段
        if hasattr(self.spec, "generate_performance_test"):
            print(f"    Running Performance Tests...")
            configs = self.spec.cuda_tuning_configs if is_cuda else self.spec.tuning_configs
            
            # --- SCALING 模式：针对不同 Shape 找最优解 ---
            if self.args.bench_mode == 'scaling':
                print(f"\n📈 Scaling Analysis:")
                # 增加了 Compute (TFLOPS) 列
                print(f"{'Shape':>20} | {'Best CFG':>12} | {'Latency':>10} | {'Throughput':>12} | {'Compute':>12}")
                print("-" * 85) # 加长分割线
                
                for size_cfg in self.spec.x_vals:
                    best_tp, best_ms, best_tflops, best_label = 0.0, 0.0, 0.0, ""
                    shape_label = "x".join([str(v) for v in size_cfg.values()])
                    
                    if current_impl_name == "cublas":
                        configs_to_try = [{}] 
                    else:
                        configs_to_try = [c for c in configs if c.get("version") == current_impl_name]

                    for config in configs_to_try:
                        p_case = {**self.spec.generate_performance_test(size_cfg), **config}
                        ms = self.measure_latency(solve_fn, p_case, is_cuda, params, has_kwargs)
                        tp = self.spec.get_throughput(p_case, ms)
                        tflops = self.spec.get_flops(p_case, ms)
                        
                        if tp > best_tp:
                            best_tp, best_ms, best_tflops = tp, ms, tflops
                            best_label = "official" if current_impl_name == "cublas" else \
                                         "_".join([str(v) for k, v in config.items() if k != 'version'])
                    
                    if best_label:
                        self.logger.record(name, f'{self.args.bench_mode}_bw', shape_label, best_tp)
                        self.logger.record(name, f'{self.args.bench_mode}_flops', shape_label, best_tflops)
                        # 这里加入了 best_tflops 的打印
                        print(f"{shape_label:>20s} | {best_label:>12s} | {best_ms:8.4f} ms | {best_tp:9.2f} GB/s | {best_tflops:9.2f} TFLOPS")

            # --- TUNING 模式：固定 Shape 对比所有配置 ---
            elif self.args.bench_mode == 'tuning':
                base_case = self.spec.generate_performance_test({})
                shape_desc = "x".join([str(v) for k, v in base_case.items() if isinstance(v, (int, float))])

                print(f"\n🔍 Tuning Analysis (Size: {shape_desc}):")
                # 增加了 Compute (TFLOPS) 列
                print(f"{'Config':>15} | {'Latency':>10} | {'Throughput':>12} | {'Compute':>12}")
                print("-" * 65)

                for config in configs:
                    if current_impl_name != "cublas" and config.get("version") != current_impl_name:
                        continue

                    tuning_case = {**base_case, **config}
                    ms = self.measure_latency(solve_fn, tuning_case, is_cuda, params, has_kwargs)
                    tp = self.spec.get_throughput(tuning_case, ms)
                    tflops = self.spec.get_flops(tuning_case, ms)

                    label = "official" if current_impl_name == "cublas" else \
                            "_".join([str(v) for k, v in config.items() if k != 'version'])
                    
                    self.logger.record(name, f'{self.args.bench_mode}_bw', label, tp)
                    self.logger.record(name, f'{self.args.bench_mode}_flops', label, tflops)
                    # 这里加入了 tflops 的打印
                    print(f"{label:>15s} | {ms:8.4f} ms | {tp:9.2f} GB/s | {tflops:9.2f} TFLOPS")