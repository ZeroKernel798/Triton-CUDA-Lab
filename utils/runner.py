import torch
import math
from .prober import KernelProber

class BenchmarkRunner:
    def __init__(self, spec, args, logger):
        self.spec = spec
        self.args = args
        self.logger = logger

    def dispatch_call(self, fn, p_case, is_cuda, params, has_kwargs):
        """严格按名绑定，保持简洁"""
        if is_cuda:
            try:
                args_to_pass = {p: p_case[p] for p in params if p != 'kwargs'}
            except KeyError as e:
                raise KeyError(f"CUDA 算子参数绑定失败！缺少参数: {e}")
            try:
                return fn(**args_to_pass)
            except TypeError:
                actual_params = [p for p in params if p != 'kwargs']
                return fn(*[args_to_pass[p] for p in actual_params])
        else:
            if has_kwargs: return fn(**p_case)
            filtered = {p: p_case[p] for p in params if p in p_case}
            return fn(**filtered)

    def measure_latency(self, solve_fn, p_case, is_cuda, params, has_kwargs):
        """性能测量核心循环"""
        for _ in range(self.args.warmup): 
            self.dispatch_call(solve_fn, p_case, is_cuda, params, has_kwargs)
        
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(self.args.epoch): 
            self.dispatch_call(solve_fn, p_case, is_cuda, params, has_kwargs)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / self.args.epoch

    def run_benchmark(self, solve_fn, name, is_cuda):
        # 1. 探测与元数据准备
        params, has_kwargs = KernelProber.probe(solve_fn)
        atol = getattr(self.spec, "atol", 1e-5)
        rtol = getattr(self.spec, "rtol", 1e-5)
        out_n = getattr(self.spec, "output_name", "C")
        
        print("\n" + "="*80)
        print(f"📂 FILE: {name} | {'CUDA' if is_cuda else 'Triton'}")
        print("="*80)

        # 🚀 内部校验辅助函数：解决“原地修改”问题，减少重复代码
        def validate_accuracy(case, label):
            # A. 运行前先备份一份给 Reference (处女地)
            ref_case = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in case.items()}
            
            # B. 分别执行测试算子和参考算子
            self.dispatch_call(solve_fn, case, is_cuda, params, has_kwargs)
            self.dispatch_call(self.spec.reference_impl, ref_case, False, set(), True)
            
            # C. 比对
            if not torch.allclose(case[out_n], ref_case[out_n], atol=atol, rtol=rtol):
                print(f"❌ {label} 精度检查失败！")
                print(f"   Max Diff: {(case[out_n] - ref_case[out_n]).abs().max().item()}")
                return False
            return True

        # 2. 功能验证阶段
        if hasattr(self.spec, "generate_example_test"):
            print(f"    Running Example Tests...")
            if not validate_accuracy(self.spec.generate_example_test(), "Example Test"): return
            print(f"    最小闭环精度校验通过")

        if hasattr(self.spec, "generate_functional_test"):
            test_cases = self.spec.generate_functional_test()
            print(f"    Running {len(test_cases)} Functional Tests...")
            for i, case in enumerate(test_cases):
                n_val = case.get('N', 'unknown')
                if not validate_accuracy(case, f"Case {i} (N={n_val})"): return
            print(f"    所有功能回归测试全部通过！")

        # 3. 性能测试阶段 (Scaling / Tuning)
        if hasattr(self.spec, "generate_performance_test"):
            print(f"    Running Performance Tests...")
            configs = self.spec.cuda_tuning_configs if is_cuda else self.spec.tuning_configs
            
            if self.args.bench_mode == 'scaling':
                print(f"\n📈 Scaling Analysis:")
                print(f"{'Size':>12} | {'Best CFG':>12} | {'Latency':>10} | {'Throughput':>12}")
                print("-" * 55)
                for n in self.spec.x_vals:
                    best_tp, best_ms, best_label = 0.0, 0.0, ""
                    for config in configs:
                        if not has_kwargs and not any(k in params for k in config.keys()): continue 
                        p_case = {**self.spec.generate_performance_test(n), **config}
                        ms = self.measure_latency(solve_fn, p_case, is_cuda, params, has_kwargs)
                        tp = self.spec.get_throughput(n, ms)
                        if tp > best_tp:
                            best_tp, best_ms, best_label = tp, ms, "_".join([f"{v}" for v in config.values()])
                    if best_label:
                        self.logger.record(name, 'scaling', n, best_tp)
                        print(f"{n:12d} | {best_label:>12s} | {best_ms:8.4f} ms | {best_tp:9.2f} GB/s")

            elif self.args.bench_mode == 'tuning':
                base_case = self.spec.generate_performance_test()
                # 兼容 N 可能不存在的情况
                n_val = base_case.get('N', 'Fixed')
                print(f"\n🔍 Tuning Analysis (Size: {n_val}):")
                print(f"{'Config':>15} | {'Latency':>10} | {'Throughput':>12}")
                print("-" * 45)
                for config in configs:
                    tuning_case = {**base_case, **config}
                    # 修复点：这里应该传 params，而不是 kernel_params
                    ms = self.measure_latency(solve_fn, tuning_case, is_cuda, params, has_kwargs)
                    tp = self.spec.get_throughput(n_val if isinstance(n_val, int) else None, ms)
                    label = "_".join([f"{v}" for v in config.values()])
                    self.logger.record(name, 'tuning', label, tp)
                    print(f"{label:>15s} | {ms:8.4f} ms | {tp:9.2f} GB/s")