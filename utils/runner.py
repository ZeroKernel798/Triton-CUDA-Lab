import torch
from triton.testing import do_bench
from .prober import KernelProber
from .logger import KernelProfiler  

class BenchmarkRunner:
    def __init__(self, spec, args, logger):
        self.spec = spec
        self.args = args
        self.logger = logger

    def dispatch_call(self, fn, p_case, is_cuda, params, has_kwargs):
        """分发 CUDA 或 Triton 调用"""
        if is_cuda:
            args_to_pass = {p: p_case[p] for p in params if p in p_case}
            return fn(**args_to_pass)
        else:
            if has_kwargs: 
                return fn(**p_case)
            filtered = {p: p_case[p] for p in params if p in p_case}
            return fn(**filtered)

    def measure_latency(self, solve_fn, p_case, is_cuda, params, has_kwargs):
        """使用 do_bench 测量中值延迟"""
        def benchmark_func():
            self.dispatch_call(solve_fn, p_case, is_cuda, params, has_kwargs)
        
        return do_bench(
            benchmark_func, 
            warmup=self.args.warmup, 
            rep=self.args.epoch,
            return_mode="median"
        )

    def run_benchmark(self, solve_fn, name, is_cuda):
        # 元数据准备
        params, has_kwargs = KernelProber.probe(solve_fn)
        atol = getattr(self.spec, "atol", 1e-5)
        rtol = getattr(self.spec, "rtol", 1e-5)
        out_n = getattr(self.spec, "output_name", "C")
        current_impl_name = name.split('.')[0]
        
        print("\n" + "="*80)
        print(f"📂 FILE: {name} | {'CUDA' if is_cuda else 'Triton'}")
        print("="*80)

        # 内部校验辅助函数
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
                return False
            return True

        # 功能验证阶段 (Profile 前必须保证代码是对的) 
        if hasattr(self.spec, "generate_example_test"):
            if not validate_accuracy(self.spec.generate_example_test(), "Example Test"): return
            print(f"    ✅ 最小闭环精度校验通过")

        if hasattr(self.spec, "generate_functional_test"):
            test_cases = self.spec.generate_functional_test()
            for i, case in enumerate(test_cases):
                if not validate_accuracy(case, f"Case {i}"): return
            print(f"    ✅ 所有功能回归测试全部通过！")

        # Profiling 采样阶段 (nsys/ncu 调试) 
        if getattr(self.args, "profile", False):
            print(f"📸 [PROFILE MODE] 正在复刻 Tuning 逻辑寻找最优配置...")

            # 使用与 Tuning 模式完全一样的 base_case
            base_case = self.spec.generate_performance_test({})
            shape_desc = "x".join([str(v) for k, v in base_case.items() if isinstance(v, (int, float))])
            
            # 获取该实现的所有候选配置
            configs = self.spec.cuda_tuning_configs if is_cuda else self.spec.tuning_configs
            if current_impl_name == "cublas":
                configs_to_try = [{}]
            else:
                configs_to_try = [c for c in configs if c.get("version") == current_impl_name]
            
            # 执行“海选” (和 Tuning 模式一模一样)
            best_ms = float('inf')
            best_cfg = {}
            
            print(f"   ⚡ 正在对 {len(configs_to_try)} 组配置进行压测筛选 (Size: {shape_desc})...")
            for config in configs_to_try:
                p_case = {**base_case, **config}
                ms = self.measure_latency(solve_fn, p_case, is_cuda, params, has_kwargs)
                if ms < best_ms:
                    best_ms = ms
                    best_cfg = config
            
            # 锁定冠军配置
            final_case = {**base_case, **best_cfg}
            display_cfg = {k: v for k, v in best_cfg.items() if k != 'version'}
            print(f"🏆 采样配置已锁定: {display_cfg} (Latency: {best_ms:.4f} ms)")

            # 预热与采样
            for _ in range(5):
                self.dispatch_call(solve_fn, final_case, is_cuda, params, has_kwargs)
            
            # 这样在 NSYS 统计表里，Instances 会直接显示为 9
            with KernelProfiler.profile_scope(enabled=True, label="Lab_Kernel_Capture"):
                self.dispatch_call(solve_fn, final_case, is_cuda, params, has_kwargs)
            
            print(f"    ✅ 采样完成。")
            return # 退出，不跑后面的逻辑

        # 性能测试阶段 
        if hasattr(self.spec, "generate_performance_test"):
            print(f"    Running Performance Tests...")
            configs = self.spec.cuda_tuning_configs if is_cuda else self.spec.tuning_configs
            
            # SCALING 模式：针对不同 Shape 找最优解 
            if self.args.bench_mode == 'scaling':
                print(f"\n📈 Scaling Analysis (Found {len(self.spec.x_vals)} shapes to test):")
                print(f"{'Shape':>20} | {'Best CFG':>12} | {'Latency':>10} | {'Throughput':>12} | {'Compute':>12}")
                print("-" * 85)
                
                # 调试：看看有没有匹配到 Config
                if current_impl_name == "cublas":
                    configs_to_try = [{}] 
                else:
                    configs_to_try = [c for c in configs if c.get("version") == current_impl_name]
                
                if not configs_to_try:
                    print(f"⚠️ 警告: 未在 test_cfg.py 中找到属于 {current_impl_name} 的配置！")

                for size_cfg in self.spec.x_vals:
                    best_tp, best_ms, best_tflops, best_label = 0.0, 0.0, 0.0, ""
                    shape_label = "x".join([str(v) for v in size_cfg.values()])
                    
                    for config in configs_to_try:
                        p_case = {**self.spec.generate_performance_test(size_cfg), **config}
                        # print(f"   Testing {shape_label} with {config}...") 
                        
                        ms = self.measure_latency(solve_fn, p_case, is_cuda, params, has_kwargs)
                        tp = self.spec.get_throughput(p_case, ms)
                        tflops = self.spec.get_flops(p_case, ms)
                        
                        if tp >= best_tp: 
                            best_tp, best_ms, best_tflops = tp, ms, tflops
                            best_label = "official" if current_impl_name == "cublas" else \
                                         "_".join([str(v) for k, v in config.items() if k != 'version'])
                    
                    if not best_label: best_label = "N/A"
                    
                    self.logger.record(name, f'{self.args.bench_mode}_bw', shape_label, best_tp)
                    self.logger.record(name, f'{self.args.bench_mode}_flops', shape_label, best_tflops)
                    self.logger.record(name, f'{self.args.bench_mode}_ms', shape_label, best_ms)
                    print(f"{shape_label:>20s} | {best_label:>12s} | {best_ms:8.4f} ms | {best_tp:9.2f} GB/s | {best_tflops:9.2f} TFLOPS")

            # TUNING 模式：固定 Shape 对比所有配置
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
                    self.logger.record(name, f'{self.args.bench_mode}_ms', label, ms) # ✨ 新增
                    # 这里加入了 tflops 的打印
                    print(f"{label:>15s} | {ms:8.4f} ms | {tp:9.2f} GB/s | {tflops:9.2f} TFLOPS")