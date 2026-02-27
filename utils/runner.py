import torch
import math
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
        current_impl_name = name.split('.')[0] # 获取版本名称 根据文件名获取
        
        print("\n" + "="*80)
        print(f"📂 FILE: {name} | {'CUDA' if is_cuda else 'Triton'}")
        print("="*80)

        # 内部校验辅助函数：支持版本感知的配置
        def validate_accuracy(case, label):
            # 1. 找到该版本的配置
            configs = self.spec.cuda_tuning_configs if is_cuda else self.spec.tuning_configs
            
            # 这里的 target_cfg 包含了 {"block_size": 256, "version": "native"}
            target_cfg = next((c for c in configs if c.get("version") == current_impl_name), {})

            # 2. 提取出纯粹的硬件参数，把 version 这种元数据挡在外面
            # 只要过滤掉 version 这个 key 即可
            pure_kernel_params = {k: v for k, v in target_cfg.items() if k != 'version'}

            # 3. 只把真正有用的参数 update 进 case
            case.update(pure_kernel_params)

            # --- 后续精度比对逻辑保持不变 ---
            ref_case = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in case.items()}
            self.dispatch_call(solve_fn, case, is_cuda, params, has_kwargs)
            self.dispatch_call(self.spec.reference_impl, ref_case, False, set(), True)
            
            if not torch.allclose(case[out_n], ref_case[out_n], atol=atol, rtol=rtol):
                print(f"❌ {label} 精度检查失败！")
                # 过滤掉 Tensor 打印出具体的配置参数
                clean_cfg = {k: v for k, v in case.items() if not isinstance(v, torch.Tensor)}
                print(f"   使用的配置: {clean_cfg}")
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
                print(f"{'Shape':>20} | {'Best CFG':>12} | {'Latency':>10} | {'Throughput':>12}")
                print("-" * 65)
                
                for size_cfg in self.spec.x_vals:
                    best_tp, best_ms, best_label = 0.0, 0.0, ""
                    shape_label = "x".join([str(v) for v in size_cfg.values()])
                    
                    # 核心修改：如果是 cublas，直接运行，不走 configs 调优循环
                    if current_impl_name == "cublas":
                        p_case = self.spec.generate_performance_test(size_cfg)
                        # 注意：这里不用传任何 bx/by，dispatch_call 里的 .get(p, 0) 会处理
                        ms = self.measure_latency(solve_fn, p_case, is_cuda, params, has_kwargs)
                        tp = self.spec.get_throughput(p_case, ms)
                        best_tp, best_ms, best_label = tp, ms, "official"
                    else:
                        # 正常的内核调优循环
                        for config in configs:
                            cfg_version = config.get("version")
                            if cfg_version != current_impl_name:
                                continue

                            # 这里的 any 检查如果对 cublas 太严格也会导致跳过
                            if not has_kwargs and not any(k in params for k in config.keys()): 
                                continue 
                            
                            p_case = {**self.spec.generate_performance_test(size_cfg), **config}
                            ms = self.measure_latency(solve_fn, p_case, is_cuda, params, has_kwargs)
                            tp = self.spec.get_throughput(p_case, ms)
                            
                            if tp > best_tp:
                                best_tp, best_ms, best_label = tp, ms, "_".join([str(v) for k, v in config.items() if k != 'version'])
                    
                    if best_label:
                        self.logger.record(name, 'scaling', shape_label, best_tp)
                        print(f"{shape_label:>20s} | {best_label:>12s} | {best_ms:8.4f} ms | {best_tp:9.2f} GB/s")

            elif self.args.bench_mode == 'tuning':
                base_case = self.spec.generate_performance_test({})
                shape_desc = "x".join([str(v) for k, v in base_case.items() if isinstance(v, (int, float))])

                print(f"\n🔍 Tuning Analysis (Size: {shape_desc}):")
                print(f"{'Config':>15} | {'Latency':>10} | {'Throughput':>12}")
                print("-" * 45)

                for config in configs:
                    # Tuning 模式同样应用版本过滤
                    if current_impl_name != "cublas" and config.get("version") != current_impl_name:
                        continue

                    tuning_case = {**base_case, **config}
                    ms = self.measure_latency(solve_fn, tuning_case, is_cuda, params, has_kwargs)
                    tp = self.spec.get_throughput(tuning_case, ms)
                    
                    # 生成干净的 Label
                    label = "_".join([str(v) for k, v in config.items() if k != 'version'])
                    self.logger.record(name, 'tuning', label, tp)
                    print(f"{label:>15s} | {ms:8.4f} ms | {tp:9.2f} GB/s")