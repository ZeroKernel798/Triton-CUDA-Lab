import torch
import glob
import os
import argparse
import importlib.util
import sys
import re        
import inspect   
from utils.compiler import KernelEngine
from utils.logger import LabLogger

def run_lab():
    parser = argparse.ArgumentParser(description="Triton-CUDA-Lab Runner")
    parser.add_argument("op", type=str, help="算子目录名")
    parser.add_argument("--mode", choices=['all', 'cuda', 'triton'], default='all')
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--bench_mode", choices=['scaling', 'tuning'], default='scaling')
    args = parser.parse_args()

    cfg_path = os.path.join(os.getcwd(), 'operators', args.op, 'test_cfg.py')
    if not os.path.exists(cfg_path):
        print(f"找不到测试文件: {cfg_path}"); return

    spec_lib = importlib.util.spec_from_file_location("spec", cfg_path)
    test_cfg = importlib.util.module_from_spec(spec_lib)
    spec_lib.loader.exec_module(test_cfg)
    spec = test_cfg.OperatorSpec()
    
    logger = LabLogger()

    def run_benchmark(solve_fn, name, is_cuda):
        print(f"Testing {name} | Mode: {args.bench_mode}...")
        
        # 1. 探测 Kernel 核心参数与特性
        has_kwargs = False
        try:
            target_fn = solve_fn.fn if hasattr(solve_fn, 'fn') else solve_fn
            sig = inspect.signature(target_fn)
            kernel_params = set(sig.parameters.keys())
            has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        except Exception:
            doc = solve_fn.__doc__ or ""
            match = re.search(r"solve\((.*?)\)", doc)
            kernel_params = set([p.split(':')[0].strip() for p in match.group(1).split(',') if p.strip()]) if match else set()

        atol = getattr(spec, "atol", 1e-5)
        rtol = getattr(spec, "rtol", 1e-5)
        output_name = getattr(spec, "output_name", "C")
            
        # 2. 智能分发器
        def dispatch_call(fn, p_case, is_cuda_target):
            if is_cuda_target:
                try:
                    sig = inspect.signature(fn)
                    req_params = list(sig.parameters.keys())
                except ValueError:
                    doc = fn.__doc__ or ""
                    match = re.search(r"\((.*)\)", doc)
                    # 建议优化后的那一行
                    req_params = [p.split(':')[0].split('=')[0].strip() for p in match.group(1).split(',') if p.strip()]

                args_to_pass = {}
                for p_name in req_params:
                    if p_name in p_case: args_to_pass[p_name] = p_case[p_name]
                    elif p_name == "Ne": args_to_pass[p_name] = p_case.get('N', 0) ** 2
                    elif p_name == "block_size": args_to_pass[p_name] = 256
                    else: raise KeyError(f"CUDA 算子缺失参数: {p_name}")

                try: return fn(**args_to_pass)
                except TypeError: return fn(*[args_to_pass[p] for p in req_params])
            else:
                # Triton 逻辑：有 **kwargs 则全传，否则按名过滤
                if has_kwargs: return fn(**p_case)
                else:
                    sig = inspect.signature(fn)
                    filtered = {k: v for k, v in p_case.items() if k in sig.parameters}
                    return fn(**filtered)
            
        # 3. 验证逻辑 (同你之前代码)
        if hasattr(spec, "generate_example_test"):
            test_case = spec.generate_example_test()
            dispatch_call(solve_fn, test_case, is_cuda)
            ref_case = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in test_case.items()}
            dispatch_call(spec.reference_impl, ref_case, False)
            if not torch.allclose(test_case[output_name], ref_case[output_name], atol=atol, rtol=rtol):
                print(f"{name} 精度检查失败！"); return

        # 4. 辅助测量
        def measure_latency(p_case):
            for _ in range(args.warmup): dispatch_call(solve_fn, p_case, is_cuda)
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.epoch): dispatch_call(solve_fn, p_case, is_cuda)
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / args.epoch

        # 5. 性能测试主循环
        configs = spec.cuda_tuning_configs if is_cuda else spec.tuning_configs
        
        if args.bench_mode == 'scaling':
            for n in spec.x_vals:
                best_throughput, best_ms, best_label = 0.0, 0.0, ""
                for config in configs:
                    if not has_kwargs and not any(k in kernel_params for k in config.keys()):
                        continue 
                    perf_case = {**spec.generate_performance_test(n), **config}
                    ms = measure_latency(perf_case)
                    tp = spec.get_throughput(n, ms)
                    if tp > best_throughput:
                        best_throughput, best_ms, best_label = tp, ms, "_".join([f"{v}" for v in config.values()])
                if best_label == "": continue
                logger.record(name, 'scaling', n, best_throughput)
                print(f"Size {n:10d} | Best CFG: {best_label:10s} | {best_ms:8.4f} ms | {best_throughput:8.2f} GB/s")

        elif args.bench_mode == 'tuning':
            base_case = spec.generate_performance_test()
            # ✅ 修复后的缩进逻辑：确保所有测试都在 config 循环内执行
            for config in configs:
                if not has_kwargs and not any(k in kernel_params for k in config.keys()):
                    continue
                tuning_case = {**base_case, **config}
                ms = measure_latency(tuning_case)
                tp = spec.get_throughput(ms=ms)
                label = "_".join([f"{v}" for v in config.values()])
                logger.record(name, 'tuning', label, tp)
                print(f"Config {label:15s} | {ms:8.2f} ms | {tp:8.2f} GB/s")

    # 6. 加载并运行算子 (保持不变)
    if args.mode in ['all', 'cuda']:
        for cu in sorted(glob.glob(f"operators/{args.op}/cuda/*.cu")):
            file_name = os.path.basename(cu)
            try:
                lib = KernelEngine.setup_cuda(cu)
                if hasattr(lib, 'solve'): run_benchmark(lib.solve, file_name, True)
            except Exception as e: print(f"运行错误 [{file_name}]: {e}")

    if args.mode in ['all', 'triton']:
        for py_file in sorted(glob.glob(f"operators/{args.op}/triton/*.py")):
            file_name = os.path.basename(py_file)
            try:
                spec_t = importlib.util.spec_from_file_location("triton_mod", py_file)
                triton_mod = importlib.util.module_from_spec(spec_t)
                spec_t.loader.exec_module(triton_mod)
                if hasattr(triton_mod, 'solve'): run_benchmark(triton_mod.solve, f"Triton_{file_name}", False)
            except Exception as e: print(f"运行错误 [{file_name}]: {e}")

    logger.plot(args.op, args.bench_mode)

if __name__ == "__main__":
    run_lab()