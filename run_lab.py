import torch
import glob
import os
import argparse
import importlib.util
import sys
from utils.complier import KernelEngine
from utils.logger import LabLogger

def run_lab():
    # 参数配置
    parser = argparse.ArgumentParser(description="Triton-CUDA-Lab Runner (Pure Performance)")
    parser.add_argument("op", type=str, help="算子目录名")
    parser.add_argument("--mode", choices=['all', 'cuda', 'triton'], default='all')
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--bench_mode", choices=['scaling', 'tuning'], default='scaling')
    args = parser.parse_args()

    # 动态加载测试内容 即每个算子文件夹内的test_cfg.py
    cfg_path = os.path.join(os.getcwd(), 'operators', args.op, 'test_cfg.py')
    if not os.path.exists(cfg_path):
        print(f"找不到测试文件: {cfg_path}"); return

    spec_lib = importlib.util.spec_from_file_location("spec", cfg_path)
    test_cfg = importlib.util.module_from_spec(spec_lib)
    spec_lib.loader.exec_module(test_cfg)
    spec = test_cfg.OperatorSpec()
    
    # 初始化记录器 统计算子性能
    logger = LabLogger()
    # 获取算子参数列表
    arg_names = spec.get_solve_signature()

    # 执行测试过程的函数
    def run_benchmark(solve_fn, name, is_cuda):
        print(f"Testing {name} | Mode: {args.bench_mode}...")

        atol = getattr(spec, "atol", 1e-5)
        rtol = getattr(spec, "rtol", 1e-5)
        output_name = getattr(spec, "output_name", "C")
            
        # 定义智能分发逻辑 
        def dispatch_call(fn, p_case, is_cuda_target):
            if is_cuda_target:
                # 获取 C++ 需要的参数顺序
                # 这里的重点是：如果 p_case 里没传 block_size，得给它一个 spec 里的默认值
                p_args = []
                for k in arg_names:
                    if k in p_case:
                        p_args.append(p_case[k])
                    elif k == "block_size":
                        p_args.append(256) # 兜底默认值
                    else:
                        raise KeyError(f"算子参数缺失: {k}")
                return fn(*p_args)
            else:
                # Triton 逻辑
                import inspect
                sig = inspect.signature(fn)
                filtered_case = {k: v for k, v in p_case.items() if k in sig.parameters}
                return fn(**filtered_case)
            

        # 最小闭环逻辑验证
        if hasattr(spec, "generate_example_test"):
            test_case = spec.generate_example_test()
            
            # 使用智能分发调用待测版本
            dispatch_call(solve_fn, test_case, is_cuda)
            
            # 运行参考实现 
            ref_case = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in test_case.items()}
            dispatch_call(spec.reference_impl, ref_case, False)
            
            if not torch.allclose(test_case[output_name], ref_case[output_name], atol=atol, rtol=rtol):
                print(f"{name} 精度检查失败！")
                return

        # 特殊输入测试 
        if hasattr(spec, "generate_functional_test"):
            for i, test_case in enumerate(spec.generate_functional_test()):
                dispatch_call(solve_fn, test_case, is_cuda)
                ref_case = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in test_case.items()}
                dispatch_call(spec.reference_impl, ref_case, False)
                
                if not torch.allclose(test_case[output_name], ref_case[output_name], atol=atol, rtol=rtol):
                    print(f"Functional Test #{i} FAILED!")
                    return

        # 辅助性能测量逻辑
        def measure_latency(p_case):
            # Warmup
            for _ in range(args.warmup):
                dispatch_call(solve_fn, p_case, is_cuda)
            
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.epoch):
                dispatch_call(solve_fn, p_case, is_cuda)
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / args.epoch

        # 性能测试主循环 (Scaling/Tuning) 
        if not hasattr(spec, "generate_performance_test"): return

        if args.bench_mode == 'scaling':
            for n in spec.x_vals:
                perf_case = spec.generate_performance_test(n)
                avg_ms = measure_latency(perf_case)
                throughput = spec.get_throughput(n, avg_ms)
                logger.record(name, 'scaling', n, throughput)
                print(f"Size {n:10d} | {avg_ms:8.4f} ms | {throughput:8.2f} GB/s")

        elif args.bench_mode == 'tuning':
            # 这里设置一下想要测试的数据规模 然后用不同的核函数配置去测试 不设置就是test_cfg.py默认的
            # fixed_n = 1024 * 1024 * 16 
            # base_case = spec.generate_performance_test(fixed_n)
            base_case = spec.generate_performance_test()
            configs = spec.cuda_tuning_configs if is_cuda else spec.tuning_configs
            for config in configs:
                tuning_case = {**base_case, **config}
                avg_ms = measure_latency(tuning_case)
                # throughput = spec.get_throughput(n=fixed_n, ms=avg_ms)
                throughput = spec.get_throughput(ms=avg_ms)
                label = "_".join([f"{v}" for v in config.values()])
                logger.record(name, 'tuning', label, throughput)
                print(f"Config {label:15s} | {avg_ms:8.2f} ms | {throughput:8.2f} GB/s")

    # cuda的处理逻辑
    if args.mode in ['all', 'cuda']:
        cuda_path = f"operators/{args.op}/cuda/*.cu"
        for cu in sorted(glob.glob(cuda_path)):
            file_name = os.path.basename(cu)
            try:
                lib = KernelEngine.setup_cuda(cu)
                if hasattr(lib, 'solve'):
                    run_benchmark(lib.solve, file_name, True)
                else:
                    print(f"跳过 [{file_name}]: 在 PYBIND11_MODULE 中未发现 solve 函数")
            except Exception as e:
                print(f"运行错误 [{file_name}]: {e}")

    # triton的处理逻辑
    if args.mode in ['all', 'triton']:
        triton_path = f"operators/{args.op}/triton/*.py"
        for py_file in sorted(glob.glob(triton_path)):
            file_name = os.path.basename(py_file)   
            try:
                spec_triton = importlib.util.spec_from_file_location("triton_mod", py_file)
                triton_mod = importlib.util.module_from_spec(spec_triton)
                spec_triton.loader.exec_module(triton_mod)
            
                if hasattr(triton_mod, 'solve'):
                    run_benchmark(triton_mod.solve, f"Triton_{file_name}", is_cuda=False)
                else:
                    print(f"跳过 [{file_name}]: 未找到 solve 函数")
            except Exception as e:
                print(f"运行错误 [{file_name}]: {e}")

    logger.plot(args.op, args.bench_mode)

if __name__ == "__main__":
    run_lab()