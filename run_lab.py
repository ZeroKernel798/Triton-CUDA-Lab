import torch
import glob
import os
import argparse
import importlib.util
import sys
from core.engine import KernelEngine
from core.benchmark import LabLogger

def run_lab():
    parser = argparse.ArgumentParser(description="Triton-CUDA-Lab Runner (Pure Performance)")
    parser.add_argument("op", type=str, help="算子目录名")
    parser.add_argument("--mode", choices=['all', 'cuda', 'triton'], default='all')
    parser.add_argument("--epoch", type=int, default=1000)
    args = parser.parse_args()

    # 1. 加载算子配置
    cfg_path = os.path.join(os.getcwd(), 'operators', args.op, 'test_cfg.py')
    if not os.path.exists(cfg_path):
        print(f"❌ 找不到配置文件: {cfg_path}"); return

    spec_lib = importlib.util.spec_from_file_location("spec", cfg_path)
    test_cfg = importlib.util.module_from_spec(spec_lib)
    spec_lib.loader.exec_module(test_cfg)
    spec = test_cfg.OperatorSpec()
    
    logger = LabLogger()
    arg_names = spec.get_solve_signature()

    def run_benchmark(solve_fn, name, is_cuda):
        print(f"🚀 Testing {name}...")

        # --- A. 准备数据 ---
        test_case = spec.generate_example_test()
        p_args = [test_case[k] for k in arg_names]
        
        # --- B. 自动识别输出张量名 ---
        # 优先级：spec 显式定义 -> "C" -> "output" -> "O" -> 字典中第一个 Tensor
        output_name = getattr(spec, "output_name", None)
        if output_name is None:
            for candidate in ["C", "output", "O"]:
                if candidate in test_case:
                    output_name = candidate; break
            if output_name is None:
                output_name = next(k for k, v in test_case.items() if isinstance(v, torch.Tensor))

        # 运行待测版本
        solve_fn(*p_args) if is_cuda else solve_fn(**test_case)
        
        # 运行参考实现
        ref_case = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in test_case.items()}
        spec.reference_impl(**ref_case)
        
        # --- C. 精度比对 (自适应) ---
        actual = test_case[output_name]
        expected = ref_case[output_name]
        
        # 使用 spec 中定义的 atol 和 rtol
        atol = getattr(spec, "atol", 1e-5)
        rtol = getattr(spec, "rtol", 1e-5)
        
        if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
            max_diff = (actual - expected).abs().max().item()
            print(f"      ❌ {name} 精度检查失败！(变量名: {output_name}, Max Diff: {max_diff})")
            return

        # --- D. 性能跑分 ---
        perf_case = spec.generate_performance_test()
        perf_args = [perf_case[k] for k in arg_names]
        
        for _ in range(10): 
            solve_fn(*perf_args) if is_cuda else solve_fn(**perf_case)
        
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(args.epoch):
            solve_fn(*perf_args) if is_cuda else solve_fn(**perf_case)
        end.record()
        
        torch.cuda.synchronize()
        
        avg_ms = start.elapsed_time(end) / args.epoch
        logger.record(name, avg_ms)
        print(f"      ✅ {name} PASS | 耗时: {avg_ms:.4f} ms")

    # 2. 执行流程 (逻辑保持不变)
    if args.mode in ['all', 'cuda']:
        cuda_path = f"operators/{args.op}/cuda/*.cu"
        for cu in sorted(glob.glob(cuda_path)):
            file_name = os.path.basename(cu)
            try:
                lib = KernelEngine.setup_cuda(cu)
                run_benchmark(lib.solve, file_name, True)
            except Exception as e:
                print(f"      ❌ 运行错误 [{file_name}]: {e}")

    if args.mode in ['all', 'triton']:
        # TODO: 接入 Triton 加载逻辑
        pass

    logger.plot(args.op)

if __name__ == "__main__":
    run_lab()