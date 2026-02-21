import glob
import os
import argparse
import importlib.util
from utils.compiler import KernelEngine
from utils.logger import LabLogger
from utils.runner import BenchmarkRunner

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
    runner = BenchmarkRunner(spec, args, logger)

    if args.mode in ['all', 'cuda']:
        for cu in sorted(glob.glob(f"operators/{args.op}/cuda/*.cu")):
            file_name = os.path.basename(cu)
            try:
                lib = KernelEngine.setup_cuda(cu)
                if hasattr(lib, 'solve'): runner.run_benchmark(lib.solve, file_name, True)
            except Exception as e: print(f"运行错误 [{file_name}]: {e}")

    if args.mode in ['all', 'triton']:
        for py_file in sorted(glob.glob(f"operators/{args.op}/triton/*.py")):
            file_name = os.path.basename(py_file)
            try:
                spec_t = importlib.util.spec_from_file_location("triton_mod", py_file)
                triton_mod = importlib.util.module_from_spec(spec_t)
                spec_t.loader.exec_module(triton_mod)
                if hasattr(triton_mod, 'solve'): runner.run_benchmark(triton_mod.solve, f"Triton_{file_name}", False)
            except Exception as e: print(f"运行错误 [{file_name}]: {e}")

    logger.plot(args.op, args.bench_mode)

if __name__ == "__main__":
    run_lab()