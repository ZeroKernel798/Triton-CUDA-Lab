import glob
import os
import argparse
import importlib.util
from utils.compiler import KernelEngine
from utils.logger import LabLogger
from utils.runner import BenchmarkRunner

def run_lab():
    parser = argparse.ArgumentParser(description="Triton-CUDA-Lab Runner")
    parser.add_argument("op", type=str, help="算子目录名 (如: 04-matrix-multiplication)")
    parser.add_argument("--mode", choices=['all', 'cuda', 'triton'], default='all')
    parser.add_argument("--epoch", type=int, default=1000, help="测试迭代次数")
    parser.add_argument("--warmup", type=int, default=10, help="预热迭代次数")
    parser.add_argument("--bench_mode", choices=['scaling', 'tuning'], default='scaling')
    parser.add_argument("--metric", choices=['bw', 'flops', 'ms', 'all'], default='all', 
                    help="绘图指标: bw, flops, ms (时间), 或 all (全部)")
    # Profiling 开关
    parser.add_argument("--profile", action="store_true", help="开启采样模式")
    parser.add_argument("--ncu", action="store_true", help="切换至 ncu 模式 (需配合 --profile)")
    args = parser.parse_args()

    # 加载算子配置文件
    cfg_path = os.path.join(os.getcwd(), 'operators', args.op, 'test_cfg.py')
    if not os.path.exists(cfg_path):
        print(f"❌ 找不到测试配置文件: {cfg_path}"); return

    spec_lib = importlib.util.spec_from_file_location("spec", cfg_path)
    test_cfg = importlib.util.module_from_spec(spec_lib)
    spec_lib.loader.exec_module(test_cfg)
    spec = test_cfg.OperatorSpec()
    
    # 初始化环境
    logger = LabLogger()
    runner = BenchmarkRunner(spec, args, logger)

    # 如果是 Profile 模式，确保报告输出目录存在
    if args.profile:
        os.makedirs("build/reports", exist_ok=True)
        print("\n" + "!"*60)
        print("📸 PROFILE 模式启动: 仅进行功能校验与单次硬件采样")
        print("!"*60)

    # 运行 CUDA 算子
    if args.mode in ['all', 'cuda']:
        # 寻找目录下所有的 .cu 文件
        for cu in sorted(glob.glob(f"operators/{args.op}/cuda/*.cu")):
            file_name = os.path.basename(cu)
            try:
                # KernelEngine.setup_cuda 内部已处理 -lineinfo
                lib = KernelEngine.setup_cuda(cu)
                if hasattr(lib, 'solve'): 
                    runner.run_benchmark(lib.solve, file_name, True)
            except Exception as e: 
                print(f"❌ 运行错误 [{file_name}]: {e}")

    # 运行 Triton 算子
    # profile 模式通常针对 CUDA，但这里逻辑也支持 Triton 的 nsys 采样
    if args.mode in ['all', 'triton']:
        for py_file in sorted(glob.glob(f"operators/{args.op}/triton/*.py")):
            file_name = os.path.basename(py_file)
            try:
                spec_t = importlib.util.spec_from_file_location("triton_mod", py_file)
                triton_mod = importlib.util.module_from_spec(spec_t)
                spec_t.loader.exec_module(triton_mod)
                if hasattr(triton_mod, 'solve'): 
                    runner.run_benchmark(triton_mod.solve, f"{file_name}", False)
            except Exception as e: 
                print(f"❌ 运行错误 [{file_name}]: {e}")

    # 结果处理
    if args.profile:
        print(f"\n✨ Profiling 任务结束。")
    else:
        base_case = spec.generate_performance_test({})
        shape_str = "x".join([str(v) for v in base_case.values() if isinstance(v, (int, float))])

        # 💡 确定需要绘制的指标列表
        if args.metric == 'all':
            metrics_to_plot = ['bw', 'flops', 'ms']
        else:
            metrics_to_plot = [args.metric]
        
        for m in metrics_to_plot:
            logger.plot(args.op, args.bench_mode, metric_type=m, current_shape=shape_str)
            

if __name__ == "__main__":
    run_lab()