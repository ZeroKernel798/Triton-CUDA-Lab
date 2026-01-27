import torch
import glob
import os
import ctypes
import importlib.util
import sys
from core.engine import KernelEngine
from core.benchmark import LabLogger

def run_lab(op_folder, epoch=1000):
    # 1. 动态加载算子配置
    cfg_path = os.path.join(os.getcwd(), 'operators', op_folder, 'test_cfg.py')
    spec_lib = importlib.util.spec_from_file_location("spec", cfg_path)
    test_cfg = importlib.util.module_from_spec(spec_lib)
    spec_lib.loader.exec_module(test_cfg)
    spec = test_cfg.OperatorSpec()
    
    logger = LabLogger()
    signature = spec.get_solve_signature()

    # --- 核心：参数准备逻辑 ---
    def prepare_args(case, is_cuda):
        if is_cuda:
            args = []
            for name, (arg_type, _) in signature.items():
                val = case[name]
                if arg_type == ctypes.POINTER(ctypes.c_float):
                    args.append(ctypes.cast(val.data_ptr(), ctypes.c_void_p))
                else:
                    args.append(ctypes.c_int(int(val)))
            return args
        else:
            return {k: case[k] for k in signature.keys()}

    # --- 执行器：集成 Example Test 逻辑 ---
    def run_benchmark(solve_fn, name, is_cuda):
        print(f"   - Testing: {name}")

        def check_accuracy(case, case_label):
            # 1. 动态清零输出
            for key, (_, direction) in signature.items():
                if direction == "out":
                    case[key].zero_()
            
            # 2. 运行实现版本
            args = prepare_args(case, is_cuda)
            if is_cuda: solve_fn(*args)
            else: solve_fn(**args)
            
            # 3. 运行参考实现 (克隆数据防止污染)
            ref_case = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in case.items()}
            # 仅传递参考实现需要的参数（过滤掉类似 'name' 这种元数据）
            ref_inputs = {k: ref_case[k] for k in ref_case if k in signature}
            spec.reference_impl(**ref_inputs)
            
            # 4. 对比结果
            for key, (_, direction) in signature.items():
                if direction == "out":
                    if not torch.allclose(case[key], ref_case[key], atol=spec.atol):
                        print(f"     ❌ Accuracy Fail [{case_label}] on output: {key}")
                        return False
            return True

        # --- A. Example Test (样题测试) ---
        if hasattr(spec, 'generate_example_test'):
            example_case = spec.generate_example_test()
            if not check_accuracy(example_case, "Example"):
                return # 样题不过，直接下一位

        # --- B. Functional Tests (功能测试) ---
        for case in spec.generate_functional_test():
            case_name = case.get('name', 'Functional')
            if not check_accuracy(case, case_name):
                return

        # --- C. Performance (性能跑分) ---
        perf_case = spec.generate_performance_test()
        args = prepare_args(perf_case, is_cuda)
        
        # Warmup
        for _ in range(10):
            if is_cuda: solve_fn(*args)
            else: solve_fn(**args)
            
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(epoch):
            if is_cuda: solve_fn(*args)
            else: solve_fn(**args)
        end.record()
        torch.cuda.synchronize()
        
        avg_ms = start.elapsed_time(end) / epoch
        logger.record(name, avg_ms, True)
        print(f"     ✅ PASS | {avg_ms:.4f} ms")

    # 扫描与执行
    for cu in glob.glob(f"operators/{op_folder}/cuda/*.cu"):
        print(f"🛠️  CUDA: {os.path.basename(cu)}")
        lib = KernelEngine.setup_cuda(cu)
        run_benchmark(lib.solve, os.path.basename(cu), True)

    for py in glob.glob(f"operators/{op_folder}/triton/*.py"):
        print(f"🚀 Triton: {os.path.basename(py)}")
        module = KernelEngine.setup_triton(py)
        run_benchmark(module.solve, os.path.basename(py), False)

    logger.plot(op_folder)

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "04-softmax-attention"
    run_lab(target)