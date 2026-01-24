import torch, glob, os, ctypes, importlib.util, sys
from core.engine import KernelEngine
from core.benchmark import LabLogger

def run_lab(op_folder, epoch = 1000, warm_up = True, warm_epoch = 100):
    # 1. 加载配置 (与之前一致)
    cfg_path = os.path.join(os.getcwd(), 'operators', op_folder, 'test_cfg.py')
    spec_name = f"spec_{op_folder.replace('-', '_')}"
    spec_lib = importlib.util.spec_from_file_location(spec_name, cfg_path)
    test_cfg = importlib.util.module_from_spec(spec_lib)
    spec_lib.loader.exec_module(test_cfg)
    
    spec = test_cfg.OperatorSpec()
    logger = LabLogger()

    # --- 获取测试用例 ---
    func_cases = spec.generate_functional_test()
    perf_case = spec.generate_performance_test()

    # --- CUDA 测试部分 ---
    cu_files = glob.glob(f"operators/{op_folder}/cuda/*.cu")
    for cu_file in cu_files:
        print(f"🛠️  Compiling & Testing CUDA: {os.path.basename(cu_file)}")
        try:
            lib = KernelEngine.setup_cuda(cu_file)
            
            # A. 功能测试 (Correctness Check)
            all_pass = True
            for case in func_cases:
                args, types = spec.get_cuda_args(case) # <-- 传 case 进去了
                c_args = [ctypes.cast(a.data_ptr(), ctypes.c_void_p) if isinstance(a, torch.Tensor) else t(a) 
                          for a, t in zip(args, types)]
                
                case["C"].zero_()
                lib.solve(*c_args)
                
                ref = spec.reference(case["A"], case["B"], torch.empty_like(case["C"]), case["N"])
                if not torch.allclose(case["C"], ref, atol=spec.atol):
                    print(f"   - ❌ Func Fail: {case['name']}")
                    all_pass = False
                    break
            
            # B. 性能测试 (Benchmarking) - 只用 perf_case
            if all_pass:
                args, types = spec.get_cuda_args(perf_case)
                c_args = [ctypes.cast(a.data_ptr(), ctypes.c_void_p) if isinstance(a, torch.Tensor) else t(a) 
                          for a, t in zip(args, types)]
                
                # Warmup
                if warm_up:
                    for _ in range(warm_epoch): lib.solve(*c_args)
                    torch.cuda.synchronize()
                
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(epoch): lib.solve(*c_args)
                end.record()
                torch.cuda.synchronize()
                
                avg_ms = start.elapsed_time(end) / epoch
                logger.record(os.path.basename(cu_file), avg_ms, True)
                print(f"   - Status: ✅ PASS | Time: {avg_ms:.4f} ms")
            else:
                logger.record(os.path.basename(cu_file), 0, False)

        except Exception as e:
            print(f"   - 💥 CUDA Error: {str(e)}")

    # --- Triton 测试部分 ---
    triton_files = glob.glob(f"operators/{op_folder}/triton/*.py")
    for py_file in triton_files:
        print(f"🚀 Launching Triton: {os.path.basename(py_file)}")
        try:
            kernel = KernelEngine.setup_triton(py_file)
            
            # A. 功能测试
            all_pass = True
            for case in func_cases:
                t_args = spec.get_triton_args(case)
                grid = lambda meta: (torch.div(case["N"] + meta['BLOCK_SIZE'] - 1, meta['BLOCK_SIZE'], rounding_mode='floor'), )
                case["C"].zero_()
                kernel[grid](**t_args)
                
                ref = spec.reference(case["A"], case["B"], torch.empty_like(case["C"]), case["N"])
                if not torch.allclose(case["C"], ref, atol=spec.atol):
                    all_pass = False; break
            
            # B. 性能测试
            if all_pass:
                t_args = spec.get_triton_args(perf_case)
                grid = lambda meta: (torch.div(perf_case["N"] + meta['BLOCK_SIZE'] - 1, meta['BLOCK_SIZE'], rounding_mode='floor'), )
                
                if warm_up:
                    for _ in range(warm_epoch): kernel[grid](**t_args)
                    torch.cuda.synchronize()
                
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(epoch): kernel[grid](**t_args)
                end.record()
                torch.cuda.synchronize()
                
                avg_ms = start.elapsed_time(end) / epoch
                logger.record(os.path.basename(py_file), avg_ms, True)
                print(f"   - Status: ✅ PASS | Time: {avg_ms:.4f} ms")

        except Exception as e:
            print(f"   - 💥 Triton Error: {str(e)}")

    logger.plot(op_folder)


if __name__ == "__main__":
    import sys
    
    # 默认跑向量加法
    target_op = "01-vector-addition"
    
    # 如果命令行传了参数，比如 python run_lab.py 02-rms-norm
    if len(sys.argv) > 1:
        target_op = sys.argv[1]
        
    print(f"🔬 Starting Lab Experiment: {target_op}")
    run_lab(target_op,
            epoch=10000,
            warm_up=True,
            warm_epoch=100)