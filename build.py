import os
import glob
import argparse
import multiprocessing
import importlib.util
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils.compiler import compiler, clean_build

def find_all_ops():
    """获取 operators 目录下所有的算子目录"""
    base_path = "operators"
    if not os.path.exists(base_path): return []
    return [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

def find_op_dir(op_name):
    """支持模糊匹配算子目录"""
    if not op_name: return []
    all_ops = find_all_ops()
    matched = [d for d in all_ops if op_name.lower() in d.lower()]
    return matched

def get_spec_instance(mod):
    """从模块中安全获取具体的算子 Spec 实例"""
    for name, obj in mod.__dict__.items():
        if isinstance(obj, type) and name.endswith('Spec') and name != 'BaseOperatorSpec':
            return obj()
    raise ImportError("未找到具体的 Spec 子类实现")

def compile_worker(op_folder, cu_file, cfg, spec_path):
    try:
        spec_lib = importlib.util.spec_from_file_location("spec", spec_path)
        mod = importlib.util.module_from_spec(spec_lib)
        spec_lib.loader.exec_module(mod)
        spec = get_spec_instance(mod)
        macros = spec.get_macros(cfg)
        compiler.compile(op_folder, cu_file, macros)
        return True, f"{os.path.basename(cu_file)} [{cfg.get('version')}]"
    except Exception as e:
        return False, str(e)

def main():
    parser = argparse.ArgumentParser(description="V2 并行构建引擎")
    parser.add_argument("--op", type=str, help="模糊匹配算子名。不加则全量编译。")
    parser.add_argument("-j", type=int, default=8, help="并行线程数")
    parser.add_argument("--clean", action="store_true", help="清理构建产物")
    args = parser.parse_args()

    # 确定目标算子
    if args.op:
        target_ops = find_op_dir(args.op)
        if not target_ops:
            print(f"❌ 未发现匹配算子: {args.op}")
            return
    else:
        target_ops = find_all_ops()

    # 处理清理逻辑 
    if args.clean:
        if args.op:
            for op in target_ops:
                clean_build(op)
        else:
            clean_build()
            print("✨ 全量构建环境已清理。")
            return

    # 编译实现 会查找config中的版本 只有特定版本被编译
    tasks = []
    for op in target_ops:
        spec_path = f"operators/{op}/test_cfg.py"
        if not os.path.exists(spec_path): continue
        
        spec_lib = importlib.util.spec_from_file_location("spec_loader", spec_path)
        mod = importlib.util.module_from_spec(spec_lib)
        spec_lib.loader.exec_module(mod)
        spec = get_spec_instance(mod)

        for cu in glob.glob(f"operators/{op}/cuda/*.cu"):
            ver = os.path.basename(cu).replace(".cu", "")
            target_cfgs = [c for c in spec.cuda_tuning_configs if c.get("version") == ver]
            for cfg in target_cfgs:
                tasks.append((op, cu, cfg, spec_path))

    if not tasks:
        print("⚠️ 未发现可编译的配置组合，请检查 spec 定义。")
        return

    # 并行编译
    mode_str = f"针对算子 [{args.op}]" if args.op else "全量"
    print(f"🚀 启动{mode_str}编译，共 {len(tasks)} 个任务...")
    
    with ProcessPoolExecutor(max_workers=args.j) as pool:
        futures = [pool.submit(compile_worker, *t) for t in tasks]
        for i, f in enumerate(as_completed(futures)):
            ok, msg = f.result()
            print(f"[{i+1}/{len(tasks)}] {'✅' if ok else '❌'} {msg}")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()