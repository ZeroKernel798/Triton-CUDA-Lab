import os
import importlib.util

def find_matched_ops(pattern: str):
    """支持模糊匹配算子目录名，并按名称排序"""
    base = "operators"
    if not os.path.exists(base):
        return []
    all_ops = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    matched = [d for d in all_ops if pattern.lower() in d.lower()]
    return sorted(matched)

def get_spec(op_folder: str):
    """获取算子 Spec 实例，支持多种命名约定"""
    for filename in ["test_cfg.py", "spec.py"]:
        path = os.path.join("operators", op_folder, filename)
        if os.path.exists(path):
            spec_name = f"spec_{op_folder}"
            spec_lib = importlib.util.spec_from_file_location(spec_name, path)
            mod = importlib.util.module_from_spec(spec_lib)
            spec_lib.loader.exec_module(mod)
            for name, obj in mod.__dict__.items():
                if isinstance(obj, type) and name.endswith('Spec') and name != 'BaseOperatorSpec':
                    return obj()
    return None

def get_kernel_files(op_folder: str):
    """
    核心优化：扫描不同后端的内核文件
    返回格式: [{"ver": "float4", "path": "...", "type": "cuda"}, ...]
    """
    kernels = []
    base_path = os.path.join("operators", op_folder)
    
    # 定义扫描规则：目录名 -> (后缀, 类型标签)
    scan_rules = {
        "cuda": (".cu", "cuda"),
        "triton": (".py", "triton")
    }

    for sub_dir, (ext, k_type) in scan_rules.items():
        dir_path = os.path.join(base_path, sub_dir)
        if not os.path.exists(dir_path):
            continue
            
        for f in sorted(os.listdir(dir_path)):
            if f.endswith(ext) and not f.startswith("__"):
                kernels.append({
                    "ver": f.replace(ext, ""),  # 版本号：如 float4
                    "path": os.path.join(dir_path, f), 
                    "type": k_type              # 类型：cuda 或 triton
                })
                
    return kernels