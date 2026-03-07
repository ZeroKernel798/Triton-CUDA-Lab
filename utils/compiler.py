import os
import sys
import torch
import hashlib
import importlib
import shutil
import time
import torch.cuda.nvtx as nvtx
from torch.utils.cpp_extension import load

class KernelEngine:
    @staticmethod
    def get_gpu_info():
        if not torch.cuda.is_available():
            return "no_cuda", "0.0", 0, 0
        major, minor = torch.cuda.get_device_capability()
        return f"sm_{major}{minor}", f"{major}.{minor}", major, minor

    @staticmethod
    def get_md5(file_path):
        if not os.path.exists(file_path): return ""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    @staticmethod
    def profile_scope(enabled=False, label="Kernel"):
        """ 用于 Nsight Profiling 的快门控制器 """
        class ProfileContext:
            def __enter__(self):
                if enabled:
                    torch.cuda.synchronize() # 采样前对齐
                    torch.cuda.profiler.start() # 开启硬件计数器
                    nvtx.range_push(label) # 在时间轴上打标

            def __exit__(self, type, value, traceback):
                if enabled:
                    torch.cuda.synchronize()
                    nvtx.range_pop()
                    torch.cuda.profiler.stop() # 停止采样
        return ProfileContext()

    @staticmethod
    def setup_cuda(cu_file, force_recompile=False):
        """编译单个 CUDA 算子"""
        # --- 策略：每个进程只管一个文件，强制 Ninja 单线程防止资源踩踏 ---
        os.environ["MAX_JOBS"] = "1" 
        arch_sm, arch_list, major_v, minor_v = KernelEngine.get_gpu_info()
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list

        abs_path = os.path.abspath(cu_file)
        path_parts = abs_path.split(os.sep)
        op_folder = path_parts[-3] if len(path_parts) > 3 else "default"
        file_base = os.path.basename(cu_file).replace('.cu', '')
        
        module_name = f"{op_folder}_{file_base}_{arch_sm}".replace('-', '_').replace('.', '_')
        build_dir = os.path.join(os.getcwd(), "build", op_folder, file_base, arch_sm)
        os.makedirs(build_dir, exist_ok=True)
        
        if build_dir not in sys.path:
            sys.path.append(build_dir)

        # --- 自动破锁逻辑：如果锁文件存在且超过 15 秒没动静，视为僵尸锁 ---
        for lock_name in ["lock", ".ninja_lock"]:
            l_path = os.path.join(build_dir, lock_name)
            if os.path.exists(l_path):
                if time.time() - os.path.getmtime(l_path) > 15:
                    try: os.remove(l_path)
                    except: pass

        # --- MD5 增量编译检查 ---
        cu_md5 = KernelEngine.get_md5(abs_path)
        md5_file = os.path.join(build_dir, "source.md5")
        
        if not force_recompile and os.path.exists(md5_file):
            with open(md5_file, 'r') as f:
                if f.read().strip() == cu_md5:
                    try:
                        return importlib.import_module(module_name)
                    except: pass

        # --- 执行编译 ---
        # nvcc_threads 设为 4-8 即可，由编译器内部优化单个文件
        module = load(
            name=module_name,
            sources=[abs_path],
            extra_cflags=['-O3', '-std=c++17'],
            extra_cuda_cflags=[
                '-lineinfo',
                '-O3', '--use_fast_math', '-allow-unsupported-compiler',
                '--threads', '4', 
                '-gencode', f'arch=compute_{major_v}{minor_v},code=sm_{major_v}{minor_v}'
            ],
            build_directory=build_dir,
            verbose=False
        )

        # 写入缓存记录
        try:
            with open(md5_file, 'w') as f:
                f.write(cu_md5)
        except: pass

        return module

def clean_build():
    build_path = os.path.join(os.getcwd(), "build")
    if os.path.exists(build_path):
        print(f"🧹 清理构建产物: {build_path}")
        shutil.rmtree(build_path, ignore_errors=True)