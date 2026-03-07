import os
import sys
import torch
import hashlib
import importlib
import shutil
from torch.utils.cpp_extension import load

class KernelEngine:
    @staticmethod
    def get_gpu_info():
        if not torch.cuda.is_available():
            return "no_cuda", "0.0"
        major, minor = torch.cuda.get_device_capability()
        return f"sm_{major}{minor}", f"{major}.{minor}"

    @staticmethod
    def get_md5(file_path):
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    @staticmethod
    def init_performance_strategy():
        """
        根据硬件环境动态调整并行策略。
        Jetson: 采用严苛的内存保护，防止 OOM。
        PC/Server: 解除封印，拉满性能。
        """
        arch_sm, arch_list = KernelEngine.get_gpu_info()
        cpu_count = os.cpu_count() or 1
        is_jetson = os.path.exists("/etc/nv_tegra_release") or "tegra" in arch_sm.lower()
        
        # 获取物理内存总大小 (GB)
        total_mem_gb = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024**3)
        
        if is_jetson:
            # --- Jetson 严苛策略 ---
            max_jobs = max(2, min(4, int(total_mem_gb / 3)))
            nvcc_threads = 4
            
            print(f"🐢 [Jetson Mode] 🛡️ 内存保护已开启: RAM={total_mem_gb:.1f}GB")
            print(f"   策略: Jobs={max_jobs} (基于 1Job/3GB), Threads={nvcc_threads}")
        else:
            # --- 独显/服务器模式 ---
            max_jobs = cpu_count
            nvcc_threads = 8
            print(f"🚀 [PC Mode] 性能全开: Using {max_jobs} CPU cores.")
        
        os.environ["MAX_JOBS"] = str(max_jobs)
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list
        return nvcc_threads

    @staticmethod
    def setup_cuda(cu_file, nvcc_threads=4, force_recompile=False):
        abs_path = os.path.abspath(cu_file)
        path_parts = abs_path.split(os.sep)
        # 获取算子目录名
        op_folder = path_parts[-3] if len(path_parts) > 3 else "default"
        file_base = os.path.basename(cu_file).replace('.cu', '')
        
        arch_sm, _ = KernelEngine.get_gpu_info()
        module_name = f"{op_folder}_{file_base}_{arch_sm}".replace('-', '_').replace('.', '_')
        build_dir = os.path.join(os.getcwd(), "build", op_folder, file_base, arch_sm)
        os.makedirs(build_dir, exist_ok=True)
        
        if build_dir not in sys.path:
            sys.path.append(build_dir)

        cu_md5 = KernelEngine.get_md5(abs_path)
        md5_file = os.path.join(build_dir, "source.md5")
        if not force_recompile and os.path.exists(md5_file):
            with open(md5_file, 'r') as f:
                if f.read() == cu_md5:
                    try:
                        return importlib.import_module(module_name)
                    except: pass

        major_v, minor_v = arch_sm[3], arch_sm[4]
        return load(
            name=module_name,
            sources=[abs_path],
            extra_cflags=['-O3', '-std=c++17'],
            extra_cuda_cflags=[
                '-O3', '--use_fast_math', '-allow-unsupported-compiler',
                '--threads', str(nvcc_threads), 
                '-gencode', f'arch=compute_{major_v}{minor_v},code=sm_{major_v}{minor_v}'
            ],
            build_directory=build_dir,
            verbose=False
        )

def clean_build():
    build_path = os.path.join(os.getcwd(), "build")
    if os.path.exists(build_path):
        print(f"🧹 Removing: {build_path}")
        shutil.rmtree(build_path, ignore_errors=True)