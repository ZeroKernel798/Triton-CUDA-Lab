import os
import time
import hashlib
import shutil
import torch
from torch.utils.cpp_extension import load

# 实现 cuda 文件的编译
class CudaCompiler:
    @staticmethod
    def get_gpu_info():
        # 探测硬件环境
        major, minor = torch.cuda.get_device_capability()
        return f"sm_{major}{minor}", f"{major}.{minor}", major, minor

    @staticmethod
    def _clean_locks(build_dir):
        # 清除锁定
        for lock_name in ["lock", ".ninja_lock"]:
            l_path = os.path.join(build_dir, lock_name)
            if os.path.exists(l_path):
                if time.time() - os.path.getmtime(l_path) > 15:
                    try: os.remove(l_path)
                    except: pass

    @staticmethod
    def compile(op_name, cu_file, macros: dict, is_nccl=False):
        arch_sm, arch_list, maj_v, min_v = CudaCompiler.get_gpu_info()

        # 设置环境变量和编译线程数 
        os.environ["MAX_JOBS"] = "1" 
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list
        
        abs_path = os.path.abspath(cu_file)
        file_base = os.path.basename(cu_file).split('.')[0]
        
        # 根据 Hash 来判断文件是否修改
        with open(abs_path, 'rb') as f:
            source_code = f.read()
        config_hash = hashlib.md5(source_code + str(sorted(macros.items())).encode()).hexdigest()
        
        module_name = f"k_{file_base}_{config_hash[:8]}"
        build_dir = os.path.join(os.getcwd(), "build", op_name, f"{file_base}_{arch_sm}_{config_hash[:8]}")
        os.makedirs(build_dir, exist_ok=True)

        CudaCompiler._clean_locks(build_dir)

        # 针对 cuda 的编译选项
        extra_cuda_cflags=[
                '-lineinfo',
                '-O3', '--use_fast_math', '-allow-unsupported-compiler',
                '-U__CUDA_NO_HALF_OPERATORS__',
                '-U__CUDA_NO_HALF_CONVERSIONS__',
                '-U__CUDA_NO_HALF2_OPERATORS__',
                '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                '--threads', '4', 
                f'-gencode=arch=compute_{maj_v}{min_v},code=sm_{maj_v}{min_v}'
            ]
        
        # 注入宏定义
        for k, v in macros.items():
            extra_cuda_cflags.append(f"-D{k}={v}")

        # 需要时链接nccl
        extra_ldflags = ['-lnccl'] if is_nccl else []
        return load(
            name=module_name,
            sources=[abs_path],
            extra_cflags=['-O3', '-std=c++17'],
            extra_cuda_cflags=extra_cuda_cflags,
            build_directory=build_dir,
            extra_ldflags=extra_ldflags,
            verbose=False
        )

def clean_build():
    build_path = os.path.join(os.getcwd(), "build")
    if os.path.exists(build_path):
        print(f"🧹清理构建产物: {build_path}")
        shutil.rmtree(build_path, ignore_errors=True)

# 实例化一下
compiler = CudaCompiler()