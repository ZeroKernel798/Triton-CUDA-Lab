# import os
# import sys
# import importlib
# import hashlib
# from torch.utils.cpp_extension import load

# class KernelEngine:
#     @staticmethod
#     # md5判断文件是否更新
#     def get_md5(file_path):
#         with open(file_path, 'rb') as f:
#             return hashlib.md5(f.read()).hexdigest()

#     @staticmethod
#     # 带缓存的nvcc编译过程
#     def setup_cuda(cu_file):
#         # 生成唯一识别符：获取算子文件夹名 + 文件名
#         # 例如: operators/04-softmax-attention/cuda/testv1.cu 
#         # 会变成: 04_softmax_attention_testv1
#         abs_path = os.path.abspath(cu_file)
#         path_parts = abs_path.split(os.sep)
#         op_name = path_parts[-3] if len(path_parts) > 3 else "default"
#         file_base = os.path.basename(cu_file).replace('.cu', '')
        
#         # 组合成唯一的模块名，去掉非法字符
#         module_name = f"{op_name}_{file_base}".replace('-', '_').replace('.', '_')
        
#         # 对应的 build 目录也区分开
#         build_dir = os.path.join(os.getcwd(), "build", op_name, file_base)
#         os.makedirs(build_dir, exist_ok=True)
        
#         # 把具体的 build_dir 加进路径 方便导入模块
#         if build_dir not in sys.path:
#             sys.path.append(build_dir)

#         # MD5 逻辑保持，但针对唯一路径
#         cu_md5 = KernelEngine.get_md5(abs_path)
#         md5_file = os.path.join(build_dir, "source.md5")
        
#         if os.path.exists(md5_file):
#             with open(md5_file, 'r') as f:
#                 if f.read() == cu_md5:
#                     # 文件没有修改 则直接加载已经编译好的模块
#                     try:
#                         if module_name in sys.modules:
#                             return sys.modules[module_name]
#                         return importlib.import_module(module_name)
#                     except Exception:
#                         pass

#         # 文件修改过 则进行正常编译
#         os.environ["MAX_JOBS"] = str(os.cpu_count())
#         print(f"[Compiler] Target: {module_name} | Dir: {op_name}")
        
#         module = load(
#             name=module_name,
#             sources=[abs_path],
#             extra_cflags={
#                 'cxx': ['-O3'],
#                 'nvcc': [
#                     '-O3', 
#                     '--use_fast_math', 
#                     '--threads', '8',
#                     '-Xcompiler', '-j8'
#                 ]
#             },
#             build_directory=build_dir,
#             verbose=False
#         )

#         with open(md5_file, 'w') as f:
#             f.write(cu_md5)
            
#         return module

import os
import sys
import torch
import hashlib
import importlib
import shutil
from torch.utils.cpp_extension import load

class KernelEngine:
    @staticmethod
    def get_md5(file_path):
        """计算文件的 MD5 值用于缓存校验"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    @staticmethod
    def get_gpu_arch():
        """
        获取当前显卡的计算能力 (Compute Capability)
        例如: RTX 4090 -> (8, 9) -> 'sm_89'
        """
        if not torch.cuda.is_available():
            return "no_cuda"
        major, minor = torch.cuda.get_device_capability()
        return f"sm_{major}{minor}"

    @staticmethod
    def setup_cuda(cu_file, force_recompile=False):
        """
        带架构隔离和缓存机制的 NVCC 编译封装
        """
        # 1. 基础路径解析
        abs_path = os.path.abspath(cu_file)
        path_parts = abs_path.split(os.sep)
        # 假设目录结构为: operators/{op_name}/cuda/{file}.cu
        op_name = path_parts[-3] if len(path_parts) > 3 else "default"
        file_base = os.path.basename(cu_file).replace('.cu', '')
        
        # 2. 获取架构后缀 (关键：版本隔离)
        arch = KernelEngine.get_gpu_arch()
        
        # 3. 构造唯一的模块名和构建目录
        # 增加 arch 后缀，确保不同显卡下的二进制文件不会冲突
        module_name = f"{op_name}_{file_base}_{arch}".replace('-', '_').replace('.', '_')
        build_dir = os.path.join(os.getcwd(), "build", op_name, file_base, arch)
        os.makedirs(build_dir, exist_ok=True)
        
        if build_dir not in sys.path:
            sys.path.append(build_dir)

        # 4. 环境变量设置：锁定架构加速编译
        # 这会告诉 nvcc 只编译当前显卡需要的指令集，大大缩短编译时间
        if arch != "no_cuda":
            # 将 'sm_89' 转为 '8.9' 格式传给 PyTorch
            arch_version = f"{arch[3]}.{arch[4]}"
            os.environ["TORCH_CUDA_ARCH_LIST"] = arch_version
        
        os.environ["MAX_JOBS"] = str(os.cpu_count())
        # 强制使用 Ninja (如果可用)
        os.environ["USE_NINJA"] = "1"

        # 5. MD5 缓存检查
        cu_md5 = KernelEngine.get_md5(abs_path)
        md5_file = os.path.join(build_dir, "source.md5")
        
        if not force_recompile and os.path.exists(md5_file):
            with open(md5_file, 'r') as f:
                if f.read() == cu_md5:
                    try:
                        # 尝试直接从缓存加载
                        if module_name in sys.modules:
                            return sys.modules[module_name]
                        return importlib.import_module(module_name)
                    except Exception:
                        # 如果加载失败（比如 .so 被意外删除），则继续向下执行编译
                        pass

        # 6. 执行即时编译 (JIT)
        
        print(f"🚀 [Compiler] Building: {module_name} for {arch}...")
        
        module = load(
            name=module_name,
            sources=[abs_path],
            extra_cflags={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3', 
                    '--use_fast_math', 
                    '-allow-unsupported-compiler', # 处理部分编译器版本冲突
                    '--threads', '4' # nvcc 内部并行化
                ]
            },
            build_directory=build_dir,
            verbose=False
        )

        # 编译成功后记录 MD5
        with open(md5_file, 'w') as f:
            f.write(cu_md5)
            
        return module

def clean_build():
    """清理所有编译产物的辅助工具"""
    build_path = os.path.join(os.getcwd(), "build")
    if os.path.exists(build_path):
        print(f"🧹 Removing build directory: {build_path}")
        shutil.rmtree(build_path)
    
    # 清理 Python 缓存
    for root, dirs, files in os.walk(os.getcwd()):
        for d in dirs:
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d))
                print(f"  Removed __pycache__ in {root}")

