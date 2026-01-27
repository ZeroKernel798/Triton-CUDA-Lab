import os
import subprocess
import ctypes
import importlib.util
import sys

class KernelEngine:
    @staticmethod
    def setup_cuda(cu_file):
        """
        编译 CUDA 文件并返回 ctypes CDLL 对象
        """
        so_file = cu_file.replace('.cu', '.so')
        
        # 简单编译指令：编译为共享库
        # -Xcompiler -fPIC 是必须的，这样 Python 才能加载
        compile_cmd = [
            "nvcc", "-O3", "--shared", "-Xcompiler", "-fPIC",
            cu_file, "-o", so_file
        ]
        
        # 检查是否需要重新编译 (简单逻辑：源码比库新就重编)
        if not os.path.exists(so_file) or os.path.getmtime(cu_file) > os.path.getmtime(so_file):
            result = subprocess.run(compile_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"CUDA Compilation Failed: {result.stderr}")
        
        # 使用 ctypes 加载
        lib = ctypes.CDLL(os.path.abspath(so_file))
        return lib

    @staticmethod
    def setup_triton(py_file):
        """
        动态加载 Triton 的 Python 文件作为模块返回
        """
        module_name = os.path.basename(py_file).replace('.py', '')
        
        # 使用 importlib 动态加载文件
        spec = importlib.util.spec_from_file_location(module_name, py_file)
        module = importlib.util.module_from_spec(spec)
        
        # 这一步极其重要：必须执行模块，否则里面的 solve 函数还没被定义
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            raise Exception(f"Failed to exec Triton module: {e}")
            
        # 检查模块里有没有 solve 函数
        if not hasattr(module, 'solve'):
            raise AttributeError(f"模块 {module_name} 中未找到 'solve' 函数！")
            
        return module # 返回整个模块，这样才能拿到 module.solve