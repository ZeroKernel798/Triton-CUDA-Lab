import torch, os, subprocess, ctypes, importlib.util

class KernelEngine:
    @staticmethod
    def setup_cuda(cu_path):
        so_path = cu_path.replace(".cu", ".so")
        subprocess.run(["nvcc", "-O3", "--shared", "-Xcompiler", "-fPIC", cu_path, "-o", so_path], check=True)
        return ctypes.CDLL(so_path)

    @staticmethod
    def setup_triton(py_path):
        spec = importlib.util.spec_from_file_location("triton_kernel", py_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.kernel # 约定入口都叫 kernel