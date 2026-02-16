import os
import sys
import importlib
import hashlib
from torch.utils.cpp_extension import load

class KernelEngine:
    @staticmethod
    def get_md5(file_path):
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    @staticmethod
    def setup_cuda(cu_file):
        # 1. 生成唯一识别符：获取算子文件夹名 + 文件名
        # 例如: operators/04-softmax-attention/cuda/testv1.cu 
        # 会变成: 04_softmax_attention_testv1
        abs_path = os.path.abspath(cu_file)
        path_parts = abs_path.split(os.sep)
        # 通常算子名在倒数第三级 (operators -> [op_name] -> cuda -> [file.cu])
        op_name = path_parts[-3] if len(path_parts) > 3 else "default"
        file_base = os.path.basename(cu_file).replace('.cu', '')
        
        # 组合成唯一的模块名，去掉非法字符
        module_name = f"{op_name}_{file_base}".replace('-', '_').replace('.', '_')
        
        # 2. 对应的 build 目录也区分开
        build_dir = os.path.join(os.getcwd(), "build", op_name, file_base)
        os.makedirs(build_dir, exist_ok=True)
        
        # 3. 必须把具体的 build_dir 加进路径
        if build_dir not in sys.path:
            sys.path.append(build_dir)

        # 4. MD5 逻辑保持，但针对唯一路径
        cu_md5 = KernelEngine.get_md5(abs_path)
        md5_file = os.path.join(build_dir, "source.md5")
        
        if os.path.exists(md5_file):
            with open(md5_file, 'r') as f:
                if f.read() == cu_md5:
                    try:
                        # 尝试加载
                        if module_name in sys.modules:
                            return sys.modules[module_name]
                        return importlib.import_module(module_name)
                    except Exception:
                        pass

        # 5. 编译流程
        os.environ["MAX_JOBS"] = str(os.cpu_count())
        print(f"🔧 [Compiler] Target: {module_name} | Dir: {op_name}")
        
        module = load(
            name=module_name,
            sources=[abs_path],
            extra_cflags={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3', 
                    '--use_fast_math', 
                    '--threads', '8',
                    '-Xcompiler', '-j8'
                ]
            },
            build_directory=build_dir,
            verbose=False
        )

        with open(md5_file, 'w') as f:
            f.write(cu_md5)
            
        return module