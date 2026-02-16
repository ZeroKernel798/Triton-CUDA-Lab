import os
import glob
import argparse
import multiprocessing  # <--- 1. 引入这个
from concurrent.futures import ProcessPoolExecutor
from core.engine import KernelEngine

def compile_job(cu_file):
    try:
        KernelEngine.setup_cuda(cu_file)
        return f"✅ {os.path.basename(cu_file)} 编译完成/已是最新"
    except Exception as e:
        return f"❌ {os.path.basename(cu_file)} 失败: {e}"

def main():
    # --- 2. 关键修复：必须在 main 的最开头设置 ---
    # spawn 会启动全新的进程，避免 fork 导致的 CUDA 环境冲突
    if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method('spawn')
        
    parser = argparse.ArgumentParser(description="Triton-CUDA-Lab 预编译工具")
    parser.add_argument("--op", type=str, help="指定编译某个算子")
    args = parser.parse_args()

    path_pattern = f"operators/{args.op if args.op else '*'}/cuda/*.cu"
    cu_files = glob.glob(path_pattern)
    
    if not cu_files:
        print("🔍 未发现待编译的 CUDA 文件。")
        return

    print(f"🚀 发现 {len(cu_files)} 个内核，开始并行构建 (Orin NX 8-Core Spawn Mode)...")

    # 3. 建议 max_workers 设为 4，避免内存和调度打架
    with ProcessPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(compile_job, cu_files))

    for r in results:
        print(r)

if __name__ == "__main__":
    main()