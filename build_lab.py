# import os
# import torch
# import glob
# import argparse
# import multiprocessing  
# from concurrent.futures import ProcessPoolExecutor
# from utils.compiler import KernelEngine

# # 找到显卡架构并设置环境变量，确保 PyTorch 编译插件能正确识别
# def auto_set_cuda_arch():
#     if torch.cuda.is_available():
#         # 获取当前显卡的算力 (例如 A100 会返回 (8, 0))
#         major, minor = torch.cuda.get_device_capability()
#         arch = f"{major}.{minor}"
        
#         # 写入环境变量，PyTorch 编译插件会自动读取
#         os.environ["TORCH_CUDA_ARCH_LIST"] = arch
        
#         print(f"🚀 [InferX Autodetect] 检测到当前 GPU 算力: {arch}")
#         print(f"✅ 已自动设置 TORCH_CUDA_ARCH_LIST={arch}，将进行针对性编译。")
#     else:
#         print("⚠️ [InferX Warning] 未检测到 CUDA 环境，将使用默认配置。")

# def compile_job(cu_file):
#     try:
#         KernelEngine.setup_cuda(cu_file)
#         return f"{os.path.basename(cu_file)} 编译完成/已是最新"
#     except Exception as e:
#         return f"{os.path.basename(cu_file)} 失败: {e}"

# def main():
#     # 关键修复：必须在 main 的最开头设置 ---
#     # spawn 会启动全新的进程，避免 fork 导致的 CUDA 环境冲突
#     if multiprocessing.get_start_method(allow_none=True) is None:
#         multiprocessing.set_start_method('spawn')
        
#     parser = argparse.ArgumentParser(description="Triton-CUDA-Lab 预编译工具")
#     parser.add_argument("--op", type=str, help="指定编译某个算子")
#     args = parser.parse_args()

#     path_pattern = f"operators/{args.op if args.op else '*'}/cuda/*.cu"
#     cu_files = glob.glob(path_pattern)
    
#     if not cu_files:
#         print("未发现待编译的 CUDA 文件。")
#         return

#     print(f"发现 {len(cu_files)} 个内核，开始并行构建....")

#     # 3. 建议 max_workers 设为 4，避免内存和调度打架
#     with ProcessPoolExecutor(max_workers=2) as executor:
#         results = list(executor.map(compile_job, cu_files))

#     for r in results:
#         print(r)

# if __name__ == "__main__":
#     # 第一时间锁定架构，确保环境变量在进程池启动前已经生效
#     auto_set_cuda_arch() 
    
#     # 启动main函数，进行编译任务
#     main()

import os
import torch
import glob
import argparse
import time
from utils.compiler import KernelEngine

def auto_set_cuda_arch():
    """自动设置算力环境变量，这是 CUDA 编译加速的关键之一"""
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}.{minor}"
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
        # 强制让编译工具使用 Ninja (如果系统已安装)
        os.environ["USE_NINJA"] = "1"
        print(f"🚀 [InferX] 检测到 GPU 算力: {arch} | 已启用 Ninja 并行加速")
    else:
        print("⚠️ [InferX Warning] 未检测到 CUDA 环境")

def get_cu_files(op_name=None):
    """灵活的文件搜索逻辑"""
    if op_name:
        # 支持精确匹配或模糊匹配目录
        path_pattern = f"operators/*{op_name}*/cuda/*.cu"
    else:
        path_pattern = "operators/*/cuda/*.cu"
    
    files = glob.glob(path_pattern)
    # 过滤掉一些不直接编译的头文件或辅助文件（可选）
    return [f for f in files if f.endswith('.cu')]

def main():
    parser = argparse.ArgumentParser(description="Triton-CUDA-Lab 高性能编译器")
    parser.add_argument("--op", type=str, help="指定编译某个算子 (例如: 'matrix_mul')")
    parser.add_argument("--force", action="store_true", help="强制全量重新编译")
    args = parser.parse_args()

    # 1. 架构锁定
    auto_set_cuda_arch()

    # 2. 搜寻目标文件
    cu_files = get_cu_files(args.op)
    
    if not cu_files:
        print(f"❌ 未发现待编译文件 (搜索模式: {args.op if args.op else 'ALL'})")
        return

    print(f"🔍 发现 {len(cu_files)} 个内核，准备进入构建流程...")
    start_time = time.time()

    # 3. 核心编译循环
    # 注意：不要在 Python 层开多进程！
    # 因为 KernelEngine.setup_cuda 内部调用的 cpp_extension.load 
    # 本身就会启动多线程 Ninja 来榨干 CPU 性能。
    success_count = 0
    for i, cu_file in enumerate(cu_files):
        target_name = os.path.basename(cu_file).replace('.cu', '')
        print(f"[{i+1}/{len(cu_files)}] 正在同步状态: {target_name}...", end="\r")
        
        try:
            # KernelEngine 内部应逻辑：如果文件未改动，Ninja 会秒跳过
            KernelEngine.setup_cuda(cu_file)
            success_count += 1
        except Exception as e:
            print(f"\n❌ {target_name} 编译失败: {e}")

    # 4. 统计
    total_time = time.time() - start_time
    print(f"\n\n✨ 编译完成！")
    print(f"📊 成功: {success_count} | 耗时: {total_time:.2f}s")
    print(f"📂 缓存目录: ~/.cache/torch_extensions/")

if __name__ == "__main__":
    main()