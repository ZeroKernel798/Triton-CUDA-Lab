import os
import torch
import glob
import argparse
import multiprocessing  
from concurrent.futures import ProcessPoolExecutor
from utils.compiler import KernelEngine

# 找到显卡架构并设置环境变量，确保 PyTorch 编译插件能正确识别
def auto_set_cuda_arch():
    if torch.cuda.is_available():
        # 获取当前显卡的算力 (例如 A100 会返回 (8, 0))
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}.{minor}"
        
        # 写入环境变量，PyTorch 编译插件会自动读取
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
        
        print(f"🚀 [InferX Autodetect] 检测到当前 GPU 算力: {arch}")
        print(f"✅ 已自动设置 TORCH_CUDA_ARCH_LIST={arch}，将进行针对性编译。")
    else:
        print("⚠️ [InferX Warning] 未检测到 CUDA 环境，将使用默认配置。")

def compile_job(cu_file):
    try:
        KernelEngine.setup_cuda(cu_file)
        return f"{os.path.basename(cu_file)} 编译完成/已是最新"
    except Exception as e:
        return f"{os.path.basename(cu_file)} 失败: {e}"

def main():
    # 关键修复：必须在 main 的最开头设置 ---
    # spawn 会启动全新的进程，避免 fork 导致的 CUDA 环境冲突
    if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method('spawn')
        
    parser = argparse.ArgumentParser(description="Triton-CUDA-Lab 预编译工具")
    parser.add_argument("--op", type=str, help="指定编译某个算子")
    args = parser.parse_args()

    path_pattern = f"operators/{args.op if args.op else '*'}/cuda/*.cu"
    cu_files = glob.glob(path_pattern)
    
    if not cu_files:
        print("未发现待编译的 CUDA 文件。")
        return

    print(f"发现 {len(cu_files)} 个内核，开始并行构建....")

    # 3. 建议 max_workers 设为 4，避免内存和调度打架
    with ProcessPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(compile_job, cu_files))

    for r in results:
        print(r)

if __name__ == "__main__":
    # 第一时间锁定架构，确保环境变量在进程池启动前已经生效
    auto_set_cuda_arch() 
    
    # 启动main函数，进行编译任务
    main()