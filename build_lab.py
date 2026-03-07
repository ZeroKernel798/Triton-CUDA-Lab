import os
import sys
import glob
import time
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils.compiler import KernelEngine, clean_build

# --- ✨ 核心修复：强制使用 spawn 模式启动子进程 ---
# 必须在脚本最顶层，且在所有 CUDA 调用之前设置
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

def format_time(seconds):
    if seconds is None or seconds < 0: return "--:--"
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"

def draw_progress(current, total, op_name, file_name, elapsed_time, bar_width=25):
    progress = float(current) / total
    filled = int(progress * bar_width)
    bar = "█" * filled + "░" * (bar_width - filled)
    eta_str = format_time((elapsed_time / current) * (total - current)) if current > 0 else "计算中"
    
    # 清理行并打印进度
    output = f"\r进度: |{bar}| {int(progress * 100):>3}% [{format_time(elapsed_time)} < {eta_str}] 处理: {op_name}/{file_name:<25}"
    sys.stdout.write(output)
    sys.stdout.flush()

def compile_worker(cu_file, force):
    """Worker 进程执行逻辑"""
    # 注意：在 spawn 模式下，子进程会重新 import 相关的库
    try:
        parts = cu_file.split(os.sep)
        op_folder = parts[-3] if len(parts) >= 3 else "root"
        file_base = os.path.basename(cu_file).replace('.cu', '')
        
        KernelEngine.setup_cuda(cu_file, force_recompile=force)
        return True, op_folder, file_base
    except Exception as e:
        return False, "Error", f"{os.path.basename(cu_file)}: {str(e)[:60]}"

def main():
    parser = argparse.ArgumentParser(description="Triton-CUDA-Lab 并行编译器")
    parser.add_argument("--op", type=str, help="模糊匹配")
    parser.add_argument("--force", action="store_true", help="强制编译")
    parser.add_argument("--clean", action="store_true", help="清理构建")
    # ✨ 修正：支持同时识别 -j 和 --j
    parser.add_argument("-j", "--j", type=int, default=4, help="并行任务数")
    args = parser.parse_args()

    if args.clean:
        clean_build()
        if not args.op: return

    # 获取硬件信息（此时在父进程初始化一次 CUDA）
    arch_sm, arch_list, _, _ = KernelEngine.get_gpu_info()
    print(f"💡 环境检测: {arch_sm} ({arch_list}) | 🚀 并行度: {args.j}")

    # 搜索内核
    all_cu = glob.glob("operators/**/*.cu", recursive=True)
    cu_files = sorted([f for f in all_cu if not args.op or args.op.lower() in f.lower()])
    
    total = len(cu_files)
    if total == 0:
        print("⚠️ 未发现匹配内核"); return

    print(f"🔍 启动并行编译流程 (Spawn 模式)...")
    start_time = time.time()
    success_count = 0

    # 使用多进程池
    with ProcessPoolExecutor(max_workers=args.j) as executor:
        futures = {executor.submit(compile_worker, f, args.force): f for f in cu_files}
        
        for i, future in enumerate(as_completed(futures)):
            success, op, msg = future.result()
            if success:
                success_count += 1
            else:
                # 编译失败时换行显示，避免覆盖进度条
                sys.stdout.write(f"\n❌ 编译失败: {msg}\n")
            
            draw_progress(i + 1, total, op, msg if success else "Error", time.time() - start_time)

    print(f"\n\n✨ 构建结束！成功: {success_count}/{total} | 耗时: {format_time(time.time() - start_time)}")

if __name__ == "__main__":
    main()