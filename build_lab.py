import os
import sys
import glob
import time
import argparse
from utils.compiler import KernelEngine, clean_build

def format_time(seconds):
    """将秒数格式化为分:秒"""
    if seconds is None or seconds < 0: return "--:--"
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"

def draw_progress(current, total, op_name, file_name, elapsed_time, bar_width=25):
    """绘制带进度条、语义化路径和精准 ETA 的界面"""
    progress = float(current) / total
    filled = int(progress * bar_width)
    bar = "█" * filled + "░" * (bar_width - filled)
    
    # 改进 ETA：只有完成 1 个以上才计算，否则显示估算中
    if current > 1:
        # 基于已完成的进度推算总时长：total_est = elapsed / (current-1) * total
        # 剩余时长 = total_est - elapsed
        eta_seconds = (elapsed_time / (current - 1)) * (total - (current - 1)) - (elapsed_time / (current - 1))
        eta_str = format_time(max(0, eta_seconds))
    else:
        eta_str = "计算中"
    
    elapsed_str = format_time(elapsed_time)
    display_text = f"{op_name}/{file_name}"
    
    # 使用 :<35 确保长路径不会导致行残影，:>3 确保百分比对齐
    sys.stdout.write(
        f"\r进度: |{bar}| {int(progress * 100):>3}% "
        f"[{elapsed_str} < {eta_str}] "
        f"处理中: {display_text:<35}"
    )
    sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser(description="Triton-CUDA-Lab 智能模糊编译器")
    parser.add_argument("--op", type=str, help="模糊匹配关键词 (如: 04, softmax, cublas)")
    parser.add_argument("--force", action="store_true", help="强制重新编译")
    parser.add_argument("--clean", action="store_true", help="清理构建产物")
    args = parser.parse_args()

    # --- 1. 环境预检 ---
    if args.clean:
        clean_build()
        if not args.op: return

    arch_sm, arch_list = KernelEngine.get_gpu_info()
    print(f"💡 Device: {arch_sm} ({arch_list})")
    nv_threads = KernelEngine.init_performance_strategy()

    # --- 2. 核心改动：递归模糊匹配 ---
    # ** 表示递归搜索所有子目录
    all_cu_files = glob.glob("operators/**/*.cu", recursive=True)
    
    if args.op:
        target = args.op.lower()
        # 只要路径中包含用户输入的字符串（不分大小写）就命中
        cu_files = sorted([f for f in all_cu_files if target in f.lower()])
    else:
        cu_files = sorted(all_cu_files)
    
    total = len(cu_files)
    if total == 0:
        print(f"⚠️ 未发现匹配 '{args.op if args.op else 'ALL'}' 的内核文件")
        return

    print(f"🔍 发现 {total} 个内核，启动构建流程...")
    start_time = time.time()

    # --- 3. 构建循环 ---
    success = 0
    for i, cu_file in enumerate(cu_files):
        # 语义化路径提取：operators/04-softmax/cuda/kernel.cu -> (04-softmax, kernel)
        parts = cu_file.split(os.sep)
        op_folder = parts[-3] if len(parts) >= 3 else "root"
        file_base = os.path.basename(cu_file).replace('.cu', '')
        
        current_elapsed = time.time() - start_time
        draw_progress(i + 1, total, op_folder, file_base, current_elapsed)
        
        try:
            KernelEngine.setup_cuda(cu_file, nvcc_threads=nv_threads, force_recompile=args.force)
            success += 1
        except Exception as e:
            # 报错时强制换行，并打印简减后的错误
            sys.stdout.write(f"\n❌ 错误: {op_folder}/{file_base} -> {str(e)[:70]}...\n")
            sys.stdout.flush()

    # --- 4. 总结 ---
    final_duration = time.time() - start_time
    print(f"\n\n✨ 构建任务结束！")
    print(f"📊 成功率: {success}/{total} | 总耗时: {format_time(final_duration)}")
    print(f"📂 产物路径: {os.path.join(os.getcwd(), 'build')}")

if __name__ == "__main__":
    main()