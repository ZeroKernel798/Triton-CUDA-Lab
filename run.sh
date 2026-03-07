#!/bin/bash

# ==============================================================================
# Triton-CUDA-Lab 一键运行与分析工具
# 使用示例:
#   1. 性能跑分: ./run.sh 04-matrix-multiplication --mode cuda --bench_mode scaling
#   2. 时间轴分析: ./run.sh 04-matrix-multiplication --profile
#   3. 硬件指标分析: ./run.sh 04-matrix-multiplication --profile --ncu
# ==============================================================================

# 基础配置
OP=$1
if [ -z "$OP" ]; then
    echo "❌ 错误: 请提供算子目录名。"
    echo "用法: $0 <op_name> [args...]"
    exit 1
fi

shift # 弹出第一个参数 (算子名)
ARGS=$@ # 剩下的参数全部透传给 Python

# 路径与报告定义
REPORT_DIR="build/reports/${OP}"
mkdir -p "$REPORT_DIR"

# 检查是否包含 --profile 参数
if [[ " $ARGS " =~ " --profile " ]]; then
    echo "📸 [Profiler] 检测到采样模式..."
    
    # 判断是使用 ncu 还是 nsys
    if [[ " $ARGS " =~ " --ncu " ]]; then
        echo "🧬 [NCU] 启动 Nsight Compute (硬件指标/Roofline)..."
        # ncu 通常需要 sudo 权限获取硬件计数器
        # -E 保证环境变量(如虚拟环境路径)不丢失
        sudo -E $(which ncu) --profile-from-start off \
            --target-processes all \
            --force-overwrite \
            -o "${REPORT_DIR}/ncu_report" \
            $(which python3) run_lab.py "$OP" $ARGS
    else
        echo "🎞️ [NSYS] 启动 Nsight Systems (时间轴/NVTX)..."
        nsys profile -c cudaProfilerApi \
            -t cuda,nvtx,osrt \
            --force-overwrite true \
            -o "${REPORT_DIR}/nsys_report" \
            $(which python3) run_lab.py "$OP" $ARGS
    fi
    
    echo "✨ 采样结束。报告已生成至: ${REPORT_DIR}"
else
    # 普通模式：直接运行
    echo "🚀 [Bench] 启动常规测试模式..."
    python3 run_lab.py "$OP" $ARGS
fi