#!/bin/bash

# --- 1. 环境与路径配置 ---
export TMPDIR=~/nvcc_tmp
export TORCH_EXTENSIONS_DIR=~/torch_build_cache
mkdir -p $TMPDIR $TORCH_EXTENSIONS_DIR reports

# --- 2. 参数处理 ---
OP=${1:-"03"}
MODE=${2:-"scaling"}
TOOL=${3:-"nsys"}

# 自动探测物理 GPU 数量
NUM_GPUS=$(nvidia-smi -L | wc -l)

# 💡 核心逻辑：通过版本名自动判断进程数
# 只要配置里包含 'nccl' 关键字，NPROC=8(或实际GPU数)；否则 NPROC=1
NPROC=$(python3 -c "
import sys
try:
    from utils import loader
    spec = loader.get_spec('$OP')
    # 打印版本名看看，方便调试 (输出到 stderr 不会影响 NPROC 赋值)
    configs = getattr(spec, 'cuda_tuning_configs', [])
    versions = [str(c.get('version', '')) for c in configs]
    
    is_nccl = any('nccl' in v.lower() for v in versions)
    
    if is_nccl:
        print($NUM_GPUS)
    else:
        print(1)
except Exception as e:
    # 打印具体的错误到终端，防止被‘吞掉’
    print(f'Detection Error: {e}', file=sys.stderr)
    print(1)
")

REPORT_NAME="reports/${OP}_${MODE}_$(date +%m%d_%H%M)"
echo "🧬 [InferX] Op: $OP | Mode: $MODE | Parallel_Procs: $NPROC"

# --- 3. 统一构造 RUN_CMD ---
# 使用 --standalone 避免手动设置 MASTER_ADDR
RUN_CMD="torchrun --standalone --nproc_per_node=$NPROC run.py --op $OP --mode $MODE"

# --- 4. 执行分发逻辑 ---
if [ "$MODE" == "profile" ]; then
    if [ "$TOOL" == "nsys" ]; then
        echo "📊 启动分布式 NSYS Timeline 分析..."
        nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
            --trace=cuda,nvtx --force-overwrite true -o "${REPORT_NAME}_nsys_%p" $RUN_CMD
    elif [ "$TOOL" == "ncu" ]; then
        echo "🧐 启动 NCU 硬件指标分析 (Target: All Processes)..."
        ncu --profile-from-start off --target-processes all --set full \
            --force-overwrite -o "${REPORT_NAME}_ncu" $RUN_CMD
    else
        $RUN_CMD
    fi
else
    # Scaling 或 Tuning 模式
    echo "🏃 运行指令: $RUN_CMD"
    $RUN_CMD --metrics all
fi