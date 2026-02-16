#!/bin/bash

# 1. 获取当前真实用户和 Python 路径
REAL_USER=$(logname || echo $SUDO_USER || whoami)
# 核心：获取当前 Conda 环境下的 python 绝对路径
CURRENT_PYTHON=$(which python3)
PROJECT_DIR=$(pwd)
BUILD_DIR="$PROJECT_DIR/build"

echo "🛠️  正在修复权限..."
if [ -d "$BUILD_DIR" ]; then
    sudo chown -R $REAL_USER:$REAL_USER "$BUILD_DIR"
    sudo chmod -R 755 "$BUILD_DIR"
fi

echo "🚀 正在锁定最高频率 (jetson_clocks)..."
sudo jetson_clocks

# 2. 关键：用获取到的特定 Python 路径运行，这样才能找到 torch
echo "🏗️  开始并行编译 (使用环境: $CURRENT_PYTHON)..."
sudo -u $REAL_USER $CURRENT_PYTHON build_lab.py

echo "✨ 全部搞定！现在你可以直接跑 run_lab.py 了。"