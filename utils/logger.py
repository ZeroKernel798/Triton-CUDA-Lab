import matplotlib.pyplot as plt
import numpy as np
import os

def plot_scaling_line(op_name, data_records, metric="gbps", output_dir="plots"):
    """绘制随多维尺寸变化的性能曲线 (Line Chart)"""
    if not data_records: return
    os.makedirs(output_dir, exist_ok=True)

    # 1. 指标配置映射
    metric_map = {
        "ms":     {"key": "ms",     "label": "Latency (ms)",      "title": "Latency Scaling"},
        "bw":     {"key": "gbps",   "label": "Throughput (GB/s)", "title": "Bandwidth Scaling"},
        "flops":  {"key": "tflops", "label": "Compute (TFLOPS)",  "title": "Compute Performance"}
    }
    cfg = metric_map.get(metric, metric_map["bw"])
    
    # 2. 提取所有唯一的尺寸标签，并按数值大小(size_val)排序，确保横轴有序
    # 这样可以处理 1024x1024, 2048x4096 这种字符串横坐标
    unique_sizes = sorted(
        list({(r['size_label'], r['size_val']) for r in data_records}),
        key=lambda x: x[1]
    )
    size_labels = [x[0] for x in unique_sizes]
    x_indexes = np.arange(len(size_labels))
    
    plt.figure(figsize=(12, 7))
    
    # 3. 按版本绘制连线
    versions = sorted(list(set(r['version'] for r in data_records)))
    for ver in versions:
        # 获取该内核版本下的所有数据
        subset = [r for r in data_records if r['version'] == ver]
        
        # 将数据对应到排序后的 x_indexes 上
        y_vals = []
        for s_lab in size_labels:
            # 找到对应 size_label 的数值，找不到填 None (绘图会跳过)
            val = next((r[cfg['key']] for r in subset if r['size_label'] == s_lab), None)
            y_vals.append(val)
        
        plt.plot(x_indexes, y_vals, marker='o', label=f"Kernel: {ver}")

    # 4. 样式修饰
    plt.title(f"{cfg['title']}: {op_name}", fontsize=14)
    plt.ylabel(cfg['label'])
    
    # --- 核心修复：将 X 轴设为字符串标签 ---
    plt.xticks(x_indexes, size_labels, rotation=30, ha='right')
    plt.xlabel("Data Size")
    
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    # 5. 保存
    save_path = os.path.join(output_dir, f"{op_name}_{metric}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"📈 [Scaling Line] 已生成: {save_path}")

def plot_tuning_bar(op_name, data_records, metric="ms", output_dir="plots"):
    if not data_records: return
    os.makedirs(output_dir, exist_ok=True)

    # 1. 映射配置
    metric_map = {
        "ms":    {"key": "ms",     "label": "Latency (ms)",      "title": "Latency Tuning"},
        "bw":    {"key": "gbps",   "label": "Throughput (GB/s)", "title": "Bandwidth Tuning"},
        "flops": {"key": "tflops", "label": "Compute (TFLOPS)",  "title": "Compute Tuning"}
    }
    cfg = metric_map.get(metric, metric_map["ms"])
    
    # 2. 获取去重后的配置标签和内核版本
    unique_configs = []
    for r in data_records:
        if r['config_label'] not in unique_configs:
            unique_configs.append(r['config_label'])
            
    versions = sorted(list(set(r['version'] for r in data_records)))
    
    # 3. 设置绘图参数
    plt.figure(figsize=(max(12, len(unique_configs)*1.2), 7))
    x = np.arange(len(unique_configs))  # 横轴基准位置
    width = 0.8 / len(versions)         # 柱子宽度
    
    # 4. 按版本绘制柱状图
    for i, ver in enumerate(versions):
        y_vals = []
        for c_label in unique_configs:
            # 匹配对应配置的数据，若无则为0
            val = next((r[cfg['key']] for r in data_records if r['version'] == ver and r['config_label'] == c_label), 0)
            y_vals.append(val)
        
        # 计算每个版本的偏移，使其并排显示
        offset = i * width - (len(versions) - 1) * width / 2
        bars = plt.bar(x + offset, y_vals, width, label=f"Kernel: {ver}")
        
        # 在柱子顶端标数值（可选，如果太挤可以关掉）
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.2f}', ha='center', va='bottom', fontsize=8, rotation=90 if len(unique_configs)>10 else 0)

    # 5. 修饰样式
    plt.title(f"{cfg['title']} - {op_name}", fontsize=14)
    plt.ylabel(cfg['label'])
    plt.xticks(x, unique_configs, rotation=45, ha='right', fontsize=9)
    plt.legend()
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{op_name}_{metric}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"📊 [Tuning Bar] 已生成: {save_path}")