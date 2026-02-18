import matplotlib.pyplot as plt
import torch
import numpy as np

class LabLogger:
    def __init__(self):
        # 结果存储: { (provider_name, mode): { x_label: metric } }
        self.results = {}

    def record(self, name, mode, x_label, metric):
        key = (name, mode)
        if key not in self.results: self.results[key] = {}
        self.results[key][x_label] = metric

    def plot(self, op_name, mode):
        if not self.results: return
        
        plt.figure(figsize=(12, 7))
        relevant_keys = [k for k in self.results.keys() if k[1] == mode]
        
        if mode == 'scaling':
            for name, _ in relevant_keys:
                data = self.results[(name, mode)]
                sizes = sorted(data.keys())
                plt.plot(sizes, [data[s] for s in sizes], marker='o', label=name, linewidth=2)
            gpu_name = torch.cuda.get_device_name()
            if "A100" in gpu_name:
                plt.axhline(y=2039, color='r', linestyle='--', alpha=0.5, label='A100 HBM2e Peak (2039 GB/s)')
            elif "3090" in gpu_name:
                plt.axhline(y=936, color='r', linestyle='--', alpha=0.5, label='RTX 3090 Peak (936 GB/s)')
            plt.xscale('log', base=2)
            plt.xlabel('Input Size (N)', fontweight='bold')
            plt.title(f'Scaling Analysis: {op_name}', fontsize=14)

        
        else:
            # Tuning 模式：分组柱状图优化 
            providers = sorted(list(set([k[0] for k in relevant_keys])))
            
            # 1. 获取所有唯一的配置标签并排序 (比如 32, 128, 32_2...)
            all_configs = []
            for p in providers:
                all_configs.extend(self.results[(p, mode)].keys())
            unique_configs = sorted(list(set(all_configs)))
            
            # 2. 设置 X 轴位置和柱子宽度
            x = np.arange(len(unique_configs))
            width = 0.8 / len(providers)  # 总宽度 0.8，按选手平分
            
            # 3. 为每个选手画柱子，并应用偏移量
            for i, name in enumerate(providers):
                data = self.results[(name, mode)]
                # 如果某个选手没有某个配置，吞吐量记为 0
                values = [data.get(c, 0) for c in unique_configs]
                
                # 计算偏移：让柱子中心对称分布
                offset = i * width - (len(providers) - 1) * width / 2
                plt.bar(x + offset, values, width, label=name, alpha=0.8, edgecolor='white')

            plt.xticks(x, unique_configs, rotation=45, ha='right')
            plt.xlabel('Configurations (BlockSize / Warps)', fontweight='bold')
            plt.title(f'Tuning Optimization (Grouped): {op_name}', fontsize=14)

        plt.ylabel('Throughput (GB/s)', fontweight='bold')
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3) # 柱状图只保留横向网格比较清晰
        plt.tight_layout()

        plt.ylabel('Throughput (GB/s)', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        path = f"operators/{op_name}/{mode}_analysis.png"
        plt.savefig(path)
        plt.close()
        print(f"📊 {mode.capitalize()} 图表已生成: {path}")