import matplotlib.pyplot as plt

class LabLogger:
    def __init__(self):
        self.results = []

    def record(self, name, time):
        """ 记录性能数据 """
        self.results.append({
            "name": name, 
            "time": time
        })

    def plot(self, op_name):
        if not self.results: return
        
        names = [r['name'] for r in self.results]
        times = [r['time'] for r in self.results]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制耗时对比柱状图
        bars = ax.bar(names, times, color='skyblue', alpha=0.8)
        
        # 在柱状图上方标注具体数值
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}ms',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), # 3点偏移
                        textcoords="offset points",
                        ha='center', va='bottom')

        ax.set_ylabel('Latency (ms)', fontweight='bold')
        ax.set_title(f'Performance Comparison: {op_name}', fontsize=14)
        
        plt.tight_layout()
        save_path = f"operators/{op_name}/performance_analysis.png"
        plt.savefig(save_path)
        print(f"📊 耗时对比图表已生成: {save_path}")