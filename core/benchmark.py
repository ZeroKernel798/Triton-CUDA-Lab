import matplotlib.pyplot as plt

class LabLogger:
    def __init__(self):
        self.results = [] # 存储格式: (version_name, time, is_correct)

    def record(self, name, time, correct):
        self.results.append({"name": name, "time": time, "correct": correct})

    def plot(self, op_name):
        names = [r['name'] for r in self.results]
        times = [r['time'] for r in self.results]
        colors = ['green' if r['correct'] else 'red' for r in self.results]

        plt.figure(figsize=(10, 6))
        plt.bar(names, times, color=colors)
        plt.ylabel('Time (ms)')
        plt.title(f'{op_name} Performance Comparison')
        plt.savefig(f"operators/{op_name}/performance.png")
        print(f"📊 图表已保存至 operators/{op_name}/performance.png")