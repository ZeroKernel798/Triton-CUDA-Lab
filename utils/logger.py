import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.cuda.nvtx as nvtx


class KernelProfiler:
    """微观分析：负责 Nsight Systems / Compute 的采样控制"""
    @staticmethod
    def profile_scope(enabled=False, label="Kernel"):  # ✨ 统一改为 profile_scope
        class ProfileContext:
            def __enter__(self):
                if enabled:
                    import torch
                    import torch.cuda.nvtx as nvtx
                    torch.cuda.synchronize()
                    torch.cuda.profiler.start()
                    nvtx.range_push(label)

            def __exit__(self, type, value, traceback):
                if enabled:
                    import torch
                    import torch.cuda.nvtx as nvtx
                    torch.cuda.synchronize()
                    nvtx.range_pop()
                    torch.cuda.profiler.stop()
        return ProfileContext()
    
class LabLogger:
    def __init__(self):
        self.results = {}
        # 硬件理论峰值字典 (可以根据开源需要继续补充)
        self.peaks = {
            "bw": {
                "A100": 2039,    # GB/s (HBM2e)
                "3090": 936,     # GB/s
                "4090": 1008,    # GB/s
            },
            "flops": {
                "A100": 19.5,    # TFLOPS (FP32)
                "3090": 35.6,    # TFLOPS
                "4090": 82.6,    # TFLOPS
            }
        }

    def record(self, name, mode, x_label, metric):
        key = (name, mode)
        if key not in self.results: self.results[key] = {}
        self.results[key][x_label] = metric

    def plot(self, op_name, bench_mode, metric_type='bw'):
        """
        bench_mode: 'scaling' 或 'tuning'
        metric_type: 'bw' (带宽) 或 'flops' (算力)
        """
        # 实际从 results 里找数据的 key
        target_mode = f"{bench_mode}_{metric_type}"
        relevant_keys = [k for k in self.results.keys() if k[1] == target_mode]
        
        if not relevant_keys:
            print(f"⚠️ 没有找到 {target_mode} 的数据，跳过绘图。")
            return
        
        plt.figure(figsize=(12, 7))
        gpu_name = torch.cuda.get_device_name()
        
        # --- 1. 处理 Scaling 模式 (折线图) ---
        if bench_mode == 'scaling':
            for name, _ in relevant_keys:
                data = self.results[(name, target_mode)]
                sizes = sorted(data.keys())
                plt.plot(sizes, [data[s] for s in sizes], marker='o', label=name, linewidth=2)
            
            # 画天花板虚线
            peak_val = next((v for k, v in self.peaks[metric_type].items() if k in gpu_name), None)
            if peak_val:
                label = f"{gpu_name} Peak ({peak_val} {'GB/s' if metric_type=='bw' else 'TFLOPS'})"
                plt.axhline(y=peak_val, color='r', linestyle='--', alpha=0.5, label=label)

            plt.xscale('log', base=2)
            plt.xlabel('Input Size (Shape)', fontweight='bold')

        # --- 2. 处理 Tuning 模式 (柱状图) ---
        else:
            providers = sorted(list(set([k[0] for k in relevant_keys])))
            all_configs = []
            for p in providers:
                all_configs.extend(self.results[(p, target_mode)].keys())
            unique_configs = sorted(list(set(all_configs)))
            
            x = np.arange(len(unique_configs))
            width = 0.8 / len(providers)
            
            for i, name in enumerate(providers):
                data = self.results[(name, target_mode)]
                values = [data.get(c, 0) for c in unique_configs]
                offset = i * width - (len(providers) - 1) * width / 2
                plt.bar(x + offset, values, width, label=name, alpha=0.8)

            plt.xticks(x, unique_configs, rotation=45, ha='right')
            plt.xlabel('Configurations', fontweight='bold')

        # --- 3. 公共格式设置 ---
        unit = "GB/s" if metric_type == 'bw' else "TFLOPS"
        plt.ylabel(f'Performance ({unit})', fontweight='bold')
        plt.title(f'{bench_mode.capitalize()} Analysis ({metric_type.upper()}): {op_name}', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 文件名区分指标
        path = f"operators/{op_name}/{bench_mode}_{metric_type}_analysis.png"
        plt.savefig(path)
        plt.close()
        print(f"✅ 图表已生成: {path}")