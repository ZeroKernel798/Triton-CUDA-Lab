import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.cuda.nvtx as nvtx


class KernelProfiler:
    """微观分析：负责 Nsight Systems / Compute 的采样控制"""
    @staticmethod
    def profile_scope(enabled=False, label="Kernel"):  
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
        self.peaks = {
            "bw": {"A100": 2039, "3090": 936, "4090": 1008},
            "flops": {"A100": 19.5, "3090": 35.6, "4090": 82.6}
        }

    def record(self, name, mode, x_label, metric):
        key = (name, mode)
        if key not in self.results: self.results[key] = {}
        self.results[key][x_label] = metric

    def plot(self, op_name, bench_mode, metric_type='bw', current_shape="Unknown"):
        target_mode = f"{bench_mode}_{metric_type}"
        relevant_keys = [k for k in self.results.keys() if k[1] == target_mode]
        if not relevant_keys: return
        
        plt.figure(figsize=(14, 8))
        gpu_name = torch.cuda.get_device_name()
        
        # 根据指标类型自动切换单位
        unit_map = {"bw": "GB/s", "flops": "TFLOPS", "ms": "ms"}
        unit = unit_map.get(metric_type, "units")
        markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']

        if bench_mode == 'scaling':
            all_shapes = []
            for name, _ in relevant_keys:
                all_shapes.extend(self.results[(name, target_mode)].keys())
            unique_shapes = sorted(list(set(all_shapes)), key=lambda x: eval(x.replace('x', '*'))) 

            for i, (name, _) in enumerate(relevant_keys):
                data = self.results[(name, target_mode)]
                y_values = [data.get(s, None) for s in unique_shapes]
                valid_indices = [idx for idx, v in enumerate(y_values) if v is not None]
                valid_values = [v for v in y_values if v is not None]
                
                plt.plot(valid_indices, valid_values, marker=markers[i % len(markers)], 
                         label=name, linewidth=1.5, markersize=5, alpha=0.7)
            
            plt.xticks(range(len(unique_shapes)), unique_shapes, rotation=30, ha='right')
            plt.xlabel('Input Shapes (MxNxK)', fontweight='bold')
        else:
            # Tuning 模式
            providers = sorted(list(set([k[0] for k in relevant_keys])))
            all_configs = []
            for p in providers: all_configs.extend(self.results[(p, target_mode)].keys())
            unique_configs = sorted(list(set(all_configs)))
            x = np.arange(len(unique_configs))
            width = 0.8 / len(providers)
            for i, name in enumerate(providers):
                data = self.results[(name, target_mode)]
                values = [data.get(c, 0) for c in unique_configs]
                offset = i * width - (len(providers) - 1) * width / 2
                plt.bar(x + offset, values, width, label=name, alpha=0.8)
            plt.xticks(x, unique_configs, rotation=45, ha='right')
            plt.xlabel('Kernel Configurations', fontweight='bold')

        # 针对时间指标，如果是 Scaling 模式且耗时差异巨大，建议开启对数轴
        if metric_type == 'ms' and bench_mode == 'scaling':
            plt.yscale('log')
            plt.ylabel(f'Latency ({unit}) - Log Scale', fontweight='bold')
        else:
            plt.ylabel(f'Performance ({unit})', fontweight='bold')

        # 天花板线 (仅限 bw 和 flops)
        if metric_type in self.peaks and bench_mode == 'scaling':
            peak_val = next((v for k, v in self.peaks[metric_type].items() if k in gpu_name), None)
            if peak_val:
                plt.axhline(y=peak_val, color='r', linestyle='--', alpha=0.3, label=f"Peak {peak_val} {unit}")

        plt.title(f'{bench_mode.upper()} Analysis - {op_name} ({metric_type.upper()})\nShape: {current_shape}', fontsize=16)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(axis='y', linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"operators/{op_name}/{bench_mode}_{metric_type}_analysis.png", dpi=150)
        plt.close()