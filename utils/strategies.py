import torch.distributed as dist
from core.oprunner import OperatorRunner, log_master
from utils.logger import plot_scaling_line, plot_tuning_bar
from utils.factory import get_executor_instance


def _resolve_metrics(raw_metrics):
    metric_aliases = {
        "bw": "mbu",
        "bandwidth": "mbu",
        "flops": "mfu",
        "compute": "mfu",
    }
    if raw_metrics.lower() == "all":
        return ["ms", "mbu", "mfu"]
    return [metric_aliases.get(m.strip().lower(), m.strip().lower()) for m in raw_metrics.split(',')]

def run_profile_strategy(spec, valid_kernels, args, op_folder):
    raw_input = getattr(spec, 'perf_input', None)
    size_cfg = raw_input[0] if isinstance(raw_input, list) else raw_input

    if isinstance(size_cfg, dict):
        size_label = "x".join([str(v) for v in size_cfg.values()])
    else:
        size_label = str(size_cfg)
        first_x_key = list(spec.x_vals[0].keys())[0]
        size_cfg = {first_x_key: size_cfg}

    rank = dist.get_rank() if dist.is_initialized() else 0

    for uid, k_info in valid_kernels.items():
        ver = k_info.get('ver')
        k_type = k_info.get('type')
        display_name = f"{ver} [{k_type}]"

        is_dist_ver = spec.is_nccl_version(ver)
        if not is_dist_ver and rank != 0:
            continue

        log_master(f"\n🔍 [Profile Mode] 正在为 {display_name} 寻找最佳配置...")
        
        if k_type == "triton":
            configs = [c for c in getattr(spec, 'tuning_configs', []) if c.get("version") == ver]
        else:
            configs = [c for c in getattr(spec, 'cuda_tuning_configs', []) if c.get("version") == ver]
        if not configs:
            configs = [{}]
        
        best_res = {"ms": float('inf'), "cfg": None}
        for cfg in configs:
            executor = get_executor_instance(op_folder, spec, k_info, cfg)
            runner = OperatorRunner(spec, executor, args)
            perf = runner.measure(size_cfg)
            if perf["ms"] < best_res["ms"]:
                best_res.update(perf)
                best_res["cfg"] = cfg
        
        best_cfg = best_res["cfg"]
        log_master(f"🏆 找到最佳配置: {best_cfg} (Latency: {best_res['ms']:.4f} ms)")

        best_executor = get_executor_instance(op_folder, spec, k_info, best_cfg)
        profiler_runner = OperatorRunner(spec, best_executor, args)
        
        log_master(f"🎬 准备就绪，开始对规模 {size_label} 进行 Profiling...")
        profiler_runner.profile(size_cfg)


def run_scaling_strategy(spec, valid_kernels, args, op_folder):
    all_results = [] 
    rank = dist.get_rank() if dist.is_initialized() else 0

    for uid, k_info in valid_kernels.items():
        ver = k_info.get('ver')
        k_type = k_info.get('type')
        display_name = f"{ver} [{k_type}]"

        is_dist_ver = spec.is_nccl_version(ver)
        if not is_dist_ver and rank != 0:
            continue

        log_master(f"\n🚀 正在评估内核实现: {display_name}")
        
        header = f"{'Size (RxC)':>15} | {'Latency':>14} | {'MBU':>12} | {'MFU':>12} | Best Config"
        log_master(header)
        log_master("   " + "-" * (len(header) + 10))

        if k_type == "triton":
            configs = [c for c in getattr(spec, 'tuning_configs', []) if c.get("version") == ver]
        else:
            configs = [c for c in getattr(spec, 'cuda_tuning_configs', []) if c.get("version") == ver]
        if not configs:
            configs = [{}]

        for size_cfg in spec.x_vals:
            best_res = {"ms": float('inf'), "gbps": 0.0, "tflops": 0.0, "mbu": 0.0, "mfu": 0.0, "cfg": None}
            display_size = "x".join([str(v) for v in size_cfg.values()])
            total_elements = 1
            for v in size_cfg.values(): total_elements *= v

            for cfg in configs:
                executor = get_executor_instance(op_folder, spec, k_info, cfg)
                runner = OperatorRunner(spec, executor, args)
                perf = runner.measure(size_cfg)
                if perf["ms"] < best_res["ms"]:
                    best_res.update(perf)
                    best_res["cfg"] = cfg

            all_results.append({
                "version": display_name,
                "size_label": display_size,  
                "size_val": total_elements, 
                "ms": best_res["ms"],
                "gbps": best_res["gbps"],
                "tflops": best_res["tflops"],
                "mbu": best_res["mbu"],
                "mfu": best_res["mfu"]
            })

            latency_str = f"{best_res['ms']:.4f} ms"
            mbu_str     = f"{best_res['mbu']:.2f}%"
            mfu_str     = f"{best_res['mfu']:.2f}%"
            display_cfg = {k: v for k, v in best_res['cfg'].items() if k != 'version'}
            log_master(f"{display_size:>15} | {latency_str:>14} | {mbu_str:>12} | {mfu_str:>12} | {display_cfg}")

    if all_results:
        if not dist.is_initialized() or dist.get_rank() == 0:
            raw_metrics = getattr(args, 'metrics', 'ms')
            target_metrics = _resolve_metrics(raw_metrics)
            for m in target_metrics:
                if m in ["ms", "mbu", "mfu", "bw", "flops"]:
                    plot_scaling_line(f"{spec.name}_scaling", all_results, metric=m)


def run_tuning_strategy(spec, valid_kernels, args, op_folder):
    all_results = [] 
    raw_input = getattr(spec, 'perf_input', None)
    if raw_input is None: return
    
    size_cfg = raw_input[0] if isinstance(raw_input, list) else raw_input
    if isinstance(size_cfg, dict):
        size_label = list(size_cfg.values())[0]
    else:
        size_label = size_cfg
        size_cfg = {list(spec.x_vals[0].keys())[0]: size_cfg}

    rank = dist.get_rank() if dist.is_initialized() else 0
    
    for uid, k_info in valid_kernels.items():
        ver = k_info.get('ver')
        k_type = k_info.get('type')
        display_name = f"{ver} [{k_type}]"

        is_dist_ver = spec.is_nccl_version(ver)
        if not is_dist_ver and rank != 0:
            continue

        log_master(f"\n🔍 [Tuning Mode] 正在深度评估内核: {display_name}")
        log_master(f"📍 Target Size: {size_label}")
        
        header = f"{'Config':>30} | {'Latency':>14} | {'MBU':>12} | {'MFU':>12}"
        log_master(header)
        log_master("-" * len(header))

        if k_type == "triton":
            configs = [c for c in getattr(spec, 'tuning_configs', []) if c.get("version") == ver]
        else:
            configs = [c for c in getattr(spec, 'cuda_tuning_configs', []) if c.get("version") == ver]
        if not configs:
            configs = [{}]
        
        for idx, cfg in enumerate(configs):
            executor = get_executor_instance(op_folder, spec, k_info, cfg)
            runner = OperatorRunner(spec, executor, args)
            perf = runner.measure(size_cfg)
            
            display_cfg_dict = {k: v for k, v in cfg.items() if k != 'version'}
            cfg_str = str(display_cfg_dict)
            
            all_results.append({
                "version": display_name, 
                "config_label": cfg_str, 
                "ms": perf['ms'],
                "gbps": perf['gbps'],
                "tflops": perf['tflops'],
                "mbu": perf['mbu'],
                "mfu": perf['mfu']
            })

            latency_str = f"{perf['ms']:.4f} ms"
            mbu_str     = f"{perf['mbu']:.2f}%"
            mfu_str     = f"{perf['mfu']:.2f}%"
            log_master(f"{cfg_str:>30} | {latency_str:>14} | {mbu_str:>12} | {mfu_str:>12}")

    if all_results:
        if not dist.is_initialized() or dist.get_rank() == 0:
            raw_metrics = getattr(args, 'metrics', 'ms')
            target_metrics = _resolve_metrics(raw_metrics)
            for m in target_metrics:
                if m in ["ms", "mbu", "mfu", "bw", "flops"]:
                    plot_tuning_bar(f"{spec.name}_tuning_sz{size_label}", all_results, metric=m)
