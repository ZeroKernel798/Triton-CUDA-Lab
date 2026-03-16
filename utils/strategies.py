import torch.distributed as dist
from core.oprunner import OperatorRunner, log_master
from utils.logger import plot_scaling_line, plot_tuning_bar
from utils.factory import get_executor_instance

# 测试方案 
def run_profile_strategy(spec, valid_kernels, args, op_folder):
    """Profile 策略：自动寻找最优配置并执行 Profiler 插点"""
    # 确定测试规模 (RxC)
    raw_input = getattr(spec, 'perf_input', None)
    size_cfg = raw_input[0] if isinstance(raw_input, list) else raw_input

    if isinstance(size_cfg, dict):
        size_label = "x".join([str(v) for v in size_cfg.values()])
    else:
        size_label = str(size_cfg)
        first_x_key = list(spec.x_vals[0].keys())[0]
        size_cfg = {first_x_key: size_cfg}

    # 处理多卡启动逻辑
    rank = dist.get_rank() if dist.is_initialized() else 0

    # 注意这里的 value 变成了 k_info
    for ver, k_info in valid_kernels.items():
        # 如果不是分布式版本，且我不是 Rank 0 -> 我直接跳过，不参与该版本的测试
        is_dist_ver = spec.is_nccl_version(ver)
        if not is_dist_ver and rank != 0:
            continue

        log_master(f"\n🔍 [Profile Mode] 正在为 {ver} 寻找最佳配置...")
        
        # 根据内核类型进行优化
        k_type = k_info.get('type', '')
        if k_type == "triton":
            configs = getattr(spec, 'tuning_configs', [])
        else:
            configs = [c for c in getattr(spec, 'cuda_tuning_configs', []) if c.get("version") == ver]
        if not configs:
            configs = [{}]
        best_res = {"ms": float('inf'), "cfg": None}

        for cfg in configs:
            # 使用 get_executor_instance 替代 exec_cls
            executor = get_executor_instance(op_folder, spec, k_info, cfg)
            runner = OperatorRunner(spec, executor, args)
            perf = runner.measure(size_cfg)
            if perf["ms"] < best_res["ms"]:
                best_res.update(perf)
                best_res["cfg"] = cfg
        
        best_cfg = best_res["cfg"]
        log_master(f"🏆 找到最佳配置: {best_cfg} (Latency: {best_res['ms']:.4f} ms)")

        # 重新实例化一个带冠军配置的 Executor 
        best_executor = get_executor_instance(op_folder, spec, k_info, best_cfg)
        profiler_runner = OperatorRunner(spec, best_executor, args)
        
        log_master(f"🎬 准备就绪，开始对规模 {size_label} 进行 Profiling...")
        profiler_runner.profile(size_cfg)


def run_scaling_strategy(spec, valid_kernels, args, op_folder):
    """scaling 测试逻辑：评估不同规模下的性能走向"""
    all_results = [] 

    rank = dist.get_rank() if dist.is_initialized() else 0

    # 注意这里的 value 变成了 k_info
    for ver, k_info in valid_kernels.items():
        # 如果不是分布式版本，且我不是 Rank 0 -> 我直接跳过，不参与该版本的测试
        is_dist_ver = spec.is_nccl_version(ver)
        if not is_dist_ver and rank != 0:
            continue

        log_master(f"\n🚀 正在评估内核实现: [{ver}]")
        
        header = f"{'Size (RxC)':>15} | {'Latency':>14} | {'Throughput':>18} | {'Compute':>15} | Best Config"
        log_master(header)
        log_master("   " + "-" * (len(header) + 10))

        # 根据内核类型选择参数
        k_type = k_info.get('type', '')
        if k_type == "triton":
            configs = getattr(spec, 'tuning_configs', [])
        else:
            configs = [c for c in getattr(spec, 'cuda_tuning_configs', []) if c.get("version") == ver]
        if not configs:
            configs = [{}]
        best_res = {"ms": float('inf'), "cfg": None}

        for size_cfg in spec.x_vals:
            best_res = {"ms": float('inf'), "gbps": 0.0, "tflops": 0.0, "cfg": None}
            
            display_size = "x".join([str(v) for v in size_cfg.values()])
            total_elements = 1
            for v in size_cfg.values(): total_elements *= v

            for cfg in configs:
                # 使用 get_executor_instance 替代 exec_cls
                executor = get_executor_instance(op_folder, spec, k_info, cfg)
                runner = OperatorRunner(spec, executor, args)
                perf = runner.measure(size_cfg)
                if perf["ms"] < best_res["ms"]:
                    best_res.update(perf)
                    best_res["cfg"] = cfg

            # 记录数据点
            all_results.append({
                "version": ver,
                "size_label": display_size,  # 绘图显示的字符串
                "size_val": total_elements, # 排序用的数值
                "ms": best_res["ms"],
                "gbps": best_res["gbps"],
                "tflops": best_res["tflops"]
            })

            latency_str = f"{best_res['ms']:.4f} ms"
            tp_str      = f"{best_res['gbps']:.2f} GB/s"
            flops_str   = f"{best_res['tflops']:.2f} TFLOPS"
            display_cfg = {k: v for k, v in best_res['cfg'].items() if k != 'version'}
            
            log_master(f"{display_size:>15} | {latency_str:>14} | {tp_str:>18} | {flops_str:>15} | {display_cfg}")

    if all_results:
        if not dist.is_initialized() or dist.get_rank() == 0:
            raw_metrics = getattr(args, 'metrics', 'ms')
            if raw_metrics.lower() == "all":
                target_metrics = ["ms", "bw", "flops"]
            else:
                target_metrics = [m.strip() for m in raw_metrics.split(',')]

            for m in target_metrics:
                if m in ["ms", "bw", "flops"]:
                    plot_scaling_line(f"{spec.name}_scaling", all_results, metric=m)


def run_tuning_strategy(spec, valid_kernels, args, op_folder):
    """tuning 测试模式：对特定规模进行全量配置扫描"""
    all_results = [] 

    # 提取默认参数 (保持你原来的标准化逻辑)
    raw_input = getattr(spec, 'perf_input', None)
    if raw_input is None: return
    size_cfg = raw_input[0] if isinstance(raw_input, list) else raw_input
    if isinstance(size_cfg, dict):
        size_label = list(size_cfg.values())[0]
    else:
        size_label = size_cfg
        size_cfg = {list(spec.x_vals[0].keys())[0]: size_cfg}

    rank = dist.get_rank() if dist.is_initialized() else 0
    
    # 注意这里的 value 变成了 k_info
    for ver, k_info in valid_kernels.items():
        # 如果不是分布式版本，且我不是 Rank 0 -> 我直接跳过，不参与该版本的测试
        is_dist_ver = spec.is_nccl_version(ver)
        if not is_dist_ver and rank != 0:
            continue

        log_master(f"\n🔍 [Tuning Mode] 正在深度评估内核: [{ver}]")
        log_master(f"📍 Target Size: {size_label}")
        
        header = f"{'Config':>30} | {'Latency':>14} | {'Throughput':>18} | {'Compute':>15}"
        log_master(header)
        log_master("-" * len(header))

        # 根据内核类型选择配置
        k_type = k_info.get('type', '')
        if k_type == "triton":
            configs = getattr(spec, 'tuning_configs', [])
        else:
            configs = [c for c in getattr(spec, 'cuda_tuning_configs', []) if c.get("version") == ver]
        if not configs:
            configs = [{}]
        best_res = {"ms": float('inf'), "cfg": None}
        
        for idx, cfg in enumerate(configs):
            # 使用 get_executor_instance 替代 exec_cls
            executor = get_executor_instance(op_folder, spec, k_info, cfg)
            runner = OperatorRunner(spec, executor, args)
            perf = runner.measure(size_cfg)
            
            display_cfg_dict = {k: v for k, v in cfg.items() if k != 'version'}
            cfg_str = str(display_cfg_dict)
            
            # --- 关键修改：存储用于柱状图的标签 ---
            all_results.append({
                "version": ver,      
                "config_label": cfg_str, # 将配置字典字符串存入，作为柱状图横轴
                "ms": perf['ms'],
                "gbps": perf['gbps'],
                "tflops": perf['tflops']
            })

            latency_str = f"{perf['ms']:.4f} ms"
            tp_str      = f"{perf['gbps']:.2f} GB/s"
            flops_str   = f"{perf['tflops']:.2f} TFLOPS"
            log_master(f"{cfg_str:>30} | {latency_str:>14} | {tp_str:>18} | {flops_str:>15}")

    if all_results:
        if not dist.is_initialized() or dist.get_rank() == 0:
            raw_metrics = getattr(args, 'metrics', 'ms')
            target_metrics = ["ms", "bw", "flops"] if raw_metrics.lower() == "all" else [m.strip() for m in raw_metrics.split(',')]

            for m in target_metrics:
                if m in ["ms", "bw", "flops"]:
                    plot_tuning_bar(f"{spec.name}_tuning_sz{size_label}", all_results, metric=m)

