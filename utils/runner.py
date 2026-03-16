import torch
import torch.distributed as dist
from utils import loader 
from utils.factory import get_executor_instance
from utils.strategies import run_profile_strategy, run_scaling_strategy, run_tuning_strategy
from core.oprunner import OperatorRunner, log_master

# 测试分发 
def dispatch(args):
    target_ops = loader.find_matched_ops(args.op)
    
    for op_folder in target_ops:
        spec = loader.get_spec(op_folder)
        if not spec: continue
        log_master(f"\n🧬 [Operator Lab] {spec.name}")

        kernel_infos = loader.get_kernel_files(op_folder)
        kernel_infos.append({
        'ver': 'official', 
        'path': None,      # 官方版本不需要文件路径
        'type': 'cuda'     # 标记为 cuda 类型，这样它会去搜 cuda_tuning_configs
    })
        
        valid_kernels = {} 
        
        if not args.skip_val:
            log_master("🧪 正在执行内核逻辑准入验证...")
            rank = dist.get_rank() if dist.is_initialized() else 0

            for k_info in kernel_infos:
                ver, k_type = k_info['ver'], k_info['type']
                
                is_dist_ver = spec.is_nccl_version(ver)
                if not is_dist_ver and rank != 0:
                    continue
                
                if k_type == "triton":
                    cfgs = getattr(spec, 'tuning_configs', [])
                else:
                    cfgs = [c for c in getattr(spec, 'cuda_tuning_configs', []) if c.get("version") == ver]
                
                test_cfg = cfgs[0] if cfgs else {}
                
                try:
                    executor = get_executor_instance(op_folder, spec, k_info, test_cfg)
                    if OperatorRunner(spec, executor, args).validate():
                        valid_kernels[ver] = k_info
                        log_master(f"   ✅ {ver} [{k_info['type']}] 通过验证")
                    else:
                        log_master(f"   ❌ {ver} [{k_info['type']}] 验证失败")
                except Exception as e:
                    log_master(f"   💥 {ver} [{k_info['type']}] 初始化异常: {e}")
        else:
            valid_kernels = {k['ver']: k for k in kernel_infos}

        # 执行后续策略
        if valid_kernels:
            if args.mode == "scaling":
                run_scaling_strategy(spec, valid_kernels, args, op_folder)
            elif args.mode == "tuning":
                run_tuning_strategy(spec, valid_kernels, args, op_folder)
            elif args.mode == "profile": 
                run_profile_strategy(spec, valid_kernels, args, op_folder)