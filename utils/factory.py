from backend.cuda_impl import CudaExecutor
from backend.nccl_impl import NcclExecutor
from backend.triton_impl import TritonExecutor 
from backend.torch_impl import TorchExecutor 

def get_executor_instance(op_folder, spec, k_info, cfg):
    """
    根据内核类型返回对应的 Executor 实例
    k_info: {'ver': ver, 'path': path, 'type': 'triton'/'cuda'/'nccl'}
    """
    ver, path, k_type = k_info['ver'], k_info['path'], k_info['type']

    if k_type == "torch":
        return TorchExecutor(spec, cfg, ver)
    
    if k_type == "triton":
        return TritonExecutor(op_folder, path, spec, cfg)
    
    if hasattr(spec, 'is_nccl_version') and spec.is_nccl_version(ver):
        return NcclExecutor(op_folder, path, spec, cfg)
        
    return CudaExecutor(op_folder, path, spec, cfg)