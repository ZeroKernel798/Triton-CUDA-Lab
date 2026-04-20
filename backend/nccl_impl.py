import os
import torch
import torch.distributed as dist
from backend.cuda_impl import CudaExecutor

class NcclExecutor(CudaExecutor):
    def __init__(self, op_name, kernel_file, spec, config):
        # 1. 解决设备冲突：必须在 init_process_group 之前 set_device
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
            
        super().__init__(op_name, kernel_file, spec, config)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def run(self, inputs):
        if self.module is None: self.compile()
        
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        input_tensor = inputs['input'].to(device)
        output_tensor = inputs['output'].to(device)
        N, k = inputs['N'], inputs['k']

        # --- 关键修改：数据分片 ---
        # 分布式算子的意义在于每张卡只算一部分。
        # 我们假设 input_tensor 是全量数据，每张卡只负责其中一段。
        chunk_size = (N + self.world_size - 1) // self.world_size
        start_idx = self.rank * chunk_size
        end_idx = min(start_idx + chunk_size, N)
        
        # 截取本卡负责的片段（如果数据太少，保证不越界）
        local_input = input_tensor[start_idx:end_idx] if start_idx < N else input_tensor[:0]
        local_N = local_input.size(0)

        # 3. 运行本地 C++ 算子 (只算本卡片段的 Top-K)
        # 注意：这里的 N 要传 local_N
        self.module.solve(local_input, output_tensor, local_N, k, self.rank, self.world_size)
        
        # 4. 跨卡决赛
        if self.world_size > 1:
            gather_list = [torch.empty_like(output_tensor) for _ in range(self.world_size)]
            dist.all_gather(gather_list, output_tensor)
            
            all_candidates = torch.cat(gather_list)
            final_topk_values, _ = torch.topk(all_candidates, k=k)
            output_tensor.copy_(final_topk_values)
            
        return output_tensor

    def benchmark(self, inputs, warmup=10, iters=50):
        """
        分布式版 Benchmarking：
        弃用 do_bench（它内部的同步机制和 NCCL 冲突），改用 cuda.Event 计时
        """
        if self.module is None: self.compile()

        # 预热
        for _ in range(warmup):
            self.run(inputs)
        
        torch.cuda.synchronize()
        dist.barrier() # 确保所有卡节奏一致

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(iters):
            self.run(inputs)
        end_event.record()

        torch.cuda.synchronize()
        dist.barrier()

        # 返回平均毫秒数
        return start_event.elapsed_time(end_event) / iters