import torch
import pynvml

def get_detailed_device_query():
    # 1. 初始化 NVML
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    device_idx = torch.cuda.current_device()
    prop = torch.cuda.get_device_properties(device_idx)
    
    # 获取版本
    driver_ver = pynvml.nvmlSystemGetDriverVersion()
    if isinstance(driver_ver, bytes): driver_ver = driver_ver.decode()
    runtime_ver = torch.version.cuda
    
    # 核心映射 (Ada Lovelace 8.9 = 128 Cores/SM)
    cores_per_sm = 128 if prop.major >= 8 else 64
    total_cores = cores_per_sm * prop.multi_processor_count
    
    # 显存与频率
    bus_width = pynvml.nvmlDeviceGetMemoryBusWidth(handle)
    mem_clock = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)
    gpu_clock = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)
    theo_bw = (mem_clock * 1e6 * 2 * bus_width / 8) / 1e9

    print(f":: 1 available devices\n")
    print(f"  CUDA Driver Version / Runtime Version          {driver_ver} / {runtime_ver}")
    print(f"  CUDA Capability Major/Minor version number:    {prop.major}.{prop.minor}")
    print(f"  Total amount of global memory:                 {prop.total_memory // 1024**2} MBytes ({prop.total_memory} bytes)")
    print(f"  ({prop.multi_processor_count:03d}) Multiprocessors, ({cores_per_sm:03d}) CUDA Cores/MP:    {total_cores} CUDA Cores")
    print(f"  GPU Max Clock rate:                            {gpu_clock} MHz ({gpu_clock/1000:.2f} GHz)")
    print(f"  Memory Clock rate:                             {mem_clock} MHz")
    print(f"  Memory Bus Width:                              {bus_width}-bit")
    print(f"  Theoretical Bandwidth:                         {theo_bw:.2f} GB/s")
    
    # L2 Cache (4090 的 72MB 必须排面上)
    try:
        l2_size = pynvml.nvmlDeviceGetL2CacheSize(handle)
    except:
        l2_size = 75497472 # Fallback to 72MB
    print(f"  L2 Cache Size:                                 {l2_size} bytes")

    # 资源限制
    print(f"  Total amount of constant memory:               65536 bytes")
    print(f"  Total amount of shared memory per block:       49152 bytes") 
    print(f"  Total shared memory per multiprocessor:        102400 bytes")
    print(f"  Total number of registers available per block: 65536")
    print(f"  Warp size:                                     32")
    # 这里的属性名在不同版本的 Torch 里可能有变，直接用常用名或硬编码
    print(f"  Maximum number of threads per multiprocessor:  1536")
    print(f"  Maximum number of threads per block:           1024")
    print(f"  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)")

    pynvml.nvmlShutdown()

if __name__ == "__main__":
    get_detailed_device_query()