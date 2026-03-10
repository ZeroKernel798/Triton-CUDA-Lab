本项目是一个算子开发与性能分析实验室。它涵盖了从基础到前沿的算子实现（CUDA & Triton），也提供了一套基于 Pybind11 的自动化测试和调优框架，同时接入了nsys和ncu工具，辅助进行更深层次性能分析。后续将会增加分布式算子的实现，持续更新中......

```text
├── operators/                              # 算子实现
│   └── 04-matrix-multiplication/     
│       ├── cuda/                           # cuda kernel实现
│       │   ├── native.cu             
│       │   ├── float4.cu             
│       │   ├── flattened_float4.cu   
│       │   └── cublas.cu                   # cublas 
│       ├── triton/                         # triton kernel实现
│       │   └── main.py            
│       └── test_cfg.py                     # 算子的测试案例及参数组合调优
├── requirements.txt    
├── build_lab.py                            # 编译脚本
├── run.sh                                  # 测试脚本
├── run_lab.py
└── utils                                   # 工具类实现
    ├── compiler.py
    ├── logger.py
    ├── prober.py
    └── runner.py
└── README.md
```

## 使用示例
### 编译命令
#### 编译所有算子
python3 build_lab.py --j 8
#### 编译指定算子 支持模糊匹配 支持指定多进程编译
python3 build_lab.py --op 04 --j 8

#### 清除编译产物
python3 build_lab.py --clean 

### 运行测试
##### 全尺寸性能跑分: ./run.sh 04-matrix-multiplication --mode cuda --bench_mode scaling
##### 测试指定尺寸: ./run.sh 04-matrix-multiplication --mode cuda --bench_mode tuning
#### triton&cuda: ./run.sh 04-matrix-multiplication --mode all --bench_mode tuning
#### nsys分析: ./run.sh 04-matrix-multiplication --profile
#### ncu分析: ./run.sh 04-matrix-multiplication --profile --ncu

### GEMM Benchmark:
#### Latency
![alt text](operators/04-matrix-multiplication/scaling_ms_analysis.png)

#### Throughput
![alt text](operators/04-matrix-multiplication/scaling_bw_analysis.png)

#### Compute
![alt text](operators/04-matrix-multiplication/scaling_flops_analysis.png)