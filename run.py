import argparse
from utils import runner
import logging

logging.getLogger("torch._inductor.select_algorithm").setLevel(logging.ERROR)

def main():
    parser = argparse.ArgumentParser(description="Triton-CUDA-Lab V2")
    parser.add_argument("--op", type=str, required=True, help="算子名模糊匹配")
    parser.add_argument("--skip-val", action="store_true", help="跳过精度校验")
    parser.add_argument("--mode", choices=["scaling", "tuning", "profile"], default="scaling")
    parser.add_argument("--metrics", type=str, default="ms", help="'ms', 'bw', 'flops' or 'all'")
    
    args = parser.parse_args()
    
    # 直接交给调度层处理
    runner.dispatch(args)

if __name__ == "__main__":
    main()