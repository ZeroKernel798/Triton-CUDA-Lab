# 性能图
# sudo -E $(which python) run_lab.py 04-softmax-attention --mode all
# python3 run_lab.py 04-matrix-multiplication --mode all --bench_mode tuning
CUDA_VISIBLE_DEVICES=5 python3 run_lab.py 04-matrix-multiplication --mode cuda --bench_mode tuning