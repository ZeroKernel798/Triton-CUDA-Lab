# 矩阵加法优化
native.cu -> 朴素实现

float4.cu -> 
matrix-additon的计算密度不大，核心思路跟vector-addition类似，用simd优化即可高效优化

flattened_float4.cu ->
跟float4.cu的优化思路类似，但改变了看待输入数据的视角，避免了每行可能都要处理数据不足4个的情况，减少判断逻辑，提高效率

