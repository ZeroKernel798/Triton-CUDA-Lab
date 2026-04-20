# 向量加法优化
native.cu -> 朴素实现

float4.cu -> vector-additon的计算密度不大，核心思路就是优化访存，采用simd优化即可实现高效vector-addition


