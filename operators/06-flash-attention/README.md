# flashattention优化

flashattentionv1.cu ->
flashattentionv1版本，循环顺序是KVQ，需要额外存储每一轮的计算结果，显存访问较多。


flashattentionv2.cu ->
flashattentionv2版本，循环顺序是QKV，每一行Q会滑动处理完整个K和V矩阵，直接得到最后的输出，数据留存在寄存器的时间大大增加，效率更高，尤其是对访存的优化。

