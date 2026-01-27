#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <device_launch_parameters.h>

// -----------------------------------------------------------------------
// Flash Attention Kernel
// 支持任意 M, N, d (带边界检查)
// -----------------------------------------------------------------------
template<int Br, int Bc, int max_d>
__global__ void flash_attn_kernel(
    const float* Q, const float* K, const float* V, float* O,
    int M, int N, int d, float scale) 
{
    // 1. 声明共享内存 (Q 固定一块, K/V 滚动)
    // 使用一维数组避免二维索引的复杂对齐问题
    extern __shared__ float s_mem[];
    float* s_Q = s_mem;                  // 大小: Br * d
    float* s_K = s_mem + Br * d;         // 大小: Bc * d
    float* s_V = s_mem + (Br + Bc) * d;  // 大小: Bc * d

    int tid = threadIdx.x; // 这里建议用 1D 线程布局 (Br 个线程)，每人负责一行 Q
    int bid = blockIdx.x;
    int q_row_start = bid * Br;
    int row = q_row_start + tid;

    // 2. 寄存器初始化 (每个线程维护自己这一行的 Online Softmax 状态)
    float m_prev = -FLT_MAX;
    float l_prev = 0.0f;
    float acc[max_d]; // 寄存器中缓存这一行的结果
    #pragma unroll
    for (int k = 0; k < max_d; k++) acc[k] = 0.0f;

    // 3. 加载 Q 到 Shared Memory (协作加载)
    // 每个线程搬运 Q 的一部分数据
    for (int i = tid; i < Br * d; i += Br) {
        int r = i / d;
        int c = i % d;
        if (q_row_start + r < M && c < d)
            s_Q[r * d + c] = Q[(q_row_start + r) * d + c];
        else
            s_Q[r * d + c] = 0.0f;
    }
    __syncthreads();

    // 4. 外层大循环：遍历 K, V 的所有分块 (Tiles)
    for (int j = 0; j < N; j += Bc) {
        
        // 协作加载 K 和 V 块到 Shared Memory
        for (int i = tid; i < Bc * d; i += Br) {
            int r = i / d;
            int c = i % d;
            if (j + r < N && c < d) {
                s_K[r * d + c] = K[(j + r) * d + c];
                s_V[r * d + c] = V[(j + r) * d + c];
            } else {
                s_K[r * d + c] = 0.0f;
                s_V[r * d + c] = 0.0f;
            }
        }
        __syncthreads();

        // 计算当前线程负责的这一行 Q 与当前 K 块的所有点积
        if (row < M) {
            for (int t = 0; t < Bc; t++) {
                if (j + t >= N) break; // 边界检查

                // 计算 S = Q_i * K_t^T
                float score = 0.0f;
                for (int k = 0; k < d; k++) {
                    score += s_Q[tid * d + k] * s_K[t * d + k];
                }
                score *= scale;

                // --- Online Softmax 核心步骤 ---
                // $m_{new} = \max(m_{old}, score)$
                float m_new = fmaxf(m_prev, score);
                // $e^{score - m_{new}}$
                float p = __expf(score - m_new);
                // 调整系数 $\alpha = e^{m_{old} - m_{new}}$
                float alpha = __expf(m_prev - m_new);

                // 更新归一化因子 $l_{new} = l_{old} \cdot \alpha + p$
                l_prev = l_prev * alpha + p;

                // 更新结果累加器 $acc = acc \cdot \alpha + p \cdot V_t$
                for (int k = 0; k < d; k++) {
                    acc[k] = acc[k] * alpha + p * s_V[t * d + k];
                }
                m_prev = m_new;
            }
        }
        __syncthreads();
    }

    // 5. 写回结果到全局显存
    if (row < M) {
        for (int k = 0; k < d; k++) {
            O[row * d + k] = acc[k] / l_prev;
        }
    }
}

// -----------------------------------------------------------------------
// 包装器函数 (Host Side)
// -----------------------------------------------------------------------
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, 
                      int M, int N, int d) 
{
    const int Br = 32; // 每个 Block 处理 32 行 Q
    const int Bc = 32; // 每次加载 32 行 K, V

    // 修正 1：Grid 尺寸必须向上取整，否则小矩阵会启动 0 个 Block
    dim3 grid((M + Br - 1) / Br);
    dim3 block(Br); // 简单的 1D 布局，每个线程负责一行

    // 修正 2：根据 d 动态计算共享内存需求
    size_t smem_size = (Br * d + Bc * d + Bc * d) * sizeof(float);

    float scale = 1.0f / sqrtf((float)d);

    // 修正 3：针对测试用例 d=4，通过模板传递 max_d（实际可设为 128 或动态）
    // 为了通过你的测试，这里硬编码 max_d 为 128 (足够覆盖 d=4)
    flash_attn_kernel<Br, Bc, 128><<<grid, block, smem_size>>>(
        Q, K, V, output, M, N, d, scale
    );

}