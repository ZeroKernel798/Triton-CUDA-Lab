#include <torch/extension.h>
#include <c10/util/BFloat16.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>


#ifndef BLOCK_X
#define BLOCK_X 32
#endif

#ifndef VEC_SIZE
#define VEC_SIZE 4
#endif

#if VEC_SIZE != 4
#error "rowblockvec_smem currently supports VEC_SIZE=4 only"
#endif

__device__ __forceinline__ float warp_sum(float val){
    #pragma unroll
    for(int offset = 16; offset > 0; offset /= 2){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void rmsnorm_rowblock_vec_smem_kernel(
    const __nv_bfloat16* X,
    const __nv_bfloat16* gamma,
    __nv_bfloat16* Y,
    int M,
    int N,
    float eps
) {
    // rowblockvec + shared memory 版本：每个 block 负责一行，先把该行 X 缓存在 shared memory 中。
    int row = blockIdx.x;
    if(row < M){
        extern __shared__ __align__(8) unsigned char smem_raw[];
        __nv_bfloat16* row_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);

        __shared__ float warp_sums[(BLOCK_X + 31) / 32];
        __shared__ float scale_smem;

        int tid = threadIdx.x;
        int lane_id = tid % 32;
        int warp_id = tid / 32;
        int num_warps = (blockDim.x + 31) / 32;

        float sum = 0.0f;
        const __nv_bfloat16* row_x = X + row * N;
        __nv_bfloat16* row_y = Y + row * N;
        bool use_vec = (N % VEC_SIZE == 0);

        if(use_vec){
            const uint2* row_x_vec = reinterpret_cast<const uint2*>(row_x);
            uint2* row_smem_vec = reinterpret_cast<uint2*>(row_smem);
            int n_vec = N / VEC_SIZE;
            for(int vec_id = tid; vec_id < n_vec; vec_id += blockDim.x){
                uint2 x_pack = row_x_vec[vec_id];
                row_smem_vec[vec_id] = x_pack;
                const __nv_bfloat16* x_vec = reinterpret_cast<const __nv_bfloat16*>(&x_pack);

                #pragma unroll
                for(int j = 0; j < VEC_SIZE; ++j){
                    float x = __bfloat162float(x_vec[j]);
                    sum += x * x;
                }
            }
        }
        else{
            for(int i = tid; i < N; i += blockDim.x){
                __nv_bfloat16 x_bf16 = row_x[i];
                row_smem[i] = x_bf16;
                float x = __bfloat162float(x_bf16);
                sum += x * x;
            }
        }

        __syncthreads();

        sum = warp_sum(sum);

        if(lane_id == 0){
            warp_sums[warp_id] = sum;
        }
        __syncthreads();

        float block_sum = 0.0f;
        if(warp_id == 0){
            block_sum = lane_id < num_warps ? warp_sums[lane_id] : 0.0f;
            block_sum = warp_sum(block_sum);
        }

        if(tid == 0){
            scale_smem = 1.0f / sqrtf(block_sum / N + eps);
        }
        __syncthreads();

        float scale = scale_smem;
        if(use_vec){
            const uint2* row_smem_vec = reinterpret_cast<const uint2*>(row_smem);
            const uint2* gamma_vec = reinterpret_cast<const uint2*>(gamma);
            uint2* row_y_vec = reinterpret_cast<uint2*>(row_y);
            int n_vec = N / VEC_SIZE;
            for(int vec_id = tid; vec_id < n_vec; vec_id += blockDim.x){
                uint2 x_pack = row_smem_vec[vec_id];
                uint2 g_pack = gamma_vec[vec_id];
                uint2 y_pack;

                const __nv_bfloat16* x_vec = reinterpret_cast<const __nv_bfloat16*>(&x_pack);
                const __nv_bfloat16* g_vec = reinterpret_cast<const __nv_bfloat16*>(&g_pack);
                __nv_bfloat16* y_vec = reinterpret_cast<__nv_bfloat16*>(&y_pack);

                #pragma unroll
                for(int j = 0; j < VEC_SIZE; ++j){
                    float x = __bfloat162float(x_vec[j]);
                    float g = __bfloat162float(g_vec[j]);
                    y_vec[j] = __float2bfloat16(x * scale * g);
                }

                row_y_vec[vec_id] = y_pack;
            }
        }
        else{
            for(int i = tid; i < N; i += blockDim.x){
                float x = __bfloat162float(row_smem[i]);
                float g = __bfloat162float(gamma[i]);
                row_y[i] = __float2bfloat16(x * scale * g);
            }
        }
    }
}


void solve(
    torch::Tensor X,
    torch::Tensor gamma,
    torch::Tensor Y,
    int M,
    int N,
    float eps
) {
    // 一个 block 负责一行，X 行数据只从 global memory 读取一次，之后从 shared memory 复用。
    dim3 threadsPerBlock(BLOCK_X);
    dim3 blocksPerGrid(M);
    size_t shared_bytes = static_cast<size_t>(N) * sizeof(__nv_bfloat16);

    rmsnorm_rowblock_vec_smem_kernel<<<blocksPerGrid, threadsPerBlock, shared_bytes>>>(
        reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<c10::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr<c10::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(Y.data_ptr<c10::BFloat16>()),
        M,
        N,
        eps
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("solve", &solve, "RMSNorm Row Block Vectorized Shared Memory Kernel",
          py::arg("X"),
          py::arg("gamma"),
          py::arg("Y"),
          py::arg("M"),
          py::arg("N"),
          py::arg("eps"));
}
