#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void matrix_add_vec_kernel(const float* A, const float* B, float* C, int Ne) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = tid * 4;

    if (offset + 3 < Ne) {
        // 利用 tid 索引 float4 指针，实现 128-bit 向量化加载/存储
        // [Image of CUDA float4 memory alignment and vectorized access]
        const float4* Av_ptr = reinterpret_cast<const float4*>(A);
        const float4* Bv_ptr = reinterpret_cast<const float4*>(B);
        float4* Cv_ptr = reinterpret_cast<float4*>(C);

        float4 Av = Av_ptr[tid];
        float4 Bv = Bv_ptr[tid];
        
        float4 Cv;
        Cv.x = Av.x + Bv.x;
        Cv.y = Av.y + Bv.y;
        Cv.z = Av.z + Bv.z;
        Cv.w = Av.w + Bv.w;

        Cv_ptr[tid] = Cv;
    } 
    else if (offset < Ne) {
        // 边界处理：处理末尾不足 4 个的部分
        for (int i = offset; i < Ne; i++) {
            C[i] = A[i] + B[i];
        }
    }
}

void solve(torch::Tensor A, torch::Tensor B, torch::Tensor C, int N, int block_size) {
    // 自动解包为扁平化后的总元素数量 Ne = N * N
    int Ne = N * N;
    
    // 提取 raw 指针
    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    // 计算执行配置
    int threadsPerBlock = block_size;
    int n_vec = (Ne + 3) / 4; // 计算需要多少个 float4 向量
    int blocksPerGrid = (n_vec + threadsPerBlock - 1) / threadsPerBlock;

    // 启动内核
    matrix_add_vec_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, Ne);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("solve", &solve, "1D Flattened Matrix Addition",
          py::arg("A"), py::arg("B"), py::arg("C"), py::arg("N"), 
          py::arg("block_size")); // 显式命名
}