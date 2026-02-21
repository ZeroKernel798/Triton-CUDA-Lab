#include <torch/extension.h>

__global__ void vector_add_float4_kernel(const float* A, const float* B, float* C, int64_t N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = tid * 4;

    // 处理能够凑齐 float4 的部分
    if (offset + 3 < N) {
        // 强制类型转换为 float4 指针，实现 128-bit 一次性读取
        const float4* A_ptr = reinterpret_cast<const float4*>(&A[offset]);
        const float4* B_ptr = reinterpret_cast<const float4*>(&B[offset]);
        float4* C_ptr = reinterpret_cast<float4*>(&C[offset]);

        float4 a = *A_ptr;
        float4 b = *B_ptr;
        float4 res;

        res.x = a.x + b.x;
        res.y = a.y + b.y;
        res.z = a.z + b.z;
        res.w = a.w + b.w;

        *C_ptr = res;
    } 
    // 处理末尾不足 4 个元素的残余部分 (Tail Handling)
    else if (offset < N) {
        for (int i = offset; i < N; ++i) {
            C[i] = A[i] + B[i];
        }
    }
}

void solve(torch::Tensor A, torch::Tensor B, torch::Tensor C, int64_t N, int block_size) {
    int threadsPerBlock = block_size; 
    // 因为每个线程处理 4 个元素，所以计算 Grid 时要除以 4
    int n_vector = (N + 3) / 4; 
    int blocksPerGrid = (n_vector + threadsPerBlock - 1) / threadsPerBlock;

    vector_add_float4_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("solve", &solve, "Vector Addition Kernel (Pybind11)",
          py::arg("A"), py::arg("B"), py::arg("C"), py::arg("N"), 
          py::arg("block_size")); // 显式命名
}