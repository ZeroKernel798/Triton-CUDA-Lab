#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void matrix_add_kernel(const float* A, const float* B, float* C, int N) 
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    
    if(x < N && y < N){
        C[y * N + x] = A[y * N + x] + B[y * N + x];
    }
}

void solve(torch::Tensor A, torch::Tensor B, torch::Tensor C, int N, int bx, int by) {
    // 获取设备指针
    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    // 计算网格规模
    dim3 threadsPerBlock(bx, by);
    dim3 blocksPerGrid((N + bx - 1) / bx, (N + by - 1) / by);

    // 启动内核
    matrix_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("solve", &solve, "2D Matrix Addition",
          py::arg("A"), py::arg("B"), py::arg("C"), py::arg("N"), 
          py::arg("bx"), py::arg("by")); // 显式命名
}