#include <cuda_runtime.h>
#include <iostream>

static inline void checkCuda(cudaError_t err, const char* msg = "") {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " : " << cudaGetErrorString(err) << std::endl;
        std::abort();
    }
}

__global__ void matMulKernel(const double* A, const double* B, double* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// wrapper: times kernel using cuda events and returns elapsed ms in gpu_ms
extern "C" void gpu_multiply_raw(const double* Arow, const double* Brow, double* Crow, int N, float &gpu_ms) {
    size_t bytes = sizeof(double) * size_t(N) * size_t(N);
    double *dA=nullptr, *dB=nullptr, *dC=nullptr;
    checkCuda(cudaMalloc((void**)&dA, bytes), "cudaMalloc dA");
    checkCuda(cudaMalloc((void**)&dB, bytes), "cudaMalloc dB");
    checkCuda(cudaMalloc((void**)&dC, bytes), "cudaMalloc dC");

    checkCuda(cudaMemcpy(dA, Arow, bytes, cudaMemcpyHostToDevice), "H2D A");
    checkCuda(cudaMemcpy(dB, Brow, bytes, cudaMemcpyHostToDevice), "H2D B");

    dim3 block(16,16);
    dim3 grid( (N + block.x - 1)/block.x, (N + block.y - 1)/block.y );

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "Create start");
    checkCuda(cudaEventCreate(&stop), "Create stop");
    checkCuda(cudaEventRecord(start), "Record start");

    matMulKernel<<<grid, block>>>(dA, dB, dC, N);
    // check kernel
    checkCuda(cudaGetLastError(), "Kernel launch");

    checkCuda(cudaEventRecord(stop), "Record stop");
    checkCuda(cudaEventSynchronize(stop), "Synchronize stop");
    checkCuda(cudaEventElapsedTime(&gpu_ms, start, stop), "ElapsedTime");

    checkCuda(cudaMemcpy(Crow, dC, bytes, cudaMemcpyDeviceToHost), "D2H C");

    checkCuda(cudaEventDestroy(start), "Destroy start");
    checkCuda(cudaEventDestroy(stop), "Destroy stop");
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}
