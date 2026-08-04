#include <stdexcept>
#include <string>

#ifdef DLF_CUDA
    #include <cuda_runtime.h>
#endif

__global__ void tmulKernel(const float* A, const float* B, float* C, size_t M, size_t K, size_t N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if( row < M && col < N){
        float sum = 0.0f;
        for(int k = 0; k < K; ++k)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

__global__ void taddKernel(const float* A, const float* B, float* C, size_t n){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<n){
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void tsubKernel(const float* A, const float* B, float* C, size_t n){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<n){
        C[idx] = A[idx] - B[idx];
    }
}

void launchTMul(const float* A, const float* B, float* C,size_t M, size_t K, size_t N){
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    tmulKernel<<<grid, block>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}

void launchTAdd(const float* A, const float* B, float* C, size_t n){
    unsigned int threads = 256;
    unsigned int blocks = (unsigned int)((n + threads - 1) / threads);
    taddKernel<<<blocks, threads>>>(A, B, C, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(std::string("TAddKernel launch failed") + cudaGetErrorString(err));
    cudaDeviceSynchronize();
    if (err != cudaSuccess) throw std::runtime_error("TAddKernel execution failed");
}

void launchTSub(const float* A, const float* B, float* C, size_t n){
    unsigned int threads = 256;
    unsigned int blocks = (unsigned int)((n + threads - 1) / threads);
    taddKernel<<<blocks, threads>>>(A, B, C, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(std::string("TSubKernel launch failed") + cudaGetErrorString(err));
    cudaDeviceSynchronize();
    if (err != cudaSuccess) throw std::runtime_error("TSubKernel execution failed");
}