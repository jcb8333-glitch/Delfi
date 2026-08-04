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

__global__ void taddKernel(const float* A, const float* B, float* C, size_t nRows, size_t nCols){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < nRows && col < numCols){
        int index = row * numCols + col;
        c[index] = A[index] + B[index];
    }
}

void launchTMul(const float* A, const float* B, float* C,size_t M, size_t K, size_t N){
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    tmulKernel<<<grid, block>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}

void launchTAdd(const float* A, const float* B, float* C, size_t nRows, size_t nCols){
    dim3 block(16, 16);
    dim3 grid((nCols + block.x-1)/block.x, (nRows + block.y-1)/block.y);
    taddKernel<<<grid, block>>>(A, B, C, nRows, nCols);
    cudaDeviceSynchronize();
}