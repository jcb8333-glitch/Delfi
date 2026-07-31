
__global__ void matmulKernel(const float* A, const float* B, float* C,int M, int K, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if( row < M && col < N){
        float sum = 0.0f;
        for(int k = 0; k<K; ++k)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

void launchMatmul(const float* A, const float* B, float* C,int M, int K, int N){
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matmulKernel<<<grid, block>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}