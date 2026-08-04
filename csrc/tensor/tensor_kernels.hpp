#pragma once
#ifdef DLF_CUDA
void launchTMul(const  float* A, const float* B, float* C, size_t M, size_t K, size_t N);
void launchTAdd(const  float* A, const float* B, float* C, size_t nRows, size_t nCols);
#endif