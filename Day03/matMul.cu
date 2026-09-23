#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include "../common/cuda_check.h"

__global__
void matMulKernel(
    float* M, float* N, float *P, int width
){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if((row < width) && (col < width)){
        float Pvalue = 0;
        for (int k=0;k<width;++k){
            Pvalue += M[row * width + k] * N[k * width + col];
        }
        P[row * width + col] = Pvalue;
    }
}

int main() {
    int width = 64; // square W only
    int size = width * width * sizeof(float);

    float *h_M = (float*)malloc(size);
    float *h_N = (float*)malloc(size);
    float *h_P = (float*)malloc(size);
    for (int i = 0; i < width * width; i++) {
        h_M[i] = (float)((i % 13) * 0.5f);
        h_N[i] = (float)((i % 7) * 0.5f);
    }

    float *d_M, *d_N, *d_P;
    CUDA_CHECK(cudaMalloc((void**)&d_M, size));
    CUDA_CHECK(cudaMalloc((void**)&d_N, size));
    CUDA_CHECK(cudaMalloc((void**)&d_P, size));
    CUDA_CHECK(cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (width + block.y - 1) / block.y);
    matMulKernel<<<grid, block>>>(d_M, d_N, d_P, width);
    KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost));

    for (int row = 0; row < width; row++) {
        for (int col = 0; col < width; col++) {
            float expected = 0;
            for (int k = 0; k < width; k++)
                expected += h_M[row * width + k] * h_N[k * width + col];
            float got = h_P[row * width + col];
            if (fabsf(got - expected) > 1e-3f) {
                printf("FAIL at (%d,%d): got %f expected %f\n", row, col, got, expected);
                return 1;
            }
        }
    }
    printf("PASS: %dx%d matmul correct\n", width, width);

    CUDA_CHECK(cudaFree(d_M));
    CUDA_CHECK(cudaFree(d_N));
    CUDA_CHECK(cudaFree(d_P));
    free(h_M);
    free(h_N);
    free(h_P);
    return 0;
}