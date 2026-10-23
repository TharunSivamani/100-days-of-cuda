#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include "../common/cuda_check.h"
#include <time.h>

__global__
void matmul(float* M, float* N, float* P, int width){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if ((row < width) && (col < width)){
        float Pvalue = 0;
        for(int k=0;k<width;++k){
            Pvalue += M[row*width+k] * N[k*width+col];
        }
        P[row*width+col] = Pvalue;
    }
}

int main(){
    int width = 1024;
    int size = sizeof(float) * width * width;

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

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    matmul<<<grid, block>>>(d_M, d_N, d_P, width);
    KERNEL_CHECK();
    CUDA_CHECK(cudaEventRecord(t0));
    matmul<<<grid, block>>>(d_M, d_N, d_P, width);
    KERNEL_CHECK();
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    printf("TIME: %.3f ms\n", ms);
    printf("GFLOPS: %.2f\n", (2.0*width*width*width) / (ms * 1e6));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));

    CUDA_CHECK(cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost));

    clock_t start = clock();
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < width; col++) {
            float expected = 0;
            for (int k = 0; k < width; k++)
                expected += h_M[row * width + k] * h_N[k * width + col];
            float got = h_P[row * width + col];
            if (fabsf(got - expected) > 1e-3f * expected) {
                printf("FAIL at (%d,%d): got %f expected %f\n", row, col, got, expected);
                return 1;
            }
        }
    }
    double cpu_sec = (double)(clock() - start) / CLOCKS_PER_SEC;
    double cpu_gflops = 2.0 * width * width * width / (cpu_sec * 1e9);
    printf("CPU: %.3f s -> %.1f GFLOPS\n", cpu_sec, cpu_gflops);

    printf("PASS: %dx%d matmul correct\n", width, width);

    CUDA_CHECK(cudaFree(d_M));
    CUDA_CHECK(cudaFree(d_N));
    CUDA_CHECK(cudaFree(d_P));
    free(h_M);
    free(h_N);
    free(h_P);
    return 0;
}