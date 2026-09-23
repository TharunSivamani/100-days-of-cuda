#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include "../common/cuda_check.h"

__global__
void dot_vec(float *A, float *B, float *C, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < N){
        float sum=0;
        for(int k=0;k<N;k++){
            sum+=B[row*N+k] * C[k];
        }
        A[row] = sum;
    }
}

int main() {
    int N = 1000; // non-multiple of 256 to test guard
    int matSize = N * N * sizeof(float);
    int vecSize = N * sizeof(float);

    float *h_B = (float*)malloc(matSize);
    float *h_C = (float*)malloc(vecSize);
    float *h_A = (float*)malloc(vecSize);
    for (int i = 0; i < N * N; i++) h_B[i] = (float)((i % 13) * 0.5f);
    for (int i = 0; i < N; i++) h_C[i] = (float)((i % 7) * 0.5f);

    float *d_B, *d_C, *d_A;
    CUDA_CHECK(cudaMalloc((void**)&d_B, matSize));
    CUDA_CHECK(cudaMalloc((void**)&d_C, vecSize));
    CUDA_CHECK(cudaMalloc((void**)&d_A, vecSize));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, matSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, vecSize, cudaMemcpyHostToDevice));

    // kernel indexes y, so size the y-dimension
    dim3 block(1, 256);
    dim3 grid(1, (N + block.y - 1) / block.y);
    dot_vec<<<grid, block>>>(d_A, d_B, d_C, N);
    KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(h_A, d_A, vecSize, cudaMemcpyDeviceToHost));

    for (int row = 0; row < N; row++) {
        float expected = 0;
        for (int k = 0; k < N; k++) expected += h_B[row * N + k] * h_C[k];
        if (fabsf(h_A[row] - expected) > 1e-3f) {
            printf("FAIL at %d: got %f expected %f\n", row, h_A[row], expected);
            return 1;
        }
    }
    printf("PASS: N=%d mat-vec correct\n", N);

    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_A));
    free(h_B);
    free(h_C);
    free(h_A);
    return 0;
}