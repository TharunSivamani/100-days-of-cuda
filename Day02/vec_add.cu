#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include "../common/cuda_check.h"

__global__
void vecAddKernel(float *A, float *B, float *C, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i<N) C[i] = A[i] + B[i];
}

void vecAdd(float* A, float* B, float* C, int n){
    float* A_d, *B_d, *C_d;
    int size = n * sizeof(float);

    CUDA_CHECK(cudaMalloc((void**) &A_d, size));
    CUDA_CHECK(cudaMalloc((void**) &B_d, size));
    CUDA_CHECK(cudaMalloc((void**) &C_d, size));

    CUDA_CHECK(cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice));

    // vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);
    vecAddKernel<<<(n+255)/256, 256>>>(A_d, B_d, C_d, n);
    KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
}

int main(){
    int n = 8000;
    int size = n * sizeof(float);

    float *A = (float*)malloc(size);
    float *B = (float*)malloc(size);
    float *C = (float*)malloc(size);

    for (int i = 0; i < n; i++) {
        A[i] = (float)i;
        B[i] = (float)(2 * i);
    }

    vecAdd(A, B, C, n);

    for (int i = 0; i < n; i++) {
        float expected = A[i] + B[i];
        if (C[i] != expected) {
            printf("FAIL at %d: got %f, expected %f\n", i, C[i], expected);
            return 1;
        }
    }
    printf("PASS: all %d elements correct\n", n);

    free(A);
    free(B);
    free(C);
    return 0;
}