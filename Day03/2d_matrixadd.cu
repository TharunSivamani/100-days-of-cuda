#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include "../common/cuda_check.h"

__global__
void addMatrix(float* A, float* B, float *C, int width){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < width && col < width){
        C[row*width + col] = A[row*width + col] + B[row*width + col];
    }
}

int main(){
    int width = 100; // non-multiple of 16 to test guard
    int size = width * width * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    for (int i = 0; i < width * width; i++) {
        h_A[i] = (float)i;
        h_B[i] = (float)(2 * i);
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (width + block.y - 1) / block.y);
    addMatrix<<<grid, block>>>(d_A, d_B, d_C, width);
    KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    for (int i = 0; i < width * width; i++) {
        float expected = h_A[i] + h_B[i];
        if (h_C[i] != expected) {
            printf("FAIL at %d: got %f expected %f\n", i, h_C[i], expected);
            return 1;
        }
    }
    printf("PASS: %dx%d matrix add correct\n", width, width);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}