#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include "../common/cuda_check.h"

__global__
void saxpy(float a, float *X, float *Y, int N){
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < N){
        Y[id] = a * X[id] + Y[id];
    }
}

int main() {
    float a = 2.5f;
    int Ns[2] = {1 << 20, 1000003};
    for (int t = 0; t < 2; t++) {
        int N = Ns[t];
        int bytes = N * sizeof(float);
        float *h_X = (float*)malloc(bytes);
        float *h_Y = (float*)malloc(bytes);
        float *h_Y0 = (float*)malloc(bytes);
        for (int i = 0; i < N; i++) {
            h_X[i] = (float)(i % 100) * 0.5f;
            h_Y[i] = (float)(i % 77) * 0.25f;
            h_Y0[i] = h_Y[i];
        }

        float *d_X, *d_Y;
        CUDA_CHECK(cudaMalloc((void**)&d_X, bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_Y, bytes));
        CUDA_CHECK(cudaMemcpy(d_X, h_X, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Y, h_Y, bytes, cudaMemcpyHostToDevice));

        dim3 block(256);
        dim3 grid((N + block.x - 1) / block.x);
        saxpy<<<grid, block>>>(a, d_X, d_Y, N);
        KERNEL_CHECK();

        CUDA_CHECK(cudaMemcpy(h_Y, d_Y, bytes, cudaMemcpyDeviceToHost));

        for (int i = 0; i < N; i++) {
            float expected = a * h_X[i] + h_Y0[i];
            if (h_Y[i] != expected) {
                printf("FAIL N=%d at %d: got %f expected %f\n", N, i, h_Y[i], expected);
                return 1;
            }
        }
        printf("PASS: N=%d saxpy correct\n", N);

        CUDA_CHECK(cudaFree(d_X));
        CUDA_CHECK(cudaFree(d_Y));
        free(h_X);
        free(h_Y);
        free(h_Y0);
    }
    return 0;
}