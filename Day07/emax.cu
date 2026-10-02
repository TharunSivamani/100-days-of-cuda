#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include "../common/cuda_check.h"

__global__
void emax(float *A, float *B, float *C, int N){
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < N){
        C[id] = fmaxf(A[id], B[id]);
    }
}

int main() {
    int Ns[2] = {1 << 20, 1000003};
    for (int t = 0; t < 2; t++) {
        int N = Ns[t];
        int bytes = N * sizeof(float);
        float *h_A = (float*)malloc(bytes);
        float *h_B = (float*)malloc(bytes);
        float *h_C = (float*)malloc(bytes);
        for (int i = 0; i < N; i++) {
            h_A[i] = (float)((i % 200) - 100) * 0.5f; // mixed signs
            h_B[i] = (float)(((3 * i) % 200) - 100) * 0.5f;
        }

        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));
        CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

        dim3 block(256);
        dim3 grid((N + block.x - 1) / block.x);
        emax<<<grid, block>>>(d_A, d_B, d_C, N);
        KERNEL_CHECK();

        CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

        for (int i = 0; i < N; i++) {
            float expected = fmaxf(h_A[i], h_B[i]);
            if (h_C[i] != expected) {
                printf("FAIL N=%d at %d: got %f expected %f\n", N, i, h_C[i], expected);
                return 1;
            }
        }
        printf("PASS: N=%d emax correct\n", N);

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        free(h_A);
        free(h_B);
        free(h_C);
    }
    return 0;
}