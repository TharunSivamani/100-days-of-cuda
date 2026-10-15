#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include "../common/cuda_check.h"

__global__
void reverseArray(int *d, int N){
    __shared__ int s[64];
    int idx = threadIdx.x;
    int revIdx = N - idx - 1;
    s[idx] = d[idx];
    __syncthreads();
    d[idx] = s[revIdx];
}

__global__
void reverseTiled(int *dout, int *din, int N){
    __shared__ int s[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) s[threadIdx.x] = din[idx];
    __syncthreads();
    if (idx < N) dout[N - 1 - idx] = s[threadIdx.x];
}

int main() {
    int N = 1 << 20;
    int bytes = N * sizeof(int);
    int *h_in = (int*)malloc(bytes);
    int *h_out = (int*)malloc(bytes);
    for (int i = 0; i < N; i++) h_in[i] = i;

    int *d_in, *d_out;
    CUDA_CHECK(cudaMalloc((void**)&d_in, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    reverseTiled<<<grid, block>>>(d_out, d_in, N);
    KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++) {
        if (h_out[i] != N - 1 - i) {
            printf("FAIL at %d: got %d expected %d\n", i, h_out[i], N - 1 - i);
            return 1;
        }
    }
    printf("PASS: N=%d tiled reverse correct\n", N);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);
    return 0;
}