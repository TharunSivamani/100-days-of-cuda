#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include "../common/cuda_check.h"

__global__
void convLite(int *din, int *dout, int N){
    __shared__ int s[256];

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < N) s[threadIdx.x] = din[idx];
    __syncthreads();
    if (idx < N) dout[idx] = s[threadIdx.x];
}

int main(){
    int N = 1<<20;
    int size = N * sizeof(int);
    int *h_in = (int*)malloc(size);
    int *h_out = (int*)malloc(size);
    for (int i = 0; i < N; i++) h_in[i] = i;

    int *d_in, *d_out;
    CUDA_CHECK(cudaMalloc((void**)&d_in, size));
    CUDA_CHECK(cudaMalloc((void**)&d_out, size));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    convLite<<<grid, block>>>(d_in, d_out, N);
    KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++) {
        if (h_out[i] != h_in[i]) {
            printf("FAIL at %d: got %d expected %d\n", i, h_out[i], h_in[i]);
            return 1;
        }
    }
    printf("PASS: N=%d conv-lite copy correct\n", N);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);
    return 0;
}