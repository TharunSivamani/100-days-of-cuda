#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include "../common/cuda_check.h"

__global__
void branchy_kernel(float *in, float *out, int N){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N){
        if (tid % 2 == 0){
            out[tid] = in[tid] + 1.0f;
        }else{
            out[tid] = in[tid] * 2.0f;
        }
    }
}

__global__
void branchless_kernel(float *in, float *out, int N){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N){
        float m = (float)(tid & 1); // 0 even, 1 odd — no branch
        out[tid] = in[tid] * (1.0f + m) + (1.0f - m);
    }
}

int main() {
    int N = 1 << 20;
    int bytes = N * sizeof(float);
    float *h_in = (float*)malloc(bytes);
    float *h_branchy = (float*)malloc(bytes);
    float *h_branchless = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_in[i] = (float)(i % 100) * 0.5f;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc((void**)&d_in, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    // warmup (context init) so timing compares kernels, not startup
    branchy_kernel<<<grid, block>>>(d_in, d_out, N);
    branchless_kernel<<<grid, block>>>(d_in, d_out, N);
    KERNEL_CHECK();

    CUDA_CHECK(cudaEventRecord(t0));
    branchy_kernel<<<grid, block>>>(d_in, d_out, N);
    KERNEL_CHECK();
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms_branchy = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_branchy, t0, t1));
    CUDA_CHECK(cudaMemcpy(h_branchy, d_out, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(t0));
    branchless_kernel<<<grid, block>>>(d_in, d_out, N);
    KERNEL_CHECK();
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms_branchless = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_branchless, t0, t1));
    CUDA_CHECK(cudaMemcpy(h_branchless, d_out, bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++) {
        if (h_branchy[i] != h_branchless[i]) {
            printf("FAIL at %d: %f vs %f\n", i, h_branchy[i], h_branchless[i]);
            return 1;
        }
    }
    printf("PASS: N=%d branchy=%.3fms branchless=%.3fms speedup=%.2fx\n",
           N, ms_branchy, ms_branchless, ms_branchy / ms_branchless);

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_branchy);
    free(h_branchless);
    return 0;
}