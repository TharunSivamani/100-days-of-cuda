#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include "../common/cuda_check.h"

__global__
void vecAddKernel(float *A, float *B, float *C, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // register-heavy: 32 DISTINCT values, all live at the final sum
    // (a linear chain would reuse ~2 regs — width, not depth, costs regs)
    if (i < N){
        float a = A[i], b = B[i];
        float t0 = a * 0.5f + b * 1.0f;
        float t1 = a * 1.5f - b * 1.0f;
        float t2 = a * 2.5f + b * 2.0f;
        float t3 = a * 3.5f - b * 2.0f;
        float t4 = a * 4.5f + b * 3.0f;
        float t5 = a * 5.5f - b * 3.0f;
        float t6 = a * 6.5f + b * 4.0f;
        float t7 = a * 7.5f - b * 4.0f;
        float t8 = a * 8.5f + b * 5.0f;
        float t9 = a * 9.5f - b * 5.0f;
        float t10 = a * 10.5f + b * 6.0f;
        float t11 = a * 11.5f - b * 6.0f;
        float t12 = a * 12.5f + b * 7.0f;
        float t13 = a * 13.5f - b * 7.0f;
        float t14 = a * 14.5f + b * 8.0f;
        float t15 = a * 15.5f - b * 8.0f;
        float t16 = a * 16.5f + b * 9.0f;
        float t17 = a * 17.5f - b * 9.0f;
        float t18 = a * 18.5f + b * 10.0f;
        float t19 = a * 19.5f - b * 10.0f;
        float t20 = a * 20.5f + b * 11.0f;
        float t21 = a * 21.5f - b * 11.0f;
        float t22 = a * 22.5f + b * 12.0f;
        float t23 = a * 23.5f - b * 12.0f;
        float t24 = a * 24.5f + b * 13.0f;
        float t25 = a * 25.5f - b * 13.0f;
        float t26 = a * 26.5f + b * 14.0f;
        float t27 = a * 27.5f - b * 14.0f;
        float t28 = a * 28.5f + b * 15.0f;
        float t29 = a * 29.5f - b * 15.0f;
        float t30 = a * 30.5f + b * 16.0f;
        float t31 = a * 31.5f - b * 16.0f;
        C[i] = (((t0+t1)+(t2+t3)) + ((t4+t5)+(t6+t7)))
             + (((t8+t9)+(t10+t11)) + ((t12+t13)+(t14+t15)))
             + (((t16+t17)+(t18+t19)) + ((t20+t21)+(t22+t23)))
             + (((t24+t25)+(t26+t27)) + ((t28+t29)+(t30+t31)));
    }

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
    // STEP 1: problem size + host alloc (h_A, h_B, h_C)
    int n = 1 << 20; // large enough for stable event timing
    int size = n * sizeof(float);

    float *A = (float*)malloc(size);
    float *B = (float*)malloc(size);
    float *C = (float*)malloc(size);

    // STEP 2: init host inputs (small values keep float rounding < tolerance)
    for (int i = 0; i < n; i++) {
        A[i] = (float)(i % 13) * 0.5f;
        B[i] = (float)(i % 7) * 0.5f;
    }

    // STEP 3: run device work via vecAdd() wrapper (mallocs d_*, H2D, launches, D2H)
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    vecAdd(A, B, C, n); // warmup: pays context-init once, outside timing
    CUDA_CHECK(cudaEventRecord(t0));
    vecAdd(A, B, C, n);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    printf("TIME: %.3f ms\n", ms);
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));

    // STEP 4: CPU verify loop — replay the 32-term sum (tolerance: GPU FMA
    // fuses mul+add in one rounding, CPU does two — expect 1-ulp diffs)
    for (int i = 0; i < n; i++) {
        float a = A[i], b = B[i];
        float t[32];
        for (int k = 0; k < 32; k++) {
            float m = (float)k + 0.5f, bm = (float)(k / 2 + 1);
            t[k] = (k % 2 == 0) ? a * m + b * bm : a * m - b * bm;
        }
        float expected = (((t[0]+t[1])+(t[2]+t[3])) + ((t[4]+t[5])+(t[6]+t[7])))
                       + (((t[8]+t[9])+(t[10]+t[11])) + ((t[12]+t[13])+(t[14]+t[15])))
                       + (((t[16]+t[17])+(t[18]+t[19])) + ((t[20]+t[21])+(t[22]+t[23])))
                       + (((t[24]+t[25])+(t[26]+t[27])) + ((t[28]+t[29])+(t[30]+t[31])));
        if (fabsf(C[i] - expected) > 1e-2f) {
            printf("FAIL at %d: got %f, expected %f\n", i, C[i], expected);
            return 1;
        }
    }
    printf("PASS: all %d elements correct\n", n);

    // STEP 5: cleanup host allocations
    free(A);
    free(B);
    free(C);
    return 0;
}