#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include "../common/cuda_check.h"

__global__
void printWarpSize(int N){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N){
        int warpId = threadIdx.x / warpSize;
        int lane = threadIdx.x % warpSize;
        printf("tid=%d block=%d tx=%d warp=%d lane=%d warpSize=%d\\n", tid, blockIdx.x, threadIdx.x, warpId, lane, warpSize);
    }
}

int main(){
    int N = 100;
    printWarpSize<<<1, 32>>>(N);
    KERNEL_CHECK();

    return 0;
}