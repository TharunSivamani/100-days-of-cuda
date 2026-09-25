#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include "../common/cuda_check.h"

__global__
void warpGroup(int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        int warp = tid / warpSize;
        int lane = tid % warpSize;
        printf("tid=%d block=%d warp=%d lane=%d\n", tid, blockIdx.x, warp, lane);
    }
}

int main() {
    int N = 100;
    int block = 32;
    int grid = (N + block - 1) / block;
    warpGroup<<<grid, block>>>(N);
    KERNEL_CHECK();
    return 0;
}
