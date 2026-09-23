#pragma once
// Syntax we use in this file:
//   __host__ __device__ inline <ret> <name>(<params>) { return ...; }
// dual-mark = callable from CPU + GPU; device-only (tcol/trow) uses __device__ alone.
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__host__ __device__ inline int rc_to_idx_rm(int row, int col, int W){
    return row*W + col;
}

__host__ __device__ inline int idx_to_row_rm(int idx, int W){
    return idx/W;
}

__host__ __device__ inline int idx_to_col_rm(int idx, int W){
    return idx % W;
}

__host__ __device__ inline int rc_to_idx_cm(int row, int col, int H){
    return col*H + row;
}

__host__ __device__ inline int idx_to_row_cm(int idx, int H){
    return idx%H;
}

__host__ __device__ inline int idx_to_col_cm(int idx, int H){
    return idx/H;
}

__device__ inline int tcol(){
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ inline int trow(){
    return blockIdx.y * blockDim.y + threadIdx.y;
}

__host__ __device__ inline bool inside(int row, int col, int H, int W){
    return ((row < H) && (col < W));
}

__host__ __device__ inline int xyz_to_idx_rm(int x, int y, int z, int W, int H){
    return ((z*H*W) + (y*W) + x);
}

__host__ __device__ inline void idx_to_xyz_rm(int idx, int W, int H, int *x, int *y, int *z){
    *x = idx % W;
    *y = (idx / W) % H;
    *z = idx / (W * H);
}

__host__ __device__ inline int xyz_to_idx_cm(int x, int y, int z, int H, int D){
    return x*H*D + y*D + z;
}