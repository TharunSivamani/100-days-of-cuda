#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include "../common/cuda_check.h"

__global__
void grayscaleKernel(
    unsigned char *Pout, unsigned char *Pin, int width, int height
){
    // TODO 1: compute col from blockIdx.x, blockDim.x, threadIdx.x
    // int col = ...;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO 2: compute row from blockIdx.y, blockDim.y, threadIdx.y
    // int row = ...;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // TODO 3: guard — return early unless (col < width && row < height)
    if (col < width && row < height){

        // TODO 4: grayOffset = row * width + col; rgbOffset = grayOffset * 3;
        int grayOffset = row * width + col;
        int rgbOffset = grayOffset * 3;

        // TODO 5: load r = Pin[rgbOffset], g = Pin[rgbOffset+1], b = Pin[rgbOffset+2];
        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset+1];
        unsigned char b = Pin[rgbOffset+2];

        // TODO 6: Pout[grayOffset] = 0.21f*r + 0.72f*g + 0.07f*b;
        Pout[grayOffset] = (unsigned char)(0.21f*r + 0.72f*g + 0.07f*b);
    }
}

int main() {
    // TODO 7: host alloc — W=64, H=48, rgbSize = W*H*3, graySize = W*H; fill h_Pin
    int W = 64;
    int H = 48;
    int rgbSize = W * H * 3;
    int graySize = H * W;

    unsigned char *h_pin = (unsigned char*)malloc(rgbSize);
    unsigned char *h_pout = (unsigned char*)malloc(graySize);
    for (int i = 0; i < W * H; i++) {
        h_pin[i * 3] = (unsigned char)(i % 256);
        h_pin[i * 3 + 1] = (unsigned char)((2 * i) % 256);
        h_pin[i * 3 + 2] = (unsigned char)((3 * i) % 256);
    }

    // TODO 8: cudaMalloc d_Pin/d_Pout + H2D copy of h_Pin
    unsigned char *d_pin, *d_pout;
    CUDA_CHECK(cudaMalloc((void**)&d_pin, rgbSize));
    CUDA_CHECK(cudaMalloc((void**)&d_pout, graySize));
    CUDA_CHECK(cudaMemcpy(d_pin, h_pin, rgbSize, cudaMemcpyHostToDevice));

    // TODO 9: dim3 block(16,16), grid ceil(W/16, H/16); launch + KERNEL_CHECK()
    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
    grayscaleKernel<<<grid, block>>>(d_pout, d_pin, W, H);
    KERNEL_CHECK();

    // TODO 10: D2H copy, CPU-verify loop, printf PASS/FAIL, cudaFree/free
    CUDA_CHECK(cudaMemcpy(h_pout, d_pout, graySize, cudaMemcpyDeviceToHost));

    for (int i = 0; i < W * H; i++) {
        unsigned char expected = (unsigned char)(0.21f * h_pin[i * 3] + 0.72f * h_pin[i * 3 + 1] + 0.07f * h_pin[i * 3 + 2]);
        if (h_pout[i] != expected) {
            printf("FAIL at %d: got %u expected %u\n", i, h_pout[i], expected);
            return 1;
        }
    }
    printf("PASS: %dx%d grayscale correct\n", W, H);

    CUDA_CHECK(cudaFree(d_pin));
    CUDA_CHECK(cudaFree(d_pout));
    free(h_pin);
    free(h_pout);
    return 0;
}