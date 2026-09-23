#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include "../common/cuda_check.h"

__global__
void colortoGrayscaleConvertion(unsigned char *Pout, unsigned char *Pin, int width, int height) {
    // 1. Global pixel coords: block start + local id; x -> col, y -> row
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // 2. Guard: grid is ceil'd, edge threads may be outside image
    if (col < width && row < height) {
        // 3. Row-major offsets: 1 gray byte/px, 3 RGB bytes/px
        int grayOffset = row * width + col;
        int rgbOffset = grayOffset * 3;
        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];
        // 4. Luminance: green heaviest, blue lightest
        Pout[grayOffset] = (unsigned char)(0.21f * r + 0.72f * g + 0.07f * b);
    }
}

int main() {
    // 5. Host alloc + synthetic RGB image
    int width = 64, height = 48;
    int rgbSize = width * height * 3 * sizeof(unsigned char);
    int graySize = width * height * sizeof(unsigned char);

    unsigned char *h_Pin = (unsigned char*)malloc(rgbSize);
    unsigned char *h_Pout = (unsigned char*)malloc(graySize);
    for (int i = 0; i < width * height; i++) {
        h_Pin[i * 3] = (unsigned char)(i % 256);
        h_Pin[i * 3 + 1] = (unsigned char)((2 * i) % 256);
        h_Pin[i * 3 + 2] = (unsigned char)((3 * i) % 256);
    }

    // 6. Device alloc + H2D copy
    unsigned char *d_Pin, *d_Pout;
    CUDA_CHECK(cudaMalloc((void**)&d_Pin, rgbSize));
    CUDA_CHECK(cudaMalloc((void**)&d_Pout, graySize));
    CUDA_CHECK(cudaMemcpy(d_Pin, h_Pin, rgbSize, cudaMemcpyHostToDevice));

    // 7. 2D launch: 16x16 blocks cover WxH, ceil division
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    colortoGrayscaleConvertion<<<grid, block>>>(d_Pout, d_Pin, width, height);
    KERNEL_CHECK();

    // 8. D2H copy + CPU reference verify
    CUDA_CHECK(cudaMemcpy(h_Pout, d_Pout, graySize, cudaMemcpyDeviceToHost));

    for (int i = 0; i < width * height; i++) {
        unsigned char expected = (unsigned char)(0.21f * h_Pin[i * 3] + 0.72f * h_Pin[i * 3 + 1] + 0.07f * h_Pin[i * 3 + 2]);
        if (h_Pout[i] != expected) {
            printf("FAIL at %d: got %u expected %u\n", i, h_Pout[i], expected);
            return 1;
        }
    }
    printf("PASS: %dx%d grayscale correct\n", width, height);

    // 9. Cleanup
    CUDA_CHECK(cudaFree(d_Pin));
    CUDA_CHECK(cudaFree(d_Pout));
    free(h_Pin);
    free(h_Pout);
    return 0;
}
