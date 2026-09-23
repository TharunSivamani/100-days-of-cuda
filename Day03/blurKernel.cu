#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include "../common/cuda_check.h"

#define BLURSIZE 1

__global__
void blurKernel(
    unsigned char *in, unsigned char *out, int w, int h
){
    // 1. Global pixel coords: block start (blockIdx*blockDim) + local id (threadIdx)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // x -> column (contiguous, coalesced)
    int row = blockIdx.y * blockDim.y + threadIdx.y; // y -> row

    // 2. Guard: grid is ceil(w/16)xceil(h/16), edge threads may fall outside
    if (col < w && row < h){
        int pixVal = 0; // sum of valid neighbours
        int pixels = 0; // count of valid neighbours (edges have fewer)

        // 3. Walk (2*BLURSIZE+1)x(2*BLURSIZE+1) box around (row,col); BLURSIZE=1 -> 3x3
        for (int blurRow = -BLURSIZE; blurRow < BLURSIZE + 1; ++blurRow){
            for (int blurCol = -BLURSIZE; blurCol < BLURSIZE + 1; ++blurCol){
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                // 4. Clamp at image borders: skip out-of-bounds neighbours
                if(curRow >= 0 && curRow < h && curCol >= 0 && curCol < w){
                    pixVal += in[curRow * w + curCol]; // row-major load
                    ++pixels;
                }
            }
        }

        // 5. Write average once, after both loops (not inside loop)
        out[row*w + col] = (unsigned char)((float)pixVal / pixels);
    }
}

int main() {
    int w = 64, h = 48;
    int size = w * h * sizeof(unsigned char);

    unsigned char *h_in = (unsigned char*)malloc(size);
    unsigned char *h_out = (unsigned char*)malloc(size);
    for (int i = 0; i < w * h; i++) h_in[i] = (unsigned char)(i % 256);

    unsigned char *d_in, *d_out;
    CUDA_CHECK(cudaMalloc((void**)&d_in, size));
    CUDA_CHECK(cudaMalloc((void**)&d_out, size));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    blurKernel<<<grid, block>>>(d_in, d_out, w, h);
    KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));

    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            int pixVal = 0, pixels = 0;
            for (int br = -BLURSIZE; br <= BLURSIZE; br++) {
                for (int bc = -BLURSIZE; bc <= BLURSIZE; bc++) {
                    int r = row + br, c = col + bc;
                    if (r >= 0 && r < h && c >= 0 && c < w) {
                        pixVal += h_in[r * w + c];
                        pixels++;
                    }
                }
            }
            unsigned char expected = (unsigned char)((float)pixVal / pixels);
            if (h_out[row * w + col] != expected) {
                printf("FAIL at (%d,%d): got %u expected %u\n", row, col, h_out[row * w + col], expected);
                return 1;
            }
        }
    }
    printf("PASS: %dx%d blur correct\n", w, h);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);
    return 0;
}