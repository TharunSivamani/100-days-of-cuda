#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include "../common/cuda_check.h"

#define BLURSIZE 1

__global__
void blurKernel(unsigned char *in, unsigned char *out, int w, int h) {
    // TODO 1: compute col from blockIdx.x, blockDim.x, threadIdx.x
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO 2: compute row from blockIdx.y, blockDim.y, threadIdx.y
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // TODO 3: guard — only continue if (col < w && row < h)
    if(col < w && row < h){
        // TODO 4: zero pixVal and pixels accumulators
        int pixVal = 0;
        int pixels = 0;
        // TODO 5: double loop blurRow/blurCol in [-BLURSIZE, BLURSIZE]; curRow = row+blurRow, curCol = col+blurCol
        for(int blurRow=-BLURSIZE; blurRow<BLURSIZE+1;++blurRow){
            // TODO 6: border check — skip OOB; else pixVal += in[curRow*w+curCol], pixels++
            for(int blurCol=-BLURSIZE;blurCol<BLURSIZE+1;++blurCol){
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                if(curRow >= 0 && curRow < h && curCol >= 0 && curCol < w){
                    pixVal += in[curRow * w + curCol]; // row major
                    ++pixels;
                }
            }
        }

        // TODO 7: AFTER both loops: out[row*w+col] = pixVal / pixels (cast to unsigned char)
        out[row * w + col] = (unsigned char)((float)pixVal / pixels);
    }
}

int main() {
    // TODO 8: host alloc — w=64, h=48, size = w*h; fill h_in
    int w = 64, h = 48;
    int size = w * h * sizeof(unsigned char);
    unsigned char *h_in = (unsigned char*)malloc(size);
    unsigned char *h_out = (unsigned char*)malloc(size);
    for (int i = 0; i < w * h; i++) h_in[i] = (unsigned char)(i % 256);

    // TODO 9: cudaMalloc d_in/d_out + H2D copy
    unsigned char *d_in, *d_out;
    CUDA_CHECK(cudaMalloc((void**)&d_in, size));
    CUDA_CHECK(cudaMalloc((void**)&d_out, size));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    // TODO 10: dim3 block(16,16), grid ceil(w/16, h/16); launch + KERNEL_CHECK()
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    blurKernel<<<grid, block>>>(d_in, d_out, w, h);
    KERNEL_CHECK();

    // TODO 11: D2H copy, CPU-verify double loop, PASS/FAIL, cudaFree/free
    CUDA_CHECK(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));

    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            int sum = 0, cnt = 0;
            for (int br = -BLURSIZE; br <= BLURSIZE; br++) {
                for (int bc = -BLURSIZE; bc <= BLURSIZE; bc++) {
                    int r = row + br, c = col + bc;
                    if (r >= 0 && r < h && c >= 0 && c < w) {
                        sum += h_in[r * w + c];
                        cnt++;
                    }
                }
            }
            unsigned char expected = (unsigned char)((float)sum / cnt);
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
