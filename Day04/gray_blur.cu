#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include "../common/cuda_check.h"

#define BLURSIZE 1

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
    // TODO 7: host alloc — W=64, H=48; h_rgb (W*H*3), h_blur (W*H); fill h_rgb
    int W = 64, H = 48;
    int rgbSize = W * H * 3 * sizeof(unsigned char);
    int graySize = W * H * sizeof(unsigned char);
    unsigned char *h_rgb = (unsigned char*)malloc(rgbSize);
    unsigned char *h_blur = (unsigned char*)malloc(graySize);
    for (int i = 0; i < W * H; i++) {
        h_rgb[i * 3] = (unsigned char)(i % 256);
        h_rgb[i * 3 + 1] = (unsigned char)((2 * i) % 256);
        h_rgb[i * 3 + 2] = (unsigned char)((3 * i) % 256);
    }

    // TODO 8: device alloc d_rgb/d_gray/d_blur; H2D copy h_rgb -> d_rgb
    unsigned char *d_rgb, *d_gray, *d_blur;
    CUDA_CHECK(cudaMalloc((void**)&d_rgb, rgbSize));
    CUDA_CHECK(cudaMalloc((void**)&d_gray, graySize));
    CUDA_CHECK(cudaMalloc((void**)&d_blur, graySize));
    CUDA_CHECK(cudaMemcpy(d_rgb, h_rgb, rgbSize, cudaMemcpyHostToDevice));

    // TODO 9: dim3 block(16,16), grid ceil; launch grayKernel -> KERNEL_CHECK()
    //         launch blurOnGray(d_blur, d_gray) -> KERNEL_CHECK()  (no CPU roundtrip!)
    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
    grayscaleKernel<<<grid, block>>>(d_gray, d_rgb, W, H);
    KERNEL_CHECK();
    blurKernel<<<grid, block>>>(d_gray, d_blur, W, H);
    KERNEL_CHECK();

    // TODO 10: D2H d_blur -> h_blur; CPU gray+blur reference; PASS/FAIL; free
    CUDA_CHECK(cudaMemcpy(h_blur, d_blur, graySize, cudaMemcpyDeviceToHost));

    // CPU reference: gray then 3x3 blur
    unsigned char *h_gray_ref = (unsigned char*)malloc(graySize);
    unsigned char *h_cpu_blur = (unsigned char*)malloc(graySize);
    for (int i = 0; i < W * H; i++)
        h_gray_ref[i] = (unsigned char)(0.21f * h_rgb[i * 3] + 0.72f * h_rgb[i * 3 + 1] + 0.07f * h_rgb[i * 3 + 2]);
    for (int row = 0; row < H; row++) {
        for (int col = 0; col < W; col++) {
            int sum = 0, cnt = 0;
            for (int br = -BLURSIZE; br <= BLURSIZE; br++) {
                for (int bc = -BLURSIZE; bc <= BLURSIZE; bc++) {
                    int r = row + br, c = col + bc;
                    if (r >= 0 && r < H && c >= 0 && c < W) {
                        sum += h_gray_ref[r * W + c];
                        cnt++;
                    }
                }
            }
            unsigned char expected = (unsigned char)((float)sum / cnt);
            h_cpu_blur[row * W + col] = expected;
            if (h_blur[row * W + col] != expected) {
                printf("FAIL at (%d,%d): got %u expected %u\n", row, col, h_blur[row * W + col], expected);
                return 1;
            }
        }
    }
    printf("PASS: %dx%d gray+blur pipeline correct\n", W, H);

    // Visual diff: dump GPU vs CPU as PGM in same folder
    FILE *fgpu = fopen("Day04/gpu.pgm", "wb");
    FILE *fcpu = fopen("Day04/cpu.pgm", "wb");
    if (fgpu && fcpu) {
        fprintf(fgpu, "P5\n%d %d\n255\n", W, H);
        fprintf(fcpu, "P5\n%d %d\n255\n", W, H);
        fwrite(h_blur, 1, graySize, fgpu);
        fwrite(h_cpu_blur, 1, graySize, fcpu);
        fclose(fgpu);
        fclose(fcpu);
        printf("Wrote Day04/gpu.pgm + Day04/cpu.pgm — open side-by-side to compare\n");
    }

    CUDA_CHECK(cudaFree(d_rgb));
    CUDA_CHECK(cudaFree(d_gray));
    CUDA_CHECK(cudaFree(d_blur));
    free(h_rgb);
    free(h_blur);
    free(h_gray_ref);
    free(h_cpu_blur);
    return 0;
}
