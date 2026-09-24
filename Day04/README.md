# Day 4: Image Processing Kernels — Color to Grayscale

* **Read:** PMPP Ch. 3 continued — mapping threads to pixels
* **Challenge exercise:** RGB → grayscale kernel on a loaded image (stb_image or similar)
* **Build:** Add a simple box-blur kernel on top of the grayscale output
* **Tool:** Visually diff CPU vs. GPU output for correctness
* **System Design:** Consistency models — strong vs. eventual vs. causal

## Files

* `grayscaleKernel.cu` — RGB→gray (`0.21r+0.72g+0.07b`), 64×48 synthetic test
* `blurKernel.cu` — 3×3 box blur (`BLURSIZE=1`), 64×48 synthetic test
* `gray_blur.cu` — chained pipeline: H2D(rgb) → gray → blur → D2H, dumps `gpu.pgm`/`cpu.pgm`
* `gpu.pgm` / `cpu.pgm` — visual diff outputs (git-ignored build artifacts)

## Notes

1 thread = 1 pixel: `col = bx*16+tx`, `row = by*16+ty`, guard `col<W && row<H`.

```cuda
dim3 block(16, 16);
dim3 grid((W + 15) / 16, (H + 15) / 16); // x sizes W, y sizes H
```

Grayscale: `gray = row*W+col`, `rgb = gray*3`, `out = 0.21r+0.72g+0.07b`.
Blur: 3×3 average with border skip, write once **after** loops.
Pipeline: no CPU roundtrip — `d_gray` stays on device between launches.

### Tool: visual diff

```bash
nvcc Day04/gray_blur.cu -o /tmp/gray_blur && /tmp/gray_blur
# PASS + Wrote Day04/gpu.pgm + Day04/cpu.pgm
xdg-open Day04/gpu.pgm
cmp Day04/gpu.pgm Day04/cpu.pgm && echo IDENTICAL
```

## Build / Run (from repo root)

```bash
nvcc Day04/grayscaleKernel.cu -o /tmp/gray4 && /tmp/gray4
nvcc Day04/blurKernel.cu -o /tmp/blur4 && /tmp/blur4
nvcc Day04/gray_blur.cu -o /tmp/gray_blur && /tmp/gray_blur
```
