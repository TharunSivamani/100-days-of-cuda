# Day 2: The CUDA Execution Model

* **Read:** PMPP Ch. 2 — Data parallel computing, kernel launch syntax
* **Challenge exercise:** Vector addition (host alloc → device copy → kernel → copy back)
* **Build:** Vector add with a reusable `CUDA_CHECK` error macro you'll use for the rest of the 100 days
* **Tool:** Time it with `nvprof`/`nsys` once just to see the trace
* **System Design:** CAP theorem

## Files

* `vec_add.cu` — `vecAddKernel` + `vecAdd()` wrapper + `main()` (N=8000 verify)
* `../common/cuda_check.h` — `CUDA_CHECK()` / `KERNEL_CHECK()` shared header

## Notes

`kernel<<<grid, block>>>` where grid = blocks, block = threads/block.

```cuda
// N=8000, block=256 -> grid=(8000+255)/256=32 (32*256=8192 threads, 192 idle)
// if (i < N) guard skips the idle threads
vecAddKernel<<<(n+255)/256, 256>>>(A_d, B_d, C_d, n);
```

```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x; // global tid
if (i < N) C[i] = A[i] + B[i];                  // 1 thread = 1 element
```

Host vs device:

```text
h_A via malloc/free (CPU RAM) vs d_A via cudaMalloc/cudaFree (GPU VRAM)
cudaMemcpy(d_A,h_A,bytes,HostToDevice) -> kernel -> cudaMemcpy(h_C,d_C,bytes,DeviceToHost)
```

Error macro (`../common/cuda_check.h`):

```cuda
#include "../common/cuda_check.h"
CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
vecAddKernel<<<grid,block>>>(...);
KERNEL_CHECK(); // cudaGetLastError + cudaDeviceSynchronize
```

### Tool: nsys trace (RTX 3060 Laptop)

`nvprof` is deprecated on CC 8.0+, use `nsys`:

```bash
nvcc Day02/vec_add.cu -o /tmp/vec_add
nsys profile -o /tmp/vec_add_report --stats=true --force-overwrite=true /tmp/vec_add
```

Result for N=8000 (0.032 MB per array):

* `cudaMalloc x3: ~111ms (99.5%)` — dominates
* `vecAddKernel x1: ~1.8us`
* `memcpy H2D x2: ~9us, D2H x1: ~6us`

Trace order: `H2D -> Kernel -> D2H`.

## Build / Run (from repo root)

```bash
nvcc Day02/vec_add.cu -o /tmp/vec_add && /tmp/vec_add
# PASS: all 8000 elements correct
```
