# Day 3: Grids, Blocks & Threads — Indexing 2D Data

* **Read:** PMPP Ch. 3 — Multidimensional grids and data
* **Challenge exercise:** 2D matrix addition with a 2D grid/block layout
* **Build:** A generic `(row, col)` indexing helper you'll reuse in image kernels
* **Blog:** Skim "CUDA C++ Programming Guide" §Thread Hierarchy
* **System Design:** PACELC theorem

## Notes

### Fig 3.1 — 2×2 grid labeling (key point)

Grid = four blocks in a 2×2 array. Each block labeled `(blockIdx.y, blockIdx.x)` — e.g. Block `(1,0)` has `blockIdx.y=1`, `blockIdx.x=0`.

Ordering: labels put highest dimension first `(y,x)`, but code config puts lowest first:

```cuda
dim3 block(16, 16); // (x, y) — x first in code
dim3 grid(2, 2);    // (x, y)
kernel<<<grid, block>>>();
```

Reversed `(y,x)` labeling matches row-major access: `row=y`, `col=x`, `idx=row*W+col`.

### Generic (row, col) helper

```cuda
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
if (row < H && col < W)
  C[row * W + col] = A[row * W + col] + B[row * W + col];
```

### Fig 3.2 — 16×16 block, 4×5 grid = 20 blocks

Block = 16 threads in x, 16 in y. Need 4 blocks in y, 5 in x → `4×5=20` blocks. Heavy lines = block boundaries, shaded = threads covering pixels.

```cuda
dim3 block(16, 16); // (x, y)
dim3 grid(5, 4);    // 5 in x, 4 in y
kernel<<<grid, block>>>();
```

Per-thread pixel:

```cuda
row = blockIdx.y * blockDim.y + threadIdx.y; // vertical
col = blockIdx.x * blockDim.x + threadIdx.x; // horizontal
```

### Row-major vs column-major

2D view (3 rows × 4 cols) — same for both:

```text
        col 0   col 1   col 2   col 3
row 0 │ (0,0) │ (0,1) │ (0,2) │ (0,3) │
row 1 │ (1,0) │ (1,1) │ (1,2) │ (1,3) │
row 2 │ (2,0) │ (2,1) │ (2,2) │ (2,3) │
```

Row-major (C/CUDA): walk row by row. `idx = row*W + col`.

```text
memory ─▶ [ (0,0) (0,1) (0,2) (0,3) │ (1,0) (1,1) (1,2) (1,3) │ (2,0) (2,1) (2,2) (2,3) ]
idx    ─▶ [   0     1     2     3   │    4     5     6     7   │    8     9    10    11   ]
```

Column-major (Fortran/MATLAB): walk column by column. `idx = col*H + row`.

```text
memory ─▶ [ (0,0) (1,0) (2,0) │ (0,1) (1,1) (2,1) │ (0,2) (1,2) (2,2) │ (0,3) (1,3) (2,3) ]
idx    ─▶ [   0     1     2   │    3     4     5   │    6     7     8   │    9    10    11   ]
```

Why it matters — one warp, `row=1`, `threadIdx.x = 0..3` → `col = 0..3`:

```text
Row-major addrs:   4, 5, 6, 7   → contiguous ✅ coalesced (1 transaction)
Column-major addrs: 3, 5, 7, 9   → stride H=3 ❌ 4 transactions
```

Rule: in CUDA keep `x = col`, `y = row`, index `row*W + col`.

### Example 1: grayscale (`grayscale.cu`, 64×48)

```cuda
dim3 block(16, 16);
dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
colortoGrayscaleConvertion<<<grid, block>>>(d_Pout, d_Pin, width, height);
KERNEL_CHECK();
```

* `block(16,16)` = 256 threads/block, `x`→cols, `y`→rows.
* `grid` = ceil div: `(64+15)/16=4`, `(48+15)/16=3` → 12 blocks, 3072 threads = 3072 pixels.
* Launch: 1 thread/pixel, `gray=row*W+col`, `rgb=gray*3`, `out=0.21r+0.72g+0.07b`.
* `KERNEL_CHECK()` = `cudaGetLastError + cudaDeviceSynchronize`.

### Example 2: blur (`blurKernel.cu`, 64×48, BLURSIZE=1)

```cuda
dim3 block(16, 16);
dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
blurKernel<<<grid, block>>>(d_in, d_out, w, h);
KERNEL_CHECK();
```

* Same 4×3 grid. Each thread averages its 3×3 box: `pixVal/pixels`.
* Border threads skip OOB neighbours (fewer than 9 pixels).
* `out[row*w+col]` written once after both loops — not inside the loop.

### Quiz: `foo` (M=150, N=300, `bd(16,32)`)

`gd=((300-1)/16+1,(150-1)/32+1)=(19,5)` → threads/block `16*32=512`, blocks `19*5=95`, threads `304*160=48,640`, run line05 `150*300=45,000` (3,640 idle via guard).

Total threads = `(gridDim.x*blockDim.x) × (gridDim.y*blockDim.y)` = threads/block × blocks.

### Quiz: 2D index (W=400, H=500, row=20, col=10)

Row-major (rows of `W`): `row*W+col = 20*400+10 = 8010`. Column-major (cols of `H`): `col*H+row = 10*500+20 = 5020`.

### Quiz: 3D index (W=400, H=500, D=300, x=10, y=20, z=5)

Row-major, `x` fastest: `idx = z*H*W + y*W + x = 5*500*400+20*400+10 = 1,008,010`.
