# Day 1: Why GPUs? The Parallel Thinking Shift

* **Read:** PMPP Ch. 1 — Heterogeneous parallel computing
* **Challenge exercise:** Hello CUDA world, device query (`cudaGetDeviceProperties`)
* **Build:** A kernel that prints thread/block IDs; query and print your GPU's SM count, warp size, max threads/block
* **Blog:** NVIDIA Devblog — "An Even Easier Introduction to CUDA"
* **System Design:** Scalability basics — vertical vs. horizontal scaling

## Notes

`threadIdx.x` contains the index of the current thread within its block, and `blockDim.x` contains the number of threads in the block.

* `threadIdx` = "Which thread am I inside my block?"
* `blockDim` = "How big is my block?"

```cuda
kernel<<<3, 4>>>
// where, 3 is the number of blocks
// and 4 is the threads per block
```

`blockDim.x` is 4.

`blockDim` is NOT the current thread. It describes the size of the block.

`blockIdx.x` answers: "Which block am I?"

```cuda
kernel<<<3, 4>>>();

// Block 0 → blockIdx.x = 0
// Block 1 → blockIdx.x = 1
// Block 2 → blockIdx.x = 2
```

```text
What am I asking?

threadIdx.x  →   Which thread am I?
blockIdx.x   →   Which block am I?
blockDim.x   →   How many threads are in my block?
```

### Mental Model

```text
                    BLOCK
        ┌──────────────────────────┐
        │                          │
        │  T0  T1  T2  T3  T4      │
        │                          │
        └──────────────────────────┘
             ↑
        threadIdx.x
```

Think of a block as a bus.

* `threadIdx` tells you: My seat number.
* `blockDim` tells you: How many seats are on this bus.
* `blockIdx` tells you: Which bus am I on?

```text
global position = (bus number × seats per bus) + (my seat number)
```

`gridDim.x`

```cuda
kernel<<<3, 4>>>();
```

```text
                    GRID
              gridDim.x = 3
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
    BLOCK 0      BLOCK 1      BLOCK 2
    blockIdx=0   blockIdx=1   blockIdx=2
       │            │            │
     4 threads    4 threads    4 threads
       │            │            │
   T0 T1 T2 T3  T0 T1 T2 T3  T0 T1 T2 T3
```

```cuda
threadIdx.x   // my thread number inside block
blockDim.x    // threads per block
blockIdx.x    // my block number
gridDim.x     // blocks in grid

globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
```

Grid-stride version:

```cuda
int stride = blockDim.x * gridDim.x;

for (int i = globalIdx; i < N; i += stride) {
    // work
}
```
