# Day 6: Warp Divergence — Branches on the GPU

* **Read:** PMPP Ch. 4 continued — control divergence, predication
* **Challenge exercise:** Write a branchy kernel (`if (tid % 2)`) vs. a branchless equivalent
* **Build:** Benchmark both versions and explain the delta in your own notes
* **Tool:** `nsys profile` both kernels, compare instruction counts
* **System Design:** Reverse proxy & API Gateway basics

## Files

* `branchy_kernel.cu` — `branchy_kernel` (even `+1` / odd `×2`) vs `branchless_kernel` (arithmetic mask), N=1M verify + `cudaEvent` timing

## Notes

Both compute the same output: even `a+1`, odd `a*2`. Branchless uses `m = tid&1`: `out = a*(1+m) + (1-m)` — no `if`.

```bash
nvcc Day06/branchy_kernel.cu -o /tmp/branchy && /tmp/branchy
# PASS: N=1048576 branchy=0.042ms branchless=0.048ms speedup=0.90x (×3 runs)
```

Delta: ~none — the compiler predicated this trivial branch (both paths = 1 ALU op), so warps never serialize. First run without warmup showed a fake 922x (50ms context init vs 0.055ms); warmup launches added so events compare kernels, not startup. Real divergence cost needs heavier per-path work (Day 6 trap: `if/else` ≠ automatic 2x).

Next (Tool): `nsys profile --stats=true /tmp/branchy` and compare instruction counts per kernel.

## Build / Run (from repo root)

```bash
nvcc Day06/branchy_kernel.cu -o /tmp/branchy && /tmp/branchy
```
