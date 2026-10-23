# Day 10: Naive Matrix Multiplication — the 60-day baseline

* **Read:** PMPP Ch. 5 continued — why naive matmul is memory-bound
* **Challenge exercise:** Naive matmul (global memory only)
* **Build:** Benchmark against a CPU reference (correctness + GFLOPS)
* **Tool:** Note the achieved GFLOPS — the number to beat for the next 60 days
* **System Design:** DNS & how requests route globally

## Files

* `naive_matmul.cu` — global-memory-only `matmul` kernel (one thread = one `C[row][col]`, CLB style), cudaEvent timing (warmup launch → timed launch), CPU reference verify + timed CPU GFLOPS, all at `width = 1024`, `block(16,16)`

## Notes

### Why naive is memory-bound (the item-1 math)

* Per thread: `N` muls + `N` adds but `2N` global loads → **1 FMA per 2 DRAM loads**.
* Total at N=1024: `2·N³ ≈ 2.15 GFLOP`, but `2·N³ loads × 4B ≈ 8.6 GB` of traffic for just 8 MB of unique data — every A/B element refetched instead of reused.
* Arithmetic intensity = `2.15e9 ÷ 8.6e9 ≈ 0.25 FLOP/byte`; the GPU needs ~25+ to saturate compute → **~100× memory-starved**. Pure-DRAM floor: 8.6 GB ÷ 360 GB/s ≈ 24 ms ⇒ ≤ ~90 GFLOPS.
* Second hit: per k-step a warp's B reads (`N[k*width+col]`, cols 0..31) are **stride-1024** → 32 separate transactions per warp instead of 1. A-row reads are same-address broadcasts (fine).

### Baseline result (the tool item)

| source | time | GFLOPS |
|---|---|---|
| **GPU naive** (warmup excluded, best of 4) | 3.24 ms | **~662** |
| GPU spread over runs | 3.22–3.86 ms | 556–661 |
| CPU reference (1 thread) | 4.0 s | ~0.5 |

**Day 10 baseline to beat: ~662 GFLOPS ≈ 6.6% of the ~10 TFLOPS peak.** Above the 90 GFLOPS DRAM floor because L1/L2 catch part of the reuse (12 MB data vs 3 MB L2), still far under peak — that gap is tiling (Day 11) onward. CPU is only 0.5 GFLOPS because the same column-stride walk thrashes its caches too — naive is memory-bound on both sides of the ISA.

Correctness: relative tolerance `fabsf(got-expected) > 1e-3f * expected` — sums reach ~5e3, one ulp there ≈ 1e-3, so an absolute check flakes (FMA 1-ulp lesson). Init `%13`/`%7` keeps magnitudes sane.

### Gotchas hit today

* `width` vs element count: `width = 1024` gives `1024² = 1<<20` elements — buffers are `width*width`, not `1<<20`.
* GFLOPS units: `2·N³ / (ms × 1e6)` — FLOPs ÷ (ms × 1e⁻³ s × 1e⁹); `clock()` → seconds needs `CLOCKS_PER_SEC`, GFLOPS is a *separate* line of arithmetic.

### System Design (15 min)

DNS request path: browser/OS cache → recursive resolver (ISP or 8.8.8.8) → root → TLD (`.com`) → authoritative NS → A/AAAA record, cached per TTL. Global routing: **anycast** (one IP, many sites, nearest wins — root/TLDs use it), round-robin and geo-DNS for load spreading; low TTL = fast failover, high TTL = fewer lookups.

## Build / Run (from repo root)

```bash
nvcc Day10/naive_matmul.cu -arch=native -o /tmp/matmul10 && /tmp/matmul10
```

Expected: `TIME` ~3.2–3.9 ms, `GFLOPS: ~560–660`, `CPU: ~4 s -> ~0.5 GFLOPS`, PASS.
