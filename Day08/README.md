# Day 8: The Memory Hierarchy — Registers to Global Memory

* **Read:** PMPP Ch. 5 — Memory architecture and data locality (intro)
* **Challenge exercise:** Measure register usage of a kernel via `nvcc --ptxas-options=-v`
* **Build:** Same kernel, two variants — one register-heavy, one spilling — compare runtimes
* **Blog:** NVIDIA Devblog — "Using Shared Memory in CUDA C/C++" (Mark Harris)
* **System Design:** Cache eviction policies (LRU, LFU, ARC)

## Files

* `vec_add.cu` — baseline (`C[i]=A[i]+B[i]`), N=1M + event timing
* `regs_heavy.cu` — 32 simultaneously-live floats summed in a tree, N=1M + event timing + tolerance verify

## Notes

### Challenge: `ptxas` report (baseline, `sm_86`)

```bash
nvcc Day08/vec_add.cu -o /tmp/d8base -arch=native --ptxas-options=-v
# Used 12 registers, 0 spill stores/loads, 0 barriers (sm_52 default said 8 — always quote sm_86)
```

12 regs/thread × 256 threads = 3K regs/block; each SM owns one 64K×32-bit file split across its 4 scheduler partitions.

### Build: reg counts (`-arch=native`, N=1M, warmup + 3 runs)

| Variant | Regs | Spill | Time |
|---|---|---|---|
| baseline | 12 | 0/0 | ~2.85ms |
| heavy (32 live) | 39 | 0/0 | ~2.9–3.2ms |
| heavy `-maxrregcount=24` | 22 | 0/0 | ~2.9ms |

Delta: ~none (noise) — and the road there had 3 lessons:
1. A linear `v0→v14` chain reuses ~2 regs (still 12 total) — **width** (simultaneously-live values), not depth, costs registers.
2. Exact `==` verify fails at 1 ulp (GPU FMA fuses mul+add, CPU doesn't) → `fabsf < 1e-2` with small inputs.
3. No true spill materialized: cap 24 → compiler **rematerialized** down to 22 regs instead of spilling; caps 16/8 get clamped to the kernel minimum (24). Why no slowdown anyway: 6 blocks × 256 threads fits full occupancy in both cases (18K vs 59.9K < 64K regs), and the kernel is memory-bound — 12MB traffic dominates, extra ALU hides underneath.

### Warps → blocks → SMs, bottom-up (the `dim3 block` connection)

Fixed by hardware: **1 warp = 32 threads**, **48 warp slots/SM** → ceiling `48 × 32 = 1536` threads/SM.
Your choice at launch: `dim3 block(256)` → **warps/block = 256 ÷ 32 = 8**.
Division: **blocks/SM = 48 ÷ 8 = 6** (also capped by 16 blocks/SM, 64K regs, shared mem — occupancy is the min of all four).

```text
32 threads = 1 warp (physics) ─▶ 8 warps = your 256-thread block (dim3) ─▶ 6 blocks = full SM (48 warps)
```

Reg check on the same numbers: baseline block bills `256×12 = 3K` desks, heavy `256×39 ≈ 10K`; 6 blocks = 18K / 59.9K < 64K desks — both 100% full, hence the tied timings.

### Blog report (Harris, "Using Shared Memory in CUDA C/C++")

**Problem:** strided global access (e.g. 2nd-dim of a 2D array) can't coalesce on any generation; misaligned access stopped mattering, stride never did. Fix: stage through shared memory so global stays unit-stride.
**What it is:** on-chip, ~100x lower latency than uncached global (bank-conflict-free), allocated per block — threads read data loaded by *other* threads in the block. Uses: user-managed cache, cooperative algorithms (reduction), de-striding global traffic.
**Sync:** `__syncthreads()` barrier after smem writes, before cross-thread reads (A-reads-B pattern across warps races otherwise); must be reached uniformly — divergent `__syncthreads()` = undefined/deadlock.
**Static vs dynamic:** `__shared__ int s[64]` (compile-time) vs `extern __shared__ int s[]` + 3rd launch arg `<<<g,b,bytes>>>`; multiple dynamic arrays = one extern block manually carved with pointers, single summed size at launch.
**Example (reverse):** `s[t]=d[t]; __syncthreads(); d[t]=s[tr]` (`tr=n-t-1`) — global touched only via linear `t` (coalesced everywhere), reversal happens in smem.
**Banks:** 32 banks, 32-bit words round-robin; same-bank different-address = serialized N-way; same-address = broadcast (cc2.0+: multicast). Fix double-precision stride with 8B bank mode.
**Carve-out:** 64KB on-chip split L1/shared (`cudaFuncSetCacheConfig`: PreferShared/L1/Equal) — more smem steals L1.

### System Design (15 min)

LRU evicts least-recently-used (great locality, scan-polluted); LFU evicts least-frequent (scan-resistant, needs aging); ARC tracks both recency+frequency and self-tunes the split.

## Build / Run (from repo root)

```bash
nvcc Day08/vec_add.cu -o /tmp/d8base -arch=native --ptxas-options=-v && /tmp/d8base
nvcc Day08/regs_heavy.cu -o /tmp/d8heavy -arch=native --ptxas-options=-v && /tmp/d8heavy
nvcc Day08/regs_heavy.cu -o /tmp/d8spill -arch=native --ptxas-options=-v -maxrregcount=24 && /tmp/d8spill
```
