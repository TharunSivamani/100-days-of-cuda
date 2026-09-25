# Day 5: Inside the SM — Warps & Scheduling

* **Read:** PMPP Ch. 4 — Compute architecture and scheduling
* **Challenge exercise:** Print `warpSize`, and which warp/lane a thread belongs to
* **Build:** A kernel that logs warp-level thread grouping for a non-multiple-of-32 grid
* **Blog:** NVIDIA Devblog — "Inside Volta"/"Inside Ampere" architecture posts (whichever matches your GPU)
* **System Design:** Load balancing algorithms (round robin, least-connections, consistent hashing)

## Program execution: software → hardware

## Diagram 1 — launch flow (software → hardware)

```text
[host: malloc/cudaMalloc/H2D] ─▶ [<<<grid(4,3), block(16,16)>>>] ─▶ [grid: 12 blocks × 256 th] ─▶ [block scheduler: blocks → SMs] ─▶ [SM executes] ─▶ [controllers → DRAM] ─▶ [host: D2H/verify/free]
                                                        col = bx*16+tx, row = by*16+ty          by regs/shared/threads
```

## Diagram 2 — inside one SM

```text
[256-thread block] ─▶ [split: 8 warps × 32, tx fastest] ─▶ [warp schedulers: pick eligible warp/cycle] ─┬─▶ [SPs/cores: 1 op/lane, lockstep] ─▶ [regs/shared: __syncthreads() here]
                                                                                                        └─▶ [global load/spill] ─▶ [L2] ─▶ [controllers ─▶ DRAM]
```

## Diagram 3 — warps: hiding + divergence

```text
hide latency:  [warp0: LD ─ ─ ─ ─ ready]   +   [warp1..7: MATH MATH MATH]  ─▶ scheduler swaps free ─▶ no SM idle
divergence:    [if (tid%2): lanes 0,2.. take A] ─▶ [lanes 1,3.. take B] ─▶ serial: ½ throughput (use __syncwarp, branchless where hot)
sizing:        [256 = 8 full warps ✅] vs [100 = 32+32+32+4 ⚠️ 28 lanes idle] vs [257 = 8 full + 31 idle ⚠️]
```

Rules: blocks → SMs (grid scheduler), threads → warps of 32 (SM splits, `x` fastest), partial warps waste lanes — keep `blockDim` a multiple of 32.

## Warps, blocks, threads — the important bits

**Threads/block/grid:** thread = 1 worker (`threadIdx`), block = cooperating group sharing smem + `__syncthreads()` (`blockIdx`), grid = all blocks. Global rank (1D): `tid = blockIdx.x*blockDim.x+threadIdx.x`. A block lives on exactly one SM; an SM hosts several blocks (limit: regs/shared/max-threads, e.g. Ampere 1536 threads/SM).

**Warp = 32 lockstep threads:** hardware executes a warp as one unit — 1 instruction for all 32 lanes per cycle (SIMT). Intra-block rank → `warpId = rank/32`, `lane = rank%32` (`rank = threadIdx.y*blockDim.x+threadIdx.x` in 2D). `warpSize` builtin is 32 on all current GPUs. Non-multiple example: 100 threads/block → 4 warps (32+32+32+4), last warp 28 lanes idle but still consumes issue slots.

**Warp scheduler (per SM, 2-4 of them):** each cycle picks an *eligible* warp (next instruction ready, operands loaded) and issues it — zero-overhead context switch. This hides latency: while one warp waits ~400 cycles on global load, 10+ others compute. More resident warps = more hiding = occupancy matters. Volta+ has independent thread scheduling (per-lane PC/mask, `__syncwarp()`), so divergence no longer deadlocks but still serializes taken/not-taken paths.

**SM anatomy (Ampere-class, your 3060):** 128 FP32/INT SPs + SFUs + Tensor + RT cores per SM; 64K×32-bit registers (spill → local/global); 128KB shared/L1 carve-out; L2 chip-wide; Hopper+: optional block clusters (`cluster.sync()`, distributed smem).

**Traps:** `__syncthreads()` syncs the *block*, not the grid; `if (tid%2)` halves throughput; print 1 lane/warp (`if (lane==0)`); 256 = 8 full warps, 257 = 8 full + 31 idle lanes.