# Day 9: Shared Memory Basics — Tiled Array Reversal

* **Read:** PMPP Ch. 5 continued — shared memory declaration, `__syncthreads`
* **Challenge exercise:** Reverse an array using shared memory tiles
* **Build:** A tiled 1D convolution-lite exercise (just copy-in/copy-out through smem)
* **Tool:** `compute-sanitizer --tool synccheck` on a kernel with a missing `__syncthreads`
* **System Design:** CDN fundamentals & edge caching

## Files

* `reverse_arr.cu` — Harris single-block `reverseArray` (reference) + tiled out-of-place `reverseTiled`, N=1M verify
* `reverse_dynamic.cu` — same, `extern __shared__` + 3rd launch arg (`block.x*sizeof(int)`)
* `conv_tile.cu` — conv-lite: tile load → sync → writeback unchanged, N=1M verify

All: `block(256)`, ceil grid, `if(idx<N)` guards on both sides of the barrier.

## Notes

Tiled reverse = `dout[N-1-idx] = staged s[tx]` — each thread carries its element to the mirror slot; out-of-place keeps blocks independent (no cross-block sync exists). Two-index rule: `threadIdx.x` addresses smem (0–255 tile), global `idx` addresses DRAM. Static (`__shared__ int s[256]`) vs dynamic (`extern __shared__`, bytes at launch) — identical bodies.

### Tool: compute-sanitizer — 4 tools, what each watches

| Tool | Watches | Our result |
|---|---|---|
| `synccheck` | barrier misuse (divergent `__syncthreads`, missing sync around smem cooperation) | silent on both builds — conv-lite threads never cross-read, so nothing to flag |
| `racecheck` | shared/global races (Write-vs-Read hazards across threads) | 0 with barrier; **256 hazards** without, on Harris `d[t]=s[tr]` (still printed "reverse ok" — passing ≠ correct) |
| `memcheck` | OOB accesses, misaligned/invalid copies | clean (guards hold; Day 47: rerun after deliberately breaking an index) |
| `initcheck` | reads of uninitialized device memory | clean (`h_in[i]=i` + full-tile H2D cover every byte read) |

```bash
compute-sanitizer --tool synccheck /tmp/conv9    # 0 errors (no cross-thread sharing)
compute-sanitizer --tool racecheck /tmp/sync_ok  # 0 hazards (barrier present)
compute-sanitizer --tool racecheck /tmp/sync_bad # 1 error: Write@+0x80 vs Read@+0x90 [256 hazards]
compute-sanitizer --tool memcheck /tmp/conv9     # OOB guard check
compute-sanitizer --tool initcheck /tmp/conv9    # uninit-read check
```

Takeaway: pick the tool by suspicion — wrong-barrier → `synccheck`, wrong-results-sometimes → `racecheck`, crash/wrong values → `memcheck`, garbage-first-run → `initcheck`. Sanitizers run 10–100x slower; use tiny N (64–1K), never 1M.

### System Design (15 min)

CDN = origin shielded behind edge PoPs near users; hit ratio decides offload; TTL balances freshness vs traffic; invalidation/purge fixes stale objects.

## Build / Run (from repo root)

```bash
nvcc Day09/reverse_arr.cu -o /tmp/rev9 && /tmp/rev9
nvcc Day09/reverse_dynamic.cu -o /tmp/rev9d && /tmp/rev9d
nvcc Day09/conv_tile.cu -o /tmp/conv9 && /tmp/conv9
compute-sanitizer --tool racecheck /tmp/sync_bad   # demo build, see above
```
