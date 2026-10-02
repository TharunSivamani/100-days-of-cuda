# Day 7: Checkpoint — Foundations Practice Set

* **Read:** Skim your Day 1–6 notes
* **Challenge exercise:** Solve 3 small problems cold — vector add, SAXPY, elementwise max
* **Build:** Package Day 1–6 kernels into one repo with a consistent build script (`Makefile`/`CMakeLists.txt`)
* **Tool:** None today — just clean code
* **System Design:** Caching fundamentals — cache-aside, write-through, write-back

## Files

* `vec_add.cu` — `C[i]=A[i]+B[i]`
* `saxpy.cu` — `Y[i]=a*X[i]+Y[i]` (`a=2.5f` by value, in-place update)
* `emax.cu` — `C[i]=fmaxf(A[i],B[i])` (mixed-sign init so both sides win)

All: 1 thread ↔ 1 element, `block(256)`, `grid=(N+255)/256`, `if(id<N)` guard, dual-`N` verify (`1<<20` + `1000003` odd).

## Notes

Cold protocol: blank editor, no peeking, 30 min each. Fixes from review: `emax.cu` kernel was misnamed `saxpy` (renamed to `emax`); saxpy keeps a pristine `h_Y0` copy since the update is in-place — verifying against the post-launch buffer silently passes.

```text
PASS: N=1048576 vec add correct / N=1000003 vec add correct
PASS: N=1048576 saxpy correct  / N=1000003 saxpy correct
PASS: N=1048576 emax correct   / N=1000003 emax correct
```

## Build / Run (from repo root)

```bash
nvcc Day07/vec_add.cu -o /tmp/d7vec && /tmp/d7vec
nvcc Day07/saxpy.cu -o /tmp/d7saxpy && /tmp/d7saxpy
nvcc Day07/emax.cu -o /tmp/d7emax && /tmp/d7emax
```
