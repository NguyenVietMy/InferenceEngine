# S47 — Tiled GEMM v1: shared memory
**Phase:** E · **Target date:** Jan 2 · **Depends on:** S46

## Goal
Reuse each loaded value many times by staging tiles in shared memory.

## Steps
1. Block tile BM×BN, K step BK. Load X and W tiles into shared memory (W converted bf16→fp32 on load), sync, accumulate, repeat.
2. Handle edges (T not a multiple of BM, etc.).
3. Test on random shapes, including odd ones. Microbench at T=512.

## Done when
- [ ] Correct on all test shapes.
- [ ] GFLOPS logged, with the speedup over naive.

## Pitfalls
- Two `__syncthreads()` per K step: after loading and before overwriting.
- Watch for shared memory bank conflicts in the W tile layout (ncu shows them, if it works).

## Notes
