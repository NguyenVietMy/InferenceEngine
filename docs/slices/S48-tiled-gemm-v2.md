# S48 — Tiled GEMM v2: register blocking
**Phase:** E · **Target date:** Jan 3 · **Depends on:** S47

## Goal
Each thread computes a small TM×TN tile of outputs from registers. This is the biggest single jump in Boehm's ladder.

## Steps
1. Extend v1: each thread loads a column of A and a row of B from shared memory into registers, then does a TM×TN outer-product update.
2. Try vectorized shared/global loads if time allows.
3. Tune BM/BN/BK/TM/TN a little (don't rabbit-hole).
4. Test and microbench.

## Done when
- [ ] Correct.
- [ ] ≥ 50% of `cublasSgemm` at T=512 on the 2048→8192 shape.
- [ ] Logged next to the cuBLAS-bf16 (tensor core) number, with a sentence on why that gap exists. It's your post-v1 hook.

## Notes
