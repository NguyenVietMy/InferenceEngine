# S38 — GEMV v1: one warp per row
**Phase:** E · **Target date:** Dec 24 · **Depends on:** S37

## Goal
The first real decode kernel. At T=1, matmul is matrix × vector and purely **memory-bound**.

## Idea
Naive matmul has each thread walk one row of W, so neighboring threads read addresses `in × 2` bytes apart. That's uncoalesced. Instead, give each **warp** one row: lanes read consecutive elements, then reduce with shuffles.

## Steps
1. Implement it, writing your reasoning about coalescing in comments.
2. Test against the CPU reference (reuse the S21 tests with T=1).
3. Microbench all 5 shapes.

## Done when
- [ ] Correct.
- [ ] ≥ 50% of peak BW on 2048→8192.

## Notes
