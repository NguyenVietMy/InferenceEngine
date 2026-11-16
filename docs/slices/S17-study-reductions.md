# S17 — Study: PMPP ch. 4–5 + reductions
**Phase:** B · **Target date:** Dec 3 · **Depends on:** S12

## Goal
Understand the memory hierarchy, coalescing, and warp-level reduction. RMSNorm, softmax, GEMV, and argmax are all reductions.

## Steps
1. Read PMPP ch. 4 (compute architecture: warps, scheduling) and ch. 5 (memory: shared memory, coalescing). Skim the reduction chapter (ch. 10 in the 4th edition).
2. Implement a sum over N floats, in three versions:
   - v1: naive (interleaved addressing, shared memory).
   - v2: sequential addressing, first add during load.
   - v3: warp shuffle (`__shfl_down_sync`) for the last 32, one atomic or a second pass at the end.
3. Test each one against a CPU sum (use a tolerance; float addition order differs).
4. Benchmark each at N = 2²⁶ and compare GB/s to your S12 number.

## Done when
- [ ] All three are correct.
- [ ] GB/s for each is in `docs/log.md`, with one sentence on why v3 is faster.

## Resources
- Mark Harris, "Optimizing Parallel Reduction in CUDA" (classic slides).

## Notes
