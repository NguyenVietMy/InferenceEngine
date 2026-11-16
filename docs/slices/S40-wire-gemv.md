# S40 — Use GEMV in decode
**Phase:** E · **Target date:** Dec 26 · **Depends on:** S39

## Goal
Real end-to-end speedup.

## Steps
1. A linear-layer dispatcher: T == 1 → GEMV; else → matmul.
2. Run the **full** test suite (layer goldens, greedy 1B and 3B, sampler).
3. Bench 1B and 3B.

## Done when
- [ ] All tests pass.
- [ ] New tok/s and % of peak are logged next to the S35 baseline.

## Notes
