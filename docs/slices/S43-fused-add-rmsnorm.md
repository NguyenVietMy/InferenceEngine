# S43 — Fused residual-add + RMSNorm
**Phase:** E · **Target date:** Dec 29 · **Depends on:** S40

## Goal
Remove a full read/write of the hidden state per norm, and one launch.

## Steps
1. One kernel: `x += delta` (residual), writes `x`, then computes `norm(x)` in the same pass.
2. Replace the add+norm pairs in the layer.
3. Tests, then bench.

## Done when
- [ ] Tests pass.
- [ ] tok/s is logged. It may be small at batch 1; note what you learned about why.

## Notes
