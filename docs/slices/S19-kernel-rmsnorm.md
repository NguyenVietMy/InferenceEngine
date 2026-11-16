# S19 — Kernel: RMSNorm
**Phase:** B · **Target date:** Dec 5 · **Depends on:** S17, S18

## Goal
RMSNorm on the GPU, using your reduction skills.

## Spec
- Input: `x[T, D]` fp32 and weight `[D]` bf16.
- Output: `y[T, D]` fp32.
- Epsilon comes from config.

## Steps
1. One block per row. Each thread accumulates the sum of squares over its strided elements. Reduce with warp shuffles, then across warps via shared memory.
2. Compute `rsqrt(mean + eps)` once, then normalize and scale.
3. Test against golden layer-0 `attn_norm` output (input: golden embed out).
4. Also test a random row with D not a multiple of the block size.

## Done when
- [ ] Within the tolerance from `tolerances.md` (expect ~1e-6 relative).

## Notes
