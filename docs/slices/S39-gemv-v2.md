# S39 — GEMV v2: vectorized loads
**Phase:** E · **Target date:** Dec 25 · **Depends on:** S38

## Goal
Squeeze bandwidth: fewer, wider memory transactions.

## Ideas to try (measure each change separately)
1. 128-bit loads: 8 bf16 per load (`uint4` or `int4` reinterpret, then unpack). Requires 16-byte alignment, so assert `in % 8 == 0`.
2. Several rows per warp or per block to increase memory-level parallelism.
3. Keep `x` in shared memory (loaded once per block) or registers.
4. Tune the block size.

## Done when
- [ ] Correct.
- [ ] ≥ 75% of peak BW on 2048→8192.
- [ ] A table of what each change bought you is logged. That's blog material.

## Notes
