# S49 — Use tiled GEMM for prefill
**Phase:** E · **Target date:** Jan 4 · **Depends on:** S48

## Goal
A faster time-to-first-token.

## Steps
1. Dispatcher: T == 1 → GEMV; T > 1 → tiled GEMM.
2. Full test suite.
3. Bench TTFT for 128- and 512-token prompts, before and after.
4. Profile prefill: is attention (still naive for T>1) now the bottleneck? Note it; don't fix it in v1.

## Done when
- [ ] All tests pass.
- [ ] TTFT before/after is logged.

## Notes
