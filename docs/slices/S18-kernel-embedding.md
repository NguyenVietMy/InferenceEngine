# S18 — Kernel: embedding lookup
**Phase:** B · **Target date:** Dec 4 · **Depends on:** S14, S16

## Goal
Your first real model kernel: token IDs → fp32 hidden states.

## Spec
- Input: `ids[T]` (int32, on device) and `embed[V, D]` (bf16).
- Output: `x[T, D]` fp32.

## Steps
1. One block per token; threads stride over D, converting bf16→fp32.
2. Test against golden `p0/embed.bin` using the golden prompt IDs.

## Done when
- [ ] Bit-exact match with golden. bf16→fp32 is exact, so any diff is a bug.

## Pitfalls
- Out-of-range IDs: add a debug-mode assert.

## Notes
