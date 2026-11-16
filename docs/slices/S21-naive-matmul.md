# S21 — Naive matmul
**Phase:** C (correct forward pass) · **Target date:** Dec 7 · **Depends on:** S13

## Goal
One correct (slow) matmul kernel you can use for every linear layer, for both prefill and decode.

## Spec
- `Y[T, out] = X[T, in] · Wᵀ`, where `W` is `[out, in]` bf16 and X and Y are fp32.
- Accumulate in fp32.

## Steps
1. One thread per output element `(t, o)`, looping over `in`.
2. Write a CPU reference in the test (a simple triple loop).
3. Test shapes: T ∈ {1, 7}; (in, out) ∈ {(64, 48), (2048, 512), (100, 33)}. Include sizes that aren't multiples of 32.

## Done when
- [ ] All shapes match the CPU reference within tolerance.

## Pitfalls
- It's slow, and that's intended. You'll replace it with GEMV in S38–S40 and the tiled GEMM in S47–S49. Correctness comes first.

## Notes
