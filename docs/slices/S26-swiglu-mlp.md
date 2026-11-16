# S26 — SwiGLU kernel + MLP block
**Phase:** C · **Target date:** Dec 12 · **Depends on:** S25

## Goal
The second half of a decoder layer: `out = h + down(silu(gate(n)) * up(n))`, where `n = norm2(h)`.

## Steps
1. An elementwise kernel: `silu(g) * u`.
2. Chain: RMSNorm → gate and up matmuls → SwiGLU → down matmul → residual.
3. Compare with golden MLP out and layer-0 out.

## Done when
- [ ] Layer-0 output matches golden.

## Notes
