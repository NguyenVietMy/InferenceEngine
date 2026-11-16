# S25 — o_proj + residual add
**Phase:** C · **Target date:** Dec 11 · **Depends on:** S24

## Goal
Finish the attention block: `h = x + o_proj(attn)`.

## Steps
1. Matmul for o_proj (reuse S21).
2. A residual-add kernel. Doing it in place is fine.
3. Chain the whole attention block for layer 0 and compare with golden "attention block out".

## Done when
- [ ] Layer-0 attention block output matches golden.

## Notes
