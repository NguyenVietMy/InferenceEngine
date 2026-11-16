# S05 — Ref: GQA attention (no cache)
**Phase:** A · **Target date:** Nov 21 · **Depends on:** S04

## Goal
Implement causal grouped-query attention for a full prompt.

## Steps
1. q = x Wqᵀ → reshape to [T, 32, 64]; k, v → [T, 8, 64] (1B shapes, but read them from config).
2. Apply RoPE to q and k.
3. For each q head h, use kv head `h // (n_heads / n_kv_heads)`.
4. scores = q·kᵀ / sqrt(head_dim); mask future positions with -inf; softmax; multiply by v.
5. Concatenate the heads, then apply o_proj.
6. Compare with HF layer 0 `self_attn`. Load HF with `attn_implementation="eager"` so it's the same math.

## Done when
- [ ] Layer-0 attention output vs HF, max abs diff < 1e-4.
- [ ] You can draw the shapes at every step from memory.

## Pitfalls
- Softmax goes over the **key** axis.
- Implement GQA as an index mapping (or `repeat_interleave`). Make sure the grouping is `h // group`, not `h % n_kv`.

## Resources
- Ainslie et al., "GQA" paper (short).

## Notes
