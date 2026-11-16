# S04 — Ref: RoPE with Llama-3 scaling
**Phase:** A · **Target date:** Nov 20 · **Depends on:** S03

## Goal
Implement rotary position embeddings exactly as Llama 3 does, including the frequency scaling.

## Why
RoPE bugs are the #1 cause of "generates fine for 20 tokens, then garbage." Get it exactly right here, once.

## Steps
1. Compute base inverse frequencies: `θ_i = rope_theta^(-2i/head_dim)` for i in [0, head_dim/2).
2. Apply Llama-3 scaling (factor 32, low_freq_factor 1, high_freq_factor 4, original ctx 8192). Derive the piecewise rule from HF's `_compute_llama3_parameters` in `modeling_rope_utils.py`. Read it and write it out in your notes **in words** before coding:
   - high-frequency dims are kept;
   - low-frequency dims are divided by the factor;
   - the middle band is smoothly interpolated.
3. Build cos/sin tables for positions [0, T).
4. Apply to q and k.

## Done when
- [ ] Your inv_freq vector matches HF's rotary embedding buffer, diff < 1e-6.
- [ ] Rotated q/k for layer 0 match HF, diff < 1e-5.

## Pitfalls
- HF Llama uses **rotate_half**: dimension `i` pairs with `i + head_dim/2`, **not** with `i+1` (interleaved). Your CUDA kernel in S20 must do the same. Write this in big letters in your notes.
- Positions start at 0.

## Resources
- RoFormer paper (Su et al.), §3.4.
- EleutherAI blog, "Rotary Embeddings: A Relative Revolution."

## Notes
