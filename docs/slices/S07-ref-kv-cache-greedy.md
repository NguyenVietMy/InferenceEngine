# S07 — Ref: KV cache + greedy loop  *(M0 exit)*
**Phase:** A · **Target date:** Nov 23 · **Depends on:** S06

## Goal
Generate token by token using a KV cache, and match HF exactly.

## Why
This is the exact algorithm your CUDA engine will run: one prefill, then N single-token decode steps.

## Steps
1. Cache shape per layer: K and V each `[max_seq, n_kv_heads, head_dim]`.
2. Prefill: run the full prompt, writing K/V for positions [0, T).
3. Decode: feed one token at position `pos`, write its K/V, and attend over [0, pos].
4. Greedy: argmax of the last logits.
5. Also confirm that a cached decode equals an uncached full re-forward for a few steps. This is a great self-test.

## Done when
- [ ] 50 greedy tokens are identical to `ref/golden/hf_greedy_1b.json`.
- [ ] Cached vs uncached logits match (< 1e-4) for the first 5 steps.

## Pitfalls
- RoPE must use the token's **absolute** position, not 0.
- The causal mask during decode is "attend to everything up to pos", so no mask matrix is needed.

## Notes
