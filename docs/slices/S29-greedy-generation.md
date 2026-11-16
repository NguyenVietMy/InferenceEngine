# S29 — Prefill + decode loop, greedy 🎯
**Phase:** C · **Target date:** Dec 15 · **Depends on:** S28

## Goal
**The big one:** your CUDA engine generates the same 50 tokens as HF.

## Steps
1. `generate(prompt_ids, max_new)`:
   - Prefill: forward the whole prompt at pos0=0.
   - Loop: copy the last logits to host → argmax → append → forward a single token at pos = len−1.
2. Test: the golden `prompt_ids` from S01 → 50 tokens → compare with `output_ids`.

## Done when
- [ ] 50/50 tokens match for 1B.
- [ ] You've celebrated and committed.

## Pitfalls
- If the first token matches but later ones don't, the decode path is wrong (positions, cache offsets, or the attention length).
- If it diverges at a token where the top-2 logits are nearly tied, compare logits instead. That may be numerics, not a bug (see `tolerances.md`).

## Notes
