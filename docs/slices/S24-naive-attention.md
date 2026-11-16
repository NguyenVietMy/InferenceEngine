# S24 — Naive attention kernel
**Phase:** C · **Target date:** Dec 10 · **Depends on:** S23

## Goal
Causal GQA attention over the KV cache. It must work for prefill (T tokens) and decode (T=1).

## Spec
- Input: q `[T, n_heads, hd]`, the cache for this layer, `pos0` (position of q's first token).
- Output: `[T, n_heads, hd]` fp32.
- Query token t (absolute position `pos0 + t`) attends to cache positions `[0, pos0 + t]`.

## Steps
1. One block per (t, q_head). The kv head is `q_head / (n_heads / n_kv_heads)`.
2. Pass 1: scores for every key (dot product × 1/√hd) → shared memory. The max length is bounded by max_seq, so size shared memory accordingly, or use global scratch.
3. Pass 2: max, then exp-sum, then normalize. This is softmax.
4. Pass 3: weighted sum over V.
5. Test against golden layer-0 attention output (pre-o_proj) for the multi-token prompt. Use golden k and v to fill the cache.

## Done when
- [ ] Matches golden within tolerance for both prompts.
- [ ] A decode-style test (T=1 at pos0 = T_prompt) matches your Python ref.

## Pitfalls
- Subtract the max before `exp`.
- 4096 floats is 16 KB of shared memory: fine. Know that limit exists.

## Notes
