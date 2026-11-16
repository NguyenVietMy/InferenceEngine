# S09 — Ref sampler + golden cases
**Phase:** A · **Target date:** Nov 25 · **Depends on:** S08

## Goal
Specify sampling precisely enough that C++ can match it **exactly**, without matching PyTorch's RNG.

## Key idea
`sample(logits, temp, top_k, top_p, u) -> token_id`, where `u ∈ [0,1)` is supplied by the caller. The RNG lives outside, so you only need to match the deterministic part.

## Decide and write down (in a docstring)
- Order of operations: temperature → top-k → top-p → softmax → inverse CDF with `u`.
- `temp == 0` means greedy.
- Sort by logit descending; break ties by **lower token id**.
- Top-p rule: keep the smallest prefix whose cumulative probability ≥ p, and always at least 1 token.
- `top_k == 0` means disabled; `top_p == 1.0` means disabled.

## Steps
1. Implement it in `ref/sampler.py`.
2. Generate ~50 cases (logits from real model outputs plus a few synthetic ones), covering greedy, temp-only, k-only, p-only, combined, k=1, p tiny, exact ties, and u near 0 and near 1.
3. Dump them as golden files (logits `.bin` + a JSON of params/u/expected id).

## Done when
- [ ] ~50 cases dumped, covering every mode and edge case above.
- [ ] Spec docstring is complete enough that someone else could implement it.

## Notes
