# S50 — Long-context correctness
**Phase:** E · **Target date:** Jan 5 · **Depends on:** S49

## Goal
Make sure nothing breaks at positions far beyond the short test prompts: RoPE scaling, cache offsets, attention splits.

## Steps
1. Build a ~1500-token prompt (e.g. a long text plus a question about it).
2. Get 200 greedy tokens from your Python ref (fp32) and from the engine.
3. If they diverge, find the first differing step and compare the top-2 logit gap there.

## Done when
- [ ] ≥ 100 tokens match, or the divergence happens at a near-tie (gap below your documented tolerance), with an explanation logged.
- [ ] The test is added to the suite (it can be marked slow).

## Notes
