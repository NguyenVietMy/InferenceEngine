# S52 — Hit the perf target
**Phase:** E · **Target date:** Jan 7 · **Depends on:** S51

## Goal
1B decode at batch 1 ≥ **115 tok/s** (≥ 50% of 576 GB/s).

## Steps
- **If you're already there:** tune 3B (shapes differ: hidden 3072, head_dim 128) and record the final numbers.
- **If you're not there:** one more profile → fix cycle. Then, regardless of the result, write a "where the time goes" explanation with the breakdown table.

## Done when
- [ ] Final 1B and 3B decode numbers are logged with % of peak.
- [ ] Either the target is met, or there's an honest written analysis of the gap.
- [ ] Phase E is done 🎉 Tag `v0.2-fast`.

## Notes
