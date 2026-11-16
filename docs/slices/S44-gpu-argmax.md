# S44 — GPU argmax
**Phase:** E · **Target date:** Dec 30 · **Depends on:** S17, S40

## Goal
Greedy decoding copies 4 bytes per token instead of 512 KB of logits.

## Steps
1. An argmax reduction kernel over 128256 logits (reuse the S17 patterns; reduce (value, index) pairs). Ties go to the lower index, matching the S09 spec.
2. Use it when temp == 0; the sampling path still copies logits.
3. Test against host argmax on random data, including ties.

## Done when
- [ ] Identical tokens to host argmax.
- [ ] tok/s is logged.

## Notes
