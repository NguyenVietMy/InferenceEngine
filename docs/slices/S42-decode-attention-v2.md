# S42 — Decode attention v2 (split-K + online softmax)
**Phase:** E · **Target date:** Dec 28 · **Depends on:** S40

## Goal
Make attention fast at long contexts, where naive "one block per head" leaves most SMs idle.

## Ideas
- **Online softmax:** keep a running max and a running sum, rescaling as you go. One pass, no scores buffer.
- **Split along the sequence ("flash-decoding"):** many blocks each handle a chunk of keys and produce a partial (max, sum, output); a second small kernel combines them.
- **GQA-aware:** one block loads a KV head's keys once and serves all 4 query heads that use it.

## Steps
1. Write the online-softmax combine math on paper first. Prove to yourself that merging two partials is exact.
2. Implement it. Test against naive attention at seq len 1, 2, 31, 512, and 4096 (random data).
3. Microbench at 1k and 4k context vs naive.

## Done when
- [ ] Matches naive within tolerance at all lengths.
- [ ] Speedup at 1k/4k context is logged.

## Resources
- Milakov & Gimelshein, "Online normalizer calculation for softmax."
- Tri Dao et al., "Flash-Decoding for long-context inference" (blog).

## Notes
