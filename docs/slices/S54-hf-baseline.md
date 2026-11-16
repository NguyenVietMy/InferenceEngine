# S54 — HF transformers baseline
**Phase:** F · **Target date:** Jan 9 · **Depends on:** S53

## Goal
A same-hardware, same-workload comparison against the most common way people run these models.

## Steps
1. Add an `hf` backend to `bench/run.py`: bf16, `attn_implementation="sdpa"`, on the GPU, greedy, with the same prompt lengths.
2. Measure TTFT (prefill) and decode tok/s separately. A `TextStreamer`-style timing hook or a manual loop works.

## Done when
- [ ] HF rows are in the CSV for 1B and 3B.

## Notes
