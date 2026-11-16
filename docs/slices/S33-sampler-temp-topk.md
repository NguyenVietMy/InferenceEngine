# S33 — C++ sampler: temperature + top-k
**Phase:** D (sampling + baseline) · **Target date:** Dec 19 · **Depends on:** S09, S29

## Goal
Port the S09 spec to C++, for temperature and top-k first.

## Steps
1. `int sample(const float* logits, int V, SamplingParams p, float u)`, on the host. 128k floats per token is cheap on the CPU.
2. Follow the S09 spec **exactly**: order of operations, tie-breaking, `temp == 0` → greedy.
3. Load the S09 golden cases in a test and run every case that uses only temp and/or top-k.

## Done when
- [ ] All temp/top-k golden cases pass with an exact token match.

## Pitfalls
- Use `std::partial_sort` or `nth_element` for top-k; a full sort of 128k elements is wasteful (but fine to start with).
- Compute softmax in double, or subtract the max, to avoid `exp` overflow.

## Notes
