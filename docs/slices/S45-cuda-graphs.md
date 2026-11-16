# S45 — CUDA Graph for the decode step
**Phase:** E · **Target date:** Dec 31 · **Depends on:** S28, S44

## Goal
Eliminate CPU launch overhead: dozens of launches per token become one graph launch.

## Key problem
A graph bakes in kernel arguments at capture time, but `pos` and the attention length change every step. **Fix:** kernels read `pos` from a device-memory int that you update (a 4-byte memcpy) before each graph launch.

## Steps
1. Refactor decode kernels to read pos and length from a device pointer.
2. Capture one decode step on a stream (`cudaStreamBeginCapture` / `EndCapture` / `cudaGraphInstantiate`).
3. Each step: write pos → `cudaGraphLaunch` → read the token.
4. Compare graph vs non-graph token output for 200 tokens.
5. In nsys, the gaps between kernels should disappear.

## Done when
- [ ] Identical output with and without the graph.
- [ ] tok/s is logged; a before/after nsys screenshot is saved.

## Pitfalls
- Anything that allocates or syncs inside the captured region breaks capture. That's why S28 mattered.
- Split-K attention (S42) with a length-dependent grid size needs a fixed max grid, with blocks exiting early.

## Notes
