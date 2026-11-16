# S28 — Activation arena
**Phase:** C · **Target date:** Dec 14 · **Depends on:** S27

## Goal
No allocations inside the forward pass. Allocate every buffer once, sized for the max prompt length.

## Steps
1. List every intermediate buffer the forward uses (x, normed, q, k, v, attn, gate, up, logits, …) and its max size.
2. Allocate them all at model init in one arena struct.
3. Add a debug counter or wrapper around `cudaMalloc`, and assert that it stays at 0 during forward.
4. Run a full forward under `compute-sanitizer`.

## Done when
- [ ] Zero mallocs during forward (asserted in a test).
- [ ] compute-sanitizer is clean: no out-of-bounds, no races reported.
- [ ] All S27 tests still pass.

## Why
Speed (cudaMalloc is slow), and it's a prerequisite for CUDA Graphs in S45.

## Notes
