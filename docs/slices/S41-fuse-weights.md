# S41 — Fuse q/k/v and gate/up
**Phase:** E · **Target date:** Dec 27 · **Depends on:** S40

## Goal
Fewer kernel launches, and better utilization on the small k/v shapes.

## Steps
1. At load time, concatenate `[Wq; Wk; Wv]` into one `[q+2kv, D]` matrix, and `[Wgate; Wup]` into `[2·I, D]`.
2. One GEMV each; q, k, and v become views into the output.
3. Optionally fuse SwiGLU so it reads the fused gate/up output directly.
4. Count launches per decode step in nsys, before and after.

## Done when
- [ ] Tests pass.
- [ ] Fewer launches per step (logged).
- [ ] tok/s is logged.

## Notes
