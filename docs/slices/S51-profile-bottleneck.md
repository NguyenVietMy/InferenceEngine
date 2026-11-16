# S51 — Re-profile and kill the #1 bottleneck
**Phase:** E · **Target date:** Jan 6 · **Depends on:** S45, S49

## Goal
Let data choose the next optimization, not intuition.

## Steps
1. Redo the S36 breakdown table on the current engine.
2. Pick the single biggest remaining cost that you can address in one session. Likely candidates:
   - the LM head GEMV (it's ~20% of 1B's weight bytes!);
   - the elementwise kernels' share;
   - attention at long context.
3. Fix it, measure, and log the before/after.

## Done when
- [ ] Before/after breakdown tables are logged.
- [ ] tok/s improved, or you've logged why the attempt didn't help.

## Notes
