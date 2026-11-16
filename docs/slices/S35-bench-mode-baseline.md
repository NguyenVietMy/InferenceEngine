# S35 — Bench mode + naive baseline
**Phase:** D · **Target date:** Dec 21 · **Depends on:** S34

## Goal
A trustworthy way to measure speed, and the "before" number for everything in Phase E.

## Steps
1. `infer --bench --prompt-len P --gen N --runs R`: synthetic prompt; R timed runs after a warm-up.
2. Report:
   - Prefill time / TTFT (ms).
   - Decode tok/s, median and min/max over runs.
   - **% of peak BW** = (weight bytes per token / time per token) / 576 GB/s. Also report it against your S12 measured peak.
   - GPU clocks and power (shell out to `nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw --format=csv` before and after).
3. Run 1B and 3B and record the numbers in `docs/log.md`.

## Done when
- [ ] Naive numbers for 1B and 3B are logged with clocks.
- [ ] Repeated runs agree within ~5%. If not, find out why (thermal? power mode?).
- [ ] Tag `v0.1-correct`.

## Pitfalls
- Weight bytes per token: count the tied embedding matrix once (it's read by the LM head; the embedding lookup reads just one row).

## Notes
