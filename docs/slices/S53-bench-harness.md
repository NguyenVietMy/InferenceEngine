# S53 — Benchmark harness
**Phase:** F (benchmarks + writeup) · **Target date:** Jan 8 · **Depends on:** S52

## Goal
One script that produces publishable numbers for any engine.

## Steps
1. `bench/run.py --engine ours --model 1b`:
   - Workloads: (prompt 128, gen 256) and (prompt 512, gen 256).
   - 20 runs after warm-up; report TTFT and decode tok/s as median and p90.
   - Log GPU clocks, power, temperature, and driver version.
   - Append rows to `bench/results.csv`.
2. Design it so S54 and S55 just add an `--engine` backend.

## Done when
- [ ] CSV rows for ours × {1B, 3B} × 2 workloads.

## Pitfalls
- Check the laptop is plugged in and on the performance profile. Make the script warn if power draw looks throttled.

## Notes
