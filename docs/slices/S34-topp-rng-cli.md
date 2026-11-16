# S34 — top-p + seeded RNG + CLI flags
**Phase:** D · **Target date:** Dec 20 · **Depends on:** S33

## Goal
Complete sampling and make it reproducible.

## Steps
1. Add top-p per the S09 spec (smallest prefix with cumulative prob ≥ p, at least 1 token).
2. RNG: `std::mt19937_64` seeded from `--seed`, giving `u` via `std::uniform_real_distribution<double>(0,1)`. Note that the distribution's output isn't guaranteed identical across standard library implementations. That's fine on one machine; document it.
3. CLI: `--temp --top-k --top-p --seed`, also exposed in `chat.py`.
4. Run all S09 golden cases.

## Done when
- [ ] All S09 golden cases pass.
- [ ] Same seed → identical output over 3 runs; different seed → different output.

## Notes
