# S37 — Kernel microbench harness
**Phase:** E · **Target date:** Dec 23 · **Depends on:** S36

## Goal
Benchmark one kernel at real shapes, fast to iterate on, without running the whole model.

## Steps
1. `kernels/bench/` → an executable `kbench <kernel> [shape]`.
2. For each run: allocate inputs once, warm up, take the median of 50 runs (cudaEvent), report µs, GB/s (bytes of weights + activations), and % of peak.
3. Shapes (1B decode, T=1): 2048→2048 (o), 2048→512 (k/v), 2048→8192 (gate/up), 8192→2048 (down), 2048→128256 (LM head).

## Done when
- [ ] A naive-matmul table at all 5 shapes is in `docs/log.md`.

## Notes
