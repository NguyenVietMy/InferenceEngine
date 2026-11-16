# S12 — Study: PMPP ch. 2–3 (threads, grids, timing)
**Phase:** B · **Target date:** Nov 28 · **Depends on:** S11

## Goal
Get comfortable with the CUDA execution model, error checking, and measuring a kernel properly.

## Steps
1. Read PMPP ch. 2 (data parallelism, kernels) and ch. 3 (multidimensional grids).
2. Write a `CUDA_CHECK(expr)` macro that prints file:line plus the error string and aborts. Also check `cudaGetLastError()` after launches.
3. Exercise: a 2D matrix-add kernel with a 2D grid. Test it on a non-square, non-multiple-of-block-size matrix.
4. Write a timing helper with `cudaEvent`: warm up, then take the median of N runs.
5. Measure vector-add bandwidth at N = 2²⁶ floats. Bytes moved = 3 × N × 4 (two reads, one write).

## Done when
- [ ] Tests pass for odd sizes (e.g. 1000×777).
- [ ] Vector-add GB/s is logged. Expect >400 GB/s; that's your "achievable peak" reference, alongside the 576 GB/s spec number.

## Pitfalls
- Kernel launches are async. Time with events or after `cudaDeviceSynchronize()`, never with the CPU clock around a bare launch.
- Plug the laptop in and use performance mode, or the numbers mean nothing.

## Notes
