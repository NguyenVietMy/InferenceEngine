# S36 — Study: Nsight Systems + Nsight Compute
**Phase:** E (make it fast) · **Target date:** Dec 22 · **Depends on:** S35

## Goal
See where decode time actually goes, instead of guessing.

## Steps
1. Nsight Systems: `nsys profile -o decode ./build/infer --bench ...`. Open it in the GUI (Windows side is fine) and look at the timeline: kernels, gaps, memcpys.
2. Nsight Compute on WSL2 needs GPU performance counters enabled in the **Windows** NVIDIA Control Panel (Developer → Manage GPU Performance Counters → allow all users). Try `ncu --set full -k <kernel> -c 1`. If it won't work, fall back to nsys plus your cudaEvent microbenchmarks.
3. Build a breakdown table for one decode step: kernel name | count | total µs | % of step.
4. Write down your top-3 suspects.

## Done when
- [ ] The breakdown table is in `docs/log.md`.
- [ ] You know whether ncu works on your setup (and noted the workaround if not).

## Resources
- GPU MODE lecture 1 (profiling).
- NVIDIA "Nsight Compute: Kernel Profiling Guide" (the roofline and memory sections).

## Notes
