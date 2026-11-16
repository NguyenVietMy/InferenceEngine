# S46 — Study: tiled matmul + cuBLAS reference
**Phase:** E · **Target date:** Jan 1 · **Depends on:** S37

## Goal
Understand why prefill is **compute-bound** (unlike decode), and set a reference bar.

## Steps
1. Read PMPP's tiled matrix multiplication section (ch. 5) and Simon Boehm's "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance," kernels 1–3 closely and skimming the rest.
2. Compute the arithmetic intensity (FLOPs/byte) of a 512×2048 · 2048×8192 matmul. Where does it sit on the roofline vs GEMV?
3. Add cuBLAS to **kbench only** (not the engine): `cublasSgemm` (fp32, SIMT-comparable) and `cublasGemmEx` with bf16 inputs (tensor cores).
4. Table at T=512 for the 4 layer shapes: naive vs cublasSgemm vs cuBLAS-bf16, in GFLOPS.

## Done when
- [ ] The table is logged.
- [ ] You can explain in 3 sentences why tiling helps.

## Notes
