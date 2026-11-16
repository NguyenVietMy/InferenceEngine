# S13 — Tensor + device buffer
**Phase:** B · **Target date:** Nov 29 · **Depends on:** S12

## Goal
The smallest data structures you need to pass GPU memory around safely.

## Design (keep it minimal)
- `DeviceBuffer`: owns a `cudaMalloc` pointer. RAII: move-only, frees in the destructor.
- `Tensor`: dtype (`F32`, `BF16`), shape (up to 4 dims), a data pointer, and a non-owning view. Contiguous only; no strides needed in v1.
- Helpers: `numel()`, `nbytes()`, `to_device(host_vec)`, `to_host()`.

## Steps
1. Implement it. Use `__nv_bfloat16` from `<cuda_bf16.h>` for the bf16 type.
2. Tests: alloc/free, H2D→D2H roundtrip for both dtypes, move semantics (no double free).
3. Run the tests under `compute-sanitizer`.

## Done when
- [ ] All tests pass.
- [ ] `compute-sanitizer --leak-check full ./build/tests` reports 0 errors and 0 leaks.

## Pitfalls
- Don't over-engineer. No autograd, no broadcasting, no strides. You can add things when a slice actually needs them.

## Notes
