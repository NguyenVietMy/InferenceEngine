# S23 — Contiguous KV cache
**Phase:** C · **Target date:** Dec 9 · **Depends on:** S22

## Goal
Somewhere to keep K and V for every past position.

## Design
- Per layer, K and V are each `[max_seq, n_kv_heads, head_dim]` fp32.
- One allocation for all layers is fine.
- After RoPE, write the new token(s)' k and v at `[pos, pos+T)`.

## Steps
1. Allocate based on config and `--max-seq` (default 4096).
2. Write a kernel or `cudaMemcpy` that writes k and v into the cache at an offset.
3. Test: write 2 chunks at different offsets, read them back, and compare.
4. Write the memory math in `docs/notes/`. Bytes per token = layers × 2 × n_kv × head_dim × 4.

## Done when
- [ ] The readback test passes.
- [ ] The memory math is documented. Check yourself: 1B is 64 KiB/token (256 MiB at 4096); 3B is 224 KiB/token (896 MiB at 4096).

## Notes
