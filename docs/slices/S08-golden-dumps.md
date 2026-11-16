# S08 — Golden dump format + dumps
**Phase:** A · **Target date:** Nov 24 · **Depends on:** S07

## Goal
Define a dead-simple binary format and dump every intermediate activation, so C++ tests can check each kernel in isolation.

## Format (decide and document in `ref/golden/FORMAT.md`)
- 4-byte magic (e.g. `GLD1`)
- int32 `ndim`
- `ndim` × int32 dims
- fp32 data, row-major, little-endian

## Steps
1. Writer and reader in `ref/golden_io.py`. Write a roundtrip test.
2. Dump for **2 prompts**: one short (~8 tokens) and the S01 chat prompt.
   - Inputs: prompt IDs.
   - Model-level: embed out, final norm out, logits.
   - Per layer: attn_norm out; q, k, v **pre-RoPE** and **post-RoPE**; attention output pre-o_proj; attention block out (after residual); mlp_norm out; MLP out; layer out.
3. Name files predictably: `p0/L03_q_rope.bin`, and so on.
4. Gitignore the dumps if they're big. Make the script regenerate them in one command.

## Done when
- [ ] Roundtrip is bit-exact.
- [ ] All dumps exist; `ref/make_golden.py` regenerates them from scratch.
- [ ] FORMAT.md is written.

## Pitfalls
- Dump from **your** ref (S07), not HF. Your ref is the spec you'll mirror.
- Make tensors contiguous before writing.

## Notes
