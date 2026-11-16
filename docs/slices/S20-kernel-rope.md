# S20 — Kernel: RoPE
**Phase:** B · **Target date:** Dec 6 · **Depends on:** S19

## Goal
Rotate q and k in place on the GPU, matching HF's rotate_half convention.

## Steps
1. On the host, precompute **Llama-3-scaled** inv_freq (port your S04 logic to C++). Build cos/sin tables `[max_seq, head_dim/2]` and upload them once.
2. Kernel: for each (token, head, i < head_dim/2), rotate the pair `(x[i], x[i + head_dim/2])` using the cos/sin at that token's **position**.
3. Pass positions explicitly (a start offset is enough), so decode at pos=37 works.
4. Test: golden `q`/`k` pre-RoPE → kernel → compare with golden post-RoPE, for layer 0.
5. Also test with a start offset ≠ 0 (compute the expected values in Python and dump a small case).

## Done when
- [ ] Matches golden within tolerance for both q (32 heads) and k (8 heads).
- [ ] The offset test passes.
- [ ] Phase B is done 🎉

## Pitfalls
- Pairing is `i` with `i + head_dim/2`, **not** `2i` with `2i+1`.
- Compute the cos/sin table in double on the host, then store it as float.

## Notes
