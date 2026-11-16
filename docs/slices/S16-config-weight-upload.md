# S16 — Config + weight upload
**Phase:** B · **Target date:** Dec 2 · **Depends on:** S13, S15

## Goal
A `Model` that loads `config.json` and puts every weight on the GPU, organized by layer.

## Steps
1. Write a `Config` struct: all fields from S02, parsed from `config.json`. Compute `head_dim` if it's absent.
2. Write a `Weights` struct: `embed`, `final_norm`, and a `vector<LayerWeights>` with q, k, v, o, gate, up, down, attn_norm, mlp_norm.
3. Upload each tensor (bf16) into a device buffer. Error out clearly if a tensor is missing or has the wrong shape.
4. Print total bytes uploaded.

## Done when
- [ ] 1B loads; the printed total is ≈ 2.47 GB, and nvidia-smi agrees (plus a few hundred MB of context).
- [ ] A test downloads `embed[0]` and `layers[5].q` row 0 back and compares them bit-exact to the Python values.

## Pitfalls
- One big `cudaMalloc` plus offsets would be faster, but per-tensor allocation is fine for v1.

## Notes
