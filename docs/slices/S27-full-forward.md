# S27 — Full forward pass
**Phase:** C · **Target date:** Dec 13 · **Depends on:** S26

## Goal
Token IDs in, logits out, through all layers.

## Steps
1. Loop the layer function over all layers, using each layer's KV cache.
2. Final RMSNorm, then the LM head (matmul with the **embedding** matrix, since embeddings are tied).
3. Only the last position's logits are needed for generation, but compute all T for this test.
4. Write a test that checks **every** layer's output against golden and reports the first layer that drifts.

## Done when
- [ ] All layers match golden.
- [ ] Logits match within the S10 tolerance.
- [ ] Argmax of the last row equals the first expected token.

## Pitfalls
- The LM head is huge (2048 × 128256). The naive kernel will be slow here; that's fine for now.

## Notes
