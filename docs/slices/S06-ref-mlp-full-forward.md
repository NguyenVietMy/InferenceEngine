# S06 — Ref: SwiGLU MLP + full forward
**Phase:** A · **Target date:** Nov 22 · **Depends on:** S05

## Goal
Complete the forward pass: prompt token IDs → logits.

## Steps
1. MLP: `down( silu(gate(x)) * up(x) )`.
2. Decoder layer: `h = x + attn(norm1(x))`; `out = h + mlp(norm2(h))`.
3. Stack all layers → final norm → LM head. The LM head reuses `embed_tokens.weight` because embeddings are tied.
4. Compare logits for the S01 prompt against HF.

## Done when
- [ ] Logits vs HF (fp32), max abs diff < 1e-3.
- [ ] Argmax of the last position equals HF's first generated token.

## Pitfalls
- `silu(x) = x * sigmoid(x)`.
- If layer 0 matches but the logits don't, compare layer by layer. Write a tiny helper now; you'll reuse the habit forever.

## Notes
