# S03 — Ref: load weights, embedding, RMSNorm
**Phase:** A · **Target date:** Nov 19 · **Depends on:** S02

## Goal
Start `ref/model.py`, your own Llama in plain PyTorch, with weight loading, embedding, and RMSNorm.

## Why
This becomes the readable spec your CUDA code is checked against. Writing it yourself (not copying HF) is where the understanding comes from.

## Steps
1. Load the safetensors into a dict of tensors. Convert **bf16 → fp32**. This is lossless, so the C++ engine (bf16 weights, fp32 math) can match it closely.
2. Embedding: index rows of `embed_tokens.weight`.
3. RMSNorm: `x / sqrt(mean(x²) + eps) * weight`, with eps from config (1e-5).
4. Compare against HF: load the HF model in fp32, then call `model.model.embed_tokens` and `model.model.layers[0].input_layernorm` on the same input.

## Done when
- [ ] Embedding output equals HF exactly.
- [ ] RMSNorm max abs diff vs HF < 1e-5.

## Pitfalls
- RMSNorm takes the mean over the **last** dim (hidden), per token.
- No mean subtraction and no bias. That's what makes it RMSNorm, not LayerNorm.

## Resources
- Zhang & Sennrich, "Root Mean Square Layer Normalization" (skim §3).

## Notes
