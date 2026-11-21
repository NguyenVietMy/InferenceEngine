# S02 — Study: Llama architecture
**Phase:** A · **Target date:** Nov 18 · **Depends on:** S01

## Goal
Be able to explain every tensor in the 1B checkpoint and what it does in the forward pass.

## Why
You're about to implement this twice (Python, then CUDA). Knowing the shapes cold makes both much faster to write and debug.

## Steps
1. Watch the attention parts of Karpathy's "Let's build GPT": self-attention, multi-head attention, the residual stream, and LayerNorm.
2. Read `config.json` for the 1B. Note: `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `intermediate_size`, `rope_theta`, `rope_scaling`, `rms_norm_eps`, `tie_word_embeddings`.
3. Write `ref/list_tensors.py`: open `model.safetensors` with the `safetensors` lib and print name, dtype, and shape for every tensor.
4. Write `docs/notes/llama-arch.md`:
   - A table of one layer's 9 tensors, with shapes and the role of each.
   - A diagram (ASCII is fine) of one decoder layer: norm → attn → +residual → norm → MLP → +residual.
   - Explain GQA: 32 query heads share 8 KV heads, so 4 query heads per KV head.

## Done when
- [X] The script lists **146** tensors for 1B (16 layers × 9 + embedding + final norm).
- [X] You can answer without looking: why is `k_proj` 512×2048? Why is there no `lm_head.weight`?
- [X] The notes file exists.

## Pitfalls
- safetensors stores weights as `[out_features, in_features]`, so `y = x @ W.T`.

## Resources
- Karpathy, "Let's build GPT: from scratch, in code, spelled out."
- HF `modeling_llama.py`. Read it, it's short.
