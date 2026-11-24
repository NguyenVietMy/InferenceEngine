from model import (load_weights, embed, rmsnorm, linear, rope_inverse_frequency,
                   rope_llama3_scale, rope_tables, apply_rope, MODEL_DIR)
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from pathlib import Path
import json
import torch

model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
model.eval()
weights, config = load_weights(MODEL_DIR)

with open(Path(__file__).parent / "golden" / "hf_greedy_1b.json") as gf:
    golden = json.load(gf)
ids = torch.tensor(golden["prompt_ids"], dtype=torch.long)
T = ids.shape[0]
n_heads = config["num_attention_heads"]
n_kv_heads = config["num_key_value_heads"]
head_dim = config["head_dim"]

with torch.no_grad():
    print("=== speeds ===")
    mine_freq = rope_llama3_scale(rope_inverse_frequency(config), config)
    hf_freq = model.model.rotary_emb.inv_freq
    print(f"max diff: {(mine_freq - hf_freq).abs().max().item()}")          # < 1e-6

    # mine
    x = embed(weights, ids)
    h = rmsnorm(x, weights["model.layers.0.input_layernorm.weight"], config["rms_norm_eps"])
    q = linear(h, weights["model.layers.0.self_attn.q_proj.weight"]).view(T, n_heads, head_dim)
    k = linear(h, weights["model.layers.0.self_attn.k_proj.weight"]).view(T, n_kv_heads, head_dim)
    cos, sin = rope_tables(mine_freq, T)
    mine_q = apply_rope(q, cos, sin)
    mine_k = apply_rope(k, cos, sin)

    # Hugging Face (uses [batch, heads, T, 64], so transpose to compare)
    layer = model.model.layers[0]
    hf_x = model.model.embed_tokens(ids.unsqueeze(0))
    hf_h = layer.input_layernorm(hf_x)
    hf_q = layer.self_attn.q_proj(hf_h).view(1, T, n_heads, head_dim).transpose(1, 2)
    hf_k = layer.self_attn.k_proj(hf_h).view(1, T, n_kv_heads, head_dim).transpose(1, 2)
    hf_cos, hf_sin = model.model.rotary_emb(hf_x, torch.arange(T).unsqueeze(0))
    hf_q, hf_k = apply_rotary_pos_emb(hf_q, hf_k, hf_cos, hf_sin)
    hf_q = hf_q[0].transpose(0, 1)
    hf_k = hf_k[0].transpose(0, 1)

    print("=== rotated q and k, layer 0 ===")
    print(f"q max diff: {(mine_q - hf_q).abs().max().item()}")              # < 1e-5
    print(f"k max diff: {(mine_k - hf_k).abs().max().item()}")              # < 1e-5