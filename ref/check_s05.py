from model import (load_weights, embed, rmsnorm, rope_inverse_frequency,
                   rope_llama3_scale, rope_tables, attention, MODEL_DIR)
from transformers import AutoModelForCausalLM
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

h = rmsnorm(embed(weights, ids), weights["model.layers.0.input_layernorm.weight"], config["rms_norm_eps"])
cos, sin = rope_tables(rope_llama3_scale(rope_inverse_frequency(config), config), T)
mine = attention(h, weights, 0, config, cos, sin)

captured = {}
def save_output(module, inputs, output):
    captured["attn"] = output[0]

handle = model.model.layers[0].self_attn.register_forward_hook(save_output)

with torch.no_grad():
    model(ids.unsqueeze(0))

handle.remove()
hf = captured["attn"][0]

print(f"attention max diff:{(mine-hf).abs().max().item()}")