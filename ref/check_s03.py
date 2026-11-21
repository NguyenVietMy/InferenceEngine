from model import load_weights, embed, rmsnorm, MODEL_DIR
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import json
import torch
# 1. Load the tokenizer and model (fp32 = our numerical source of truth).
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
model.eval()

# 1. Load model model.py
weights, config = load_weights(MODEL_DIR)
with open(Path(__file__).parent / "golden" / "hf_greedy_1b.json") as gf:
    golden = json.load(gf)
    ids = torch.tensor(golden["prompt_ids"], dtype=torch.long)
    with torch.no_grad():    
        hf_x = model.model.embed_tokens(ids)
        print(torch.equal(embed(weights, ids), hf_x))
        norm0 = weights["model.layers.0.input_layernorm.weight"]
        eps = config["rms_norm_eps"]
        a = rmsnorm(hf_x, norm0, eps)
        b = model.model.layers[0].input_layernorm(hf_x)
        print((a-b).abs().max().item())
