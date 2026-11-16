import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = Path.home() / "models" / "Llama-3.2-1B-Instruct"
OUT_PATH = Path(__file__).parent / "golden" / "hf_greedy_1b.json"
MAX_NEW_TOKENS = 50

# 1. Load the tokenizer and model (fp32 = our numerical source of truth).
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
model.eval()

# 2. Build the prompt with the chat template
#    date_string pins the "Today Date" line so the prompt is the same every day.

messages = [{"role": "user", "content": "What is the capital of France? Answer in two sentences."}]
enc = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    date_string="16 Nov 2026",
    return_tensors="pt",
    return_dict=True,
)
prompt_ids = enc["input_ids"]
print("=== prompt as the model sees it ===")
print(tokenizer.decode(prompt_ids[0]))

# 3. Greedy generation: argmax at every step, no randomness.
with torch.no_grad():
    out = model.generate(
        **enc,
        do_sample=False,  # greedy decoding
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.pad_token_id
    )

# 4. generate() returns prompt + new tokens; keep only the new ones.
T = prompt_ids.shape[1]
new_ids = out[0, T:]
print("=== model output ===")
print(tokenizer.decode(new_ids, skip_special_tokens=True))
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(
        {"prompt_ids": prompt_ids[0].tolist(), "output_ids": new_ids.tolist()},
        f,
        indent=2,
    )
print(f"saved {T} prompt ids + {len(new_ids)} output ids -> {OUT_PATH}")