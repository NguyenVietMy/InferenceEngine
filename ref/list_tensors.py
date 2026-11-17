from safetensors import safe_open
from pathlib import Path

TENSOR_PATH = Path.home() / "models" / "Llama-3.2-1B-Instruct" / "model.safetensors"
with safe_open(TENSOR_PATH, framework="pt") as f:
    sum = 0
    for name in f.keys():
        tensors = f.get_tensor(name)
        print(f"{name}: shape: {tensors.shape}, dtype: {tensors.dtype}")
        sum += tensors.numel()
    print(f"total tensors: {len(f.keys())}")
    print(f"total parameters: {sum:,} ({sum / 1e9:.2f}B)")
    print(f"total size: {sum*2:,} bytes")