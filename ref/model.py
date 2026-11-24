import torch
from safetensors import safe_open
from pathlib import Path
import json
from torch import Tensor
import math

MODEL_DIR = Path.home() / "models" / "Llama-3.2-1B-Instruct"
def load_weights(model_dir: Path) -> tuple[dict[str, Tensor], dict]:
    tensors = {}
    tensor_path = model_dir / "model.safetensors"
    config_path = model_dir / "config.json"
    with safe_open(tensor_path, framework="pt") as f:
        for name in f.keys():
            tensor = f.get_tensor(name)
            tensor = tensor.float()
            tensors[name] = tensor
    with open(config_path) as cf:
        config = json.load(cf)

    return tensors, config

def embed(weights: dict[str, Tensor], ids: Tensor) -> Tensor:
    table = weights["model.embed_tokens.weight"]
    return table[ids]

def rmsnorm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    mean_sq = (x*x).mean(dim=-1, keepdim=True)
    size = torch.sqrt(mean_sq+eps)
    return x / size * weight

# each value is just speed in radians per position
def rope_inverse_frequency(config: dict) -> Tensor:
    rope_theta = config["rope_theta"]
    pairs = config["head_dim"] // 2
    i = torch.arange(pairs, dtype=torch.float32)
    # 1 / theta^(i/pairs), not theta^(-i/pairs): same math, but this rounds exactly like HF
    return 1.0 / (rope_theta ** (i / pairs))


def rope_llama3_scale(inv_freq: Tensor, config: dict) -> Tensor:
    high_freq_factor = config["rope_scaling"]["high_freq_factor"]
    low_freq_factor = config["rope_scaling"]["low_freq_factor"]
    factor = config["rope_scaling"]["factor"]
    old_context = config["rope_scaling"]["original_max_position_embeddings"]

    high_cutoff = old_context / high_freq_factor
    low_cutoff = old_context / low_freq_factor
    wavelengths = 2*math.pi / inv_freq

    # 3 candidate version
    slowed = inv_freq / factor
    s = (old_context / wavelengths - low_freq_factor) / (high_freq_factor - low_freq_factor)
    blended = (1 - s) * slowed + s * inv_freq

    # pick per pair
    result = torch.where(wavelengths > low_cutoff, slowed, blended)
    result = torch.where(wavelengths < high_cutoff, inv_freq, result)
    return result


def rope_tables(inv_freq: Tensor, T: int) -> tuple[Tensor, Tensor]:
    pos = torch.arange(T, dtype=torch.float32).unsqueeze(1)
    angles = pos * inv_freq
    return torch.cos(angles), torch.sin(angles)

def linear(x: Tensor, weight: Tensor) -> Tensor:
    return x @ weight.T

def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # x: [T, heads, 64]; cos, sin: [T, 32]
    # arrow i = (number i, number i + 32)  <- rotate_half pairing, NOT neighbors
    half = x.shape[-1] // 2
    a = x[..., :half]                 # first number of each arrow  (numbers 0-31)
    b = x[..., half:]                 # second number of each arrow (numbers 32-63)
    c = cos.unsqueeze(1)              # [T, 1, 32]: same angles for every head
    s = sin.unsqueeze(1)
    return torch.cat((a * c - b * s, a * s + b * c), dim=-1)   # rotate each arrow, join back to 64


if __name__ == "__main__":
    weights, config = load_weights(MODEL_DIR)

    print("\n=== weights ===")
    emb = weights["model.embed_tokens.weight"]
    norm0 = weights["model.layers.0.input_layernorm.weight"]
    print(f"embedding shape: {emb.shape}")      # expect torch.Size([128256, 2048])
    print(f"embedding dtype: {emb.dtype}")      # expect torch.float32
    print(f"layer 0 norm shape: {norm0.shape}") # expect torch.Size([2048])
    print(f"rms_norm_eps: {config['rms_norm_eps']}")  # expect 1e-05

    print("\n=== embedding ===")
    with open(Path(__file__).parent / "golden" / "hf_greedy_1b.json") as gf:
        golden = json.load(gf)
    ids = torch.tensor(golden["prompt_ids"], dtype=torch.long)
    x = embed(weights,ids)
    print(f"ids shape: {ids.shape}")
    print(f"embed shape: {x.shape}")
    print(torch.equal(x[0], weights["model.embed_tokens.weight"][ids[0]]))

    print("\n=== rmsnorm ===")
    y = rmsnorm(x, norm0, config["rms_norm_eps"])
    print(y.shape)

    ones_weight = torch.ones(2048)
    tmp = rmsnorm(x, ones_weight, config["rms_norm_eps"])
    rms = torch.sqrt((tmp * tmp).mean(dim=-1))
    print(rms)

    print("\n=== rope frequencies ===")
    inv_freq = rope_inverse_frequency(config)
    print(f"inverse frequency shape: {inv_freq.shape}")
    print(f"first: {inv_freq[0].item()}")
    print(f"last: {inv_freq[-1].item()}")

    scaled = rope_llama3_scale(inv_freq, config)
    # fast
    print(f"0-14 kept: {torch.equal(scaled[:15], inv_freq[:15])}")
    # slow
    print(f"18-31 slowed: {torch.equal(scaled[18:], inv_freq[18:] / 32)}")
    # blended
    print(f"slowed:  {inv_freq[15:18] / 32}")
    print(f"scaled:  {scaled[15:18]}")
    print(f"kept:    {inv_freq[15:18]}")

    print("\n=== rope tables ===")
    T = ids.shape[0]
    cos, sin = rope_tables(scaled, T)
    print(f"cos shape: {cos.shape}, sin shape: {sin.shape}")          # [47, 32] each
    print(f"row 0 cos all 1: {torch.equal(cos[0], torch.ones(32))}")  # True
    print(f"row 0 sin all 0: {torch.equal(sin[0], torch.zeros(32))}") # True
    print(f"cos[1, 0]: {cos[1, 0].item():.4f}")                       # 0.5403

    print("\n=== layer 0 q and k ===")
    h = rmsnorm(x, norm0, config["rms_norm_eps"])                        # [47, 2048]
    q = linear(h, weights["model.layers.0.self_attn.q_proj.weight"])     # [47, 2048]
    k = linear(h, weights["model.layers.0.self_attn.k_proj.weight"])     # [47, 512]
    print(f"q shape: {q.shape}")
    print(f"k shape: {k.shape}")

    print("\n=== split into heads ===")
    n_heads = config["num_attention_heads"]       # 32
    n_kv_heads = config["num_key_value_heads"]    # 8
    head_dim = config["head_dim"]                 # 64

    q_heads = q.view(T, n_heads, head_dim)
    k_heads = k.view(T, n_kv_heads, head_dim)
    print(f"q_heads shape: {q_heads.shape}")
    print(f"k_heads shape: {k_heads.shape}")
    print(f"head 1 = numbers 64-127: {torch.equal(q_heads[:, 1, :], q[:, 64:128])}")  # True

    print("\n=== rotate q and k ===")
    q_rot = apply_rope(q_heads, cos, sin)
    k_rot = apply_rope(k_heads, cos, sin)
    print(f"q_rot shape: {q_rot.shape}")                                  # [47, 32, 64]
    print(f"k_rot shape: {k_rot.shape}")                                  # [47, 8, 64]
    print(f"position 0 unchanged: {torch.equal(q_rot[0], q_heads[0])}")   # True: angle 0
    # rotation keeps each arrow's length
    len_before = q_heads[..., :32] ** 2 + q_heads[..., 32:] ** 2
    len_after = q_rot[..., :32] ** 2 + q_rot[..., 32:] ** 2
    print(f"arrow lengths kept: {torch.allclose(len_before, len_after, atol=1e-4)}")  # True