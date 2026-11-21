import torch
from safetensors import safe_open
from pathlib import Path
import json
from torch import Tensor

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



if __name__ == "__main__":
    weights, config = load_weights(MODEL_DIR)
    emb = weights["model.embed_tokens.weight"]
    norm0 = weights["model.layers.0.input_layernorm.weight"]
    print(f"embedding shape: {emb.shape}")      # expect torch.Size([128256, 2048])
    print(f"embedding dtype: {emb.dtype}")      # expect torch.float32
    print(f"layer 0 norm shape: {norm0.shape}") # expect torch.Size([2048])
    print(f"rms_norm_eps: {config['rms_norm_eps']}")  # expect 1e-05


    with open(Path(__file__).parent / "golden" / "hf_greedy_1b.json") as gf:
        golden = json.load(gf)
    ids = torch.tensor(golden["prompt_ids"], dtype=torch.long)
    x = embed(weights,ids)
    print(f"ids shape: {ids.shape}")
    print(f"embed shape: {x.shape}")
    print(torch.equal(x[0], weights["model.embed_tokens.weight"][ids[0]]))

    y = rmsnorm(x, norm0, config["rms_norm_eps"])
    print(y.shape)

    ones_weight = torch.ones(2048)
    tmp = rmsnorm(x, ones_weight, config["rms_norm_eps"])
    rms = torch.sqrt((tmp * tmp).mean(dim=-1))
    print(rms)