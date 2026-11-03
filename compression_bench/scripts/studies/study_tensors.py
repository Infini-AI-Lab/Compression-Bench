"""Tensor helpers shared by the study scripts that run inside a container with this directory mounted at /study:
the SHA-256 of a tensor's raw bytes and the tensor-name index of a safetensors directory."""
import hashlib
from pathlib import Path


def tensor_sha256(t) -> str:
    import torch
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def safetensors_index(root) -> dict:
    """{tensor name: the .safetensors file under root that holds it}; a name in two files is refused."""
    from safetensors import safe_open
    out = {}
    for p in sorted(Path(root).glob("*.safetensors")):
        with safe_open(p, framework="pt") as f:
            for k in f.keys():
                assert k not in out, ("tensor stored twice", k)
                out[k] = p
    return out
