#!/usr/bin/env python3
"""Make a baseline independent of its BF16 source without changing any tensor.

The quantizer leaves links to source shards for the few unquantized tensors.
Copy only those indexed tensors, verify their bytes, atomically update the
index, and remove those links. Packed codes/scales/zeros are untouched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def compact(model):
    index_path = model / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    mapping = index["weight_map"]
    suffixes = (".gptq_codes", ".gptq_scales", ".gptq_zeros")
    kept = {name: shard for name, shard in mapping.items() if not name.endswith(suffixes)}
    destination = model / "kept-bf16.safetensors"
    if set(kept.values()) == {destination.name}:
        print("Already compacted", model)
        return
    tensors, evidence = {}, {}
    for shard in sorted(set(kept.values())):
        with safe_open(model / shard, framework="pt", device="cpu") as reader:
            for name, file in kept.items():
                if file != shard:
                    continue
                tensor = reader.get_tensor(name).contiguous().clone()
                tensors[name] = tensor
                evidence[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                                  "bytes": tensor.numel() * tensor.element_size(),
                                  "sha256": hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
    temporary = destination.with_suffix(".tmp")
    save_file(tensors, temporary, metadata={"format": "pt"})
    with safe_open(temporary, framework="pt", device="cpu") as reader:
        assert set(reader.keys()) == set(tensors)
        for name, tensor in tensors.items():
            assert torch.equal(reader.get_tensor(name).reshape(-1).view(torch.uint8), tensor.reshape(-1).view(torch.uint8)), name
    os.replace(temporary, destination)
    for name in kept:
        mapping[name] = destination.name
    temporary_index = index_path.with_suffix(".tmp")
    temporary_index.write_text(json.dumps(index, indent=1) + "\n")
    os.replace(temporary_index, index_path)
    removed = []
    for name in set(kept.values()):
        path = model / name
        if path.is_symlink():
            path.unlink()
            removed.append(name)
    report = {"status": "complete", "model": str(model), "bit_exact": True,
              "packed_tensors_unchanged": True, "kept_tensors": evidence,
              "kept_file_bytes": destination.stat().st_size,
              "removed_source_links": sorted(removed)}
    (model.parent / "compact-kept-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Compacted {len(tensors)} tensors without changing their bytes: {model}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    compact(parser.parse_args().model.resolve())
