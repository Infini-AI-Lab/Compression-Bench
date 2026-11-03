#!/usr/bin/env python3
"""The structure of a safetensors checkpoint, from its config.json and the shards'
headers only: no tensor is read, nothing is loaded, no GPU is used.

    model_inventory.py DIR [--json FILE]

Prints the configuration's structural fields (layer counts and kinds, widths,
heads, experts, vocabulary) and every stored tensor grouped by name pattern
(numeric path components as `*`), with the indices of the layers each group
covers, its stored dtype and shape, its tensor count, parameters and bytes.
`--json` also writes every tensor.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import struct
from pathlib import Path

DTYPE_BITS = {
    "BOOL": 8, "U8": 8, "I8": 8, "F8_E4M3": 8, "F8_E5M2": 8, "F8_E8M0": 8,
    "U16": 16, "I16": 16, "F16": 16, "BF16": 16,
    "U32": 32, "I32": 32, "F32": 32, "U64": 64, "I64": 64, "F64": 64,
}
#: Structural configuration fields, wherever a model family names them.
CONFIG_FIELDS = (
    "num_hidden_layers", "hidden_size", "intermediate_size", "moe_intermediate_size",
    "shared_expert_intermediate_size", "num_attention_heads", "num_key_value_heads",
    "head_dim", "global_head_dim", "num_global_key_value_heads", "sliding_window",
    "num_experts", "num_local_experts", "n_routed_experts", "num_experts_per_tok", "top_k_experts",
    "n_shared_experts", "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim",
    "linear_value_head_dim", "linear_conv_kernel_dim", "mamba_num_heads", "mamba_head_dim",
    "ssm_state_size", "n_groups", "conv_kernel", "hybrid_override_pattern",
    "full_attention_interval", "mtp_num_hidden_layers", "num_nextn_predict_layers",
    "vocab_size", "tie_word_embeddings", "max_position_embeddings", "dtype", "torch_dtype",
)
_INDEX = re.compile(r"(?<=\.)\d+(?=\.|$)")


def read_header(shard: Path) -> dict:
    """One shard's tensor table: name -> {dtype, shape, data_offsets}."""
    with shard.open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        if not 0 < length <= 1 << 28:
            raise ValueError(f"{shard.name}: implausible safetensors header length {length}")
        header = json.loads(handle.read(length))
    header.pop("__metadata__", None)
    return header


def layer_span(indices: list[int], total: int | None) -> str:
    """Indices as a reader states them: all N, every k-th, all but every k-th, or ranges."""
    idx = sorted(set(indices))
    if not idx:
        return ""

    def progression(values):
        if len(values) >= 3 and all(b - a == values[1] - values[0] > 1 for a, b in zip(values, values[1:])):
            return f"{values[0]}, {values[1]}, …, {values[-1]} (every {values[1] - values[0]})"
        return None
    if total and idx == list(range(total)):
        return f"all {total}"
    if progression(idx):
        return progression(idx)
    rest = [i for i in range(total or 0) if i not in set(idx)]
    if total and progression(rest):
        return f"all but {progression(rest)}"
    runs, start = [], idx[0]
    for prev, cur in zip(idx, idx[1:] + [None]):
        if cur != prev + 1:
            runs.append(f"{start}" if start == prev else f"{start}–{prev}")
            start = cur
    return ", ".join(runs)


def scan(model_dir: Path) -> dict:
    model_dir = model_dir.resolve(strict=True)
    index_path = model_dir / "model.safetensors.index.json"
    weight_map = None
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("The weight index must contain a nonempty weight_map")
        shard_names = sorted(set(weight_map.values()))
    else:
        shard_names = [p.name for p in sorted(model_dir.glob("*.safetensors"))]
    if not shard_names:
        raise ValueError("No safetensors weights found")

    tensors, seen = [], set()
    for shard_name in shard_names:
        # A shard is named inside the directory (an HF cache snapshot links it to a blob).
        if Path(shard_name).is_absolute() or ".." in Path(shard_name).parts:
            raise ValueError(f"Shard is outside model directory: {shard_name}")
        shard = model_dir / shard_name
        for name, entry in read_header(shard).items():
            if name in seen:
                raise ValueError(f"Duplicate tensor: {name}")
            if weight_map is not None and weight_map.get(name) != shard_name:
                raise ValueError(f"Index/shard mismatch: {name}")
            seen.add(name)
            dtype, shape = entry["dtype"], list(entry["shape"])
            if dtype not in DTYPE_BITS:
                raise ValueError(f"Unsupported storage dtype {dtype}: {name}")
            elements = math.prod(shape)
            tensors.append({"name": name, "shape": shape, "dtype": dtype, "elements": elements,
                            "bytes": elements * DTYPE_BITS[dtype] // 8, "shard": shard_name})
    if weight_map is not None and set(weight_map) != seen:
        raise ValueError("Index references tensors missing from the shards")
    if not tensors:
        raise ValueError("Checkpoint contains no tensors")

    # How many layers sit under each prefix, so a group's span can say "all N".
    extent: dict[str, int] = {}
    for tensor in tensors:
        match = _INDEX.search(tensor["name"])
        if match:
            prefix = tensor["name"][:match.start()]
            extent[prefix] = max(extent.get(prefix, 0), int(match.group()) + 1)
    groups: dict[tuple, dict] = {}
    for tensor in tensors:
        pattern = _INDEX.sub("*", tensor["name"])
        match = _INDEX.search(tensor["name"])
        key = (pattern, tensor["dtype"], tuple(tensor["shape"]))
        row = groups.setdefault(key, {"pattern": pattern, "dtype": tensor["dtype"],
                                      "shape": tensor["shape"], "count": 0, "elements": 0,
                                      "bytes": 0, "indices": [],
                                      "prefix": tensor["name"][:match.start()] if match else None})
        row["count"] += 1
        row["elements"] += tensor["elements"]
        row["bytes"] += tensor["bytes"]
        if match:
            row["indices"].append(int(match.group()))
    total_bytes = sum(t["bytes"] for t in tensors)
    rows = sorted(groups.values(), key=lambda r: (-r["bytes"], r["pattern"]))
    for row in rows:
        prefix = row.pop("prefix")
        row["layers"] = layer_span(row.pop("indices"), extent.get(prefix)) if prefix else ""
        row["percent"] = 100 * row["bytes"] / total_bytes
    return {"model_dir": str(model_dir), "shard_count": len(shard_names),
            "tensor_count": len(tensors), "parameters": sum(t["elements"] for t in tensors),
            "tensor_bytes": total_bytes,
            "dtypes": {d: sum(t["bytes"] for t in tensors if t["dtype"] == d)
                       for d in sorted({t["dtype"] for t in tensors})},
            "groups": rows, "tensors": tensors}


def config_facts(model_dir: Path) -> dict:
    """The structural fields of config.json, per sub-configuration, and its layer kinds."""
    config = json.loads((Path(model_dir) / "config.json").read_text())
    out = {"architectures": config.get("architectures"), "model_type": config.get("model_type"),
           "parts": {}}
    parts = {"": config}
    for key, value in config.items():
        if key.endswith("_config") and isinstance(value, dict):
            parts[key] = value
    for name, part in parts.items():
        fields = {k: part[k] for k in CONFIG_FIELDS if k in part and part[k] is not None}
        for key in ("layer_types", "layers_block_type", "mtp_layers_block_type"):
            kinds = part.get(key)
            if isinstance(kinds, list) and kinds and all(isinstance(k, str) for k in kinds):
                fields[key] = {kind: f"{kinds.count(kind)}: " + layer_span(
                    [i for i, k in enumerate(kinds) if k == kind], len(kinds))
                    for kind in sorted(set(kinds), key=kinds.index)}
        if fields:
            out["parts"][name or "model"] = {"model_type": part.get("model_type"), **fields}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--json", type=Path, help="also write the configuration and every tensor as JSON")
    args = parser.parse_args()
    try:
        report = scan(args.model_dir)
        report["config"] = config_facts(args.model_dir)
        if args.json:
            # Never write into the scanned model, even outside its read-only mount.
            if args.json.resolve().is_relative_to(Path(report["model_dir"])):
                raise ValueError("Write the report outside the input model directory")
            args.json.write_text(json.dumps(report, indent=2) + "\n")
    except Exception as exc:
        parser.exit(1, f"Inventory failed: {exc}\n")
    config = report["config"]
    print(f"{config['architectures']} ({config['model_type']}): {report['parameters']:,} stored "
          f"parameters in {report['tensor_count']} tensors, {report['shard_count']} shards, "
          f"{report['tensor_bytes'] / 2**30:.3f} GiB; by dtype: "
          + ", ".join(f"{d} {b / 2**30:.3f} GiB" for d, b in report["dtypes"].items()))
    for part, fields in config["parts"].items():
        print(f"[{part}] " + "; ".join(f"{k}={v}" for k, v in fields.items()))
    print("pattern | layers | dtype | shape | tensors | parameters | MiB | %")
    for row in report["groups"]:
        print(f"{row['pattern']} | {row['layers'] or '-'} | {row['dtype']} | {row['shape']} | "
              f"{row['count']} | {row['elements']:,} | {row['bytes'] / 2**20:.2f} | {row['percent']:.4f}")


if __name__ == "__main__":
    main()
