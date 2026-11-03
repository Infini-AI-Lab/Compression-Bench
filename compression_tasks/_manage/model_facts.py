"""The checkpoint's structure, frozen into a task and stated in its instruction.

`taskctl model-facts <task>` reads the pinned checkpoint's config.json and its
shards' headers (no tensor is read) with the agent's own inventory tool
(`_template/docker/environment/model_inventory.py`) and writes
`assets/model_facts.json`; `instruction_render` states it under *The
checkpoint*: the architecture, its structural configuration fields with the
layer kinds and which layers have each, and every stored tensor grouped by name
with the layers it covers, its dtype, shape, count, parameters and size.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

INVENTORY = Path(__file__).resolve().parents[1] / "_template" / "docker" / "environment" / "model_inventory.py"
SCHEMA = "compression-bench-model-facts-v1"


def _inventory():
    spec = importlib.util.spec_from_file_location("compression_bench_model_inventory", INVENTORY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(model_dir: Path, contract: dict) -> dict:
    inventory = _inventory()
    report = inventory.scan(Path(model_dir))
    return {
        "schema": SCHEMA, "model_id": contract["model"]["id"], "model_revision": contract["model"]["revision"],
        "config_sha256": hashlib.sha256((Path(model_dir) / "config.json").read_bytes()).hexdigest(),
        "config": inventory.config_facts(Path(model_dir)),
        **{k: report[k] for k in ("shard_count", "tensor_count", "parameters", "tensor_bytes", "dtypes")},
        "groups": report["groups"],
    }


def problems(task_dir: Path, contract: dict) -> list[str]:
    path = task_dir / "assets" / "model_facts.json"
    if not path.is_file():
        return ["assets/model_facts.json is missing (taskctl model-facts)"]
    facts = json.loads(path.read_text())
    if facts.get("schema") != SCHEMA:
        return [f"assets/model_facts.json is not {SCHEMA}"]
    if (facts.get("model_id"), facts.get("model_revision")) != (contract["model"]["id"], contract["model"]["revision"]):
        return ["assets/model_facts.json describes another checkpoint than the contract pins"]
    return []


def _cell(value) -> str:
    if isinstance(value, dict):
        return "; ".join(f"{k}: {v}" for k, v in value.items())
    return f"`{value}`" if isinstance(value, (str, bool)) else f"{value:,}" if isinstance(value, int) else str(value)


def markdown(facts: dict) -> str:
    config = facts["config"]
    dtypes = ", ".join(f"{d} {b / 2**30:.3f} GiB" for d, b in facts["dtypes"].items())
    lines = [f"`{', '.join(config['architectures'] or [])}` (`{config['model_type']}`): "
             f"{facts['parameters']:,} stored parameters in {facts['tensor_count']:,} tensors "
             f"({facts['shard_count']} safetensors shards, {facts['tensor_bytes'] / 2**30:.3f} GiB; {dtypes}). "
             "Its whole configuration is `/input/model/config.json`; "
             "`python3 /app/model_inventory.py /input/model` prints this inventory.",
             "", "| configuration | field | value |", "|---|---|---|"]
    for part, fields in config["parts"].items():
        for key, value in fields.items():
            lines.append(f"| `{part}` | `{key}` | {_cell(value)} |")
    lines += ["", "Every stored tensor, grouped by name (`*` stands for a layer, block or expert index; "
              "*layers* are the indices of the first `*` the group covers):", "",
              "| tensor | layers | dtype | shape | tensors | parameters | MiB |", "|---|---|---|---|---:|---:|---:|"]
    for row in facts["groups"]:
        lines.append(f"| `{row['pattern']}` | {row['layers'] or '–'} | {row['dtype']} | "
                     f"`{row['shape']}` | {row['count']:,} | {row['elements']:,} | {row['bytes'] / 2**20:,.2f} |")
    return "\n".join(lines)
