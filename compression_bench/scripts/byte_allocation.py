#!/usr/bin/env python3
"""Where each final submission puts its bytes: resident GiB and effective bits per original weight, by component.

    python3 compression_bench/scripts/byte_allocation.py [--out evaluation_progress/trajectory_analysis]

Inputs: the hidden verdict's census in each run's `METRICS.json` (`verdicts.final.detail.census.largest_groups`: resident
bytes per module group and dtype, covering >= 99.9% of the model) and the task's `assets/model_facts.json` (the BF16
checkpoint's parameter groups). A component's *effective bits per original weight* is its resident bytes (codes, scales,
codebooks and metadata together) times 8 over its original parameter count, so a pruned component is charged for the
weights it dropped. Components: routed experts, shared experts, dense MLP, attention / SSM mixers, embeddings, output head;
the vision tower and multi-token-prediction heads are excluded from the denominators (the text-only
evaluation never runs them).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_common import DATA, REPO, run_dir  # noqa: E402

GIB = 2 ** 30

PARAM_RULES = [  # model_facts group patterns -> component (first match wins)
    ("excluded", r"visual|vision|radio|audio|sound|\.mtp\.|^mtp\."),
    ("embed", r"embed"),
    ("lm_head", r"lm_head"),
    ("shared", r"shared_expert"),
    ("routed", r"experts"),
    ("attention", r"self_attn|linear_attn|\.mixer\.(in_proj|out_proj|qkv_proj|o_proj)|attn"),
    ("dense_mlp", r"\.mlp\.|\.mlp$"),
]
CENSUS_RULES = [  # resident module names -> component
    ("vision", r"visual|vision|radio|audio|sound"),
    ("embed", r"embed"),
    ("lm_head", r"lm_head"),
    ("shared", r"shared"),
    ("routed", r"expert|vq_|moe|\.mlp\.w(13|2)[_.]"),
    ("attention", r"self_attn|linear_attn|mixer\.(in_proj|out_proj|qkv_proj|o_proj)|attn|mamba|ssm"),
    ("dense_mlp", r"\.mlp\.|mlp"),
]
COMPONENTS = ["routed", "shared", "dense_mlp", "attention", "embed", "lm_head"]


def classify(name: str, rules) -> str:
    for component, rx in rules:
        if re.search(rx, name, re.I):
            return component
    return "other"


def parameters(task_dir: Path) -> dict[str, float]:
    facts = json.loads((task_dir / "assets" / "model_facts.json").read_text())
    out = {c: 0.0 for c in COMPONENTS + ["excluded", "other"]}
    for group in facts["groups"]:
        out[classify(group["pattern"], PARAM_RULES)] += group["elements"]
    out["total"] = facts["parameters"]
    if out["routed"] and out["dense_mlp"] < 0.005 * sum(out[c] for c in COMPONENTS):
        out["other"] += out["dense_mlp"]          # an MoE model's only "dense MLP" tensors are the routers
        out["dense_mlp"] = 0.0
    return out


def allocation(run: Path) -> dict | None:
    """Resident GiB, shares and bits per original weight by component of a run directory's final submission."""
    try:
        m = json.loads((run / "METRICS.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    census = (((m.get("verdicts") or {}).get("final") or {}).get("detail") or {}).get("census") or {}
    groups = census.get("largest_groups") or []
    if not groups:
        return None
    task_id = (m.get("task") or {}).get("task_id") or ""
    task_dir = REPO / "compression_tasks" / task_id
    if not (task_dir / "assets" / "model_facts.json").is_file():
        return None
    params = parameters(task_dir)
    resident = {c: 0.0 for c in COMPONENTS + ["vision", "other"]}
    for g in groups:
        resident[classify(g["module"], CENSUS_RULES)] += g["bytes"]
    if not params["dense_mlp"]:
        resident["other"] += resident["dense_mlp"]
        resident["dense_mlp"] = 0.0
    total = sum(resident.values())
    kept = sum(params[c] for c in COMPONENTS)
    out = {"run": m["run"]["run_id"], "task": task_id, "resident_gib": total / GIB,
           "covered": (sum(g["bytes"] for g in groups) / (census.get("worker_bytes") or total)),
           "params_share": {c: params[c] / kept for c in COMPONENTS}, "params": {c: params[c] for c in COMPONENTS},
           "gib": {c: resident[c] / GIB for c in resident}, "share": {c: resident[c] / total for c in resident},
           "bits_per_weight": {c: (resident[c] * 8 / params[c]) if params[c] else None for c in COMPONENTS},
           "overall_bits": total * 8 / kept, "dtypes": {k.replace("torch.", ""): v / GIB for k, v in (census.get("bytes_by_dtype") or {}).items()},
           "unclassified_gib": resident["other"] / GIB}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(DATA))
    args = parser.parse_args()
    rows = json.loads((Path(args.out) / "features.json").read_text())
    got = {}
    for r in rows:
        found = run_dir(r["run"])
        a = allocation(found) if found else None
        if a:
            got[r["run"]] = a
    (Path(args.out) / "byte_allocation.json").write_text(json.dumps(got, indent=1) + "\n")
    print(f"{len(got)} of {len(rows)} runs -> {args.out}/byte_allocation.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
