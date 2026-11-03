#!/usr/bin/env python3
"""Find the pinned checkpoint a task names, on this host. Prints its directory.

    model_for_task.py <task dir> [--debug]

Looks, in order, for
  1. $COMPRESSION_BENCH_MODELS_ROOT/<local_name>-bf16, then $COMPRESSION_BENCH_MODELS_ROOT/<local_name>
     (a local model store, bound per host in env.local.sh);
  2. the Hugging Face cache snapshot of <model id>@<revision>
     ($HF_HUB_CACHE, else $HF_HOME/hub, else ~/.cache/huggingface/hub).
A directory counts only if it holds config.json and weight files, including
every shard named in a weight index. `--debug` resolves the
contract's debug_model instead (local_name only; no revision is pinned).
Exits 1 with the places it looked when nothing matches.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

from lifecycle.utils import read_contract  # noqa: E402


def hf_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path("~/.cache/huggingface/hub").expanduser()


def candidates(model: dict, *, models_root: str | None, cache: Path) -> list[Path]:
    found = []
    name = model.get("local_name")
    if models_root and name:
        root = Path(models_root)
        found += [root / f"{name}-bf16", root / name]
    model_id, revision = model.get("id"), model.get("revision")
    if model_id and revision:
        found.append(cache / ("models--" + model_id.replace("/", "--")) / "snapshots" / revision)
    return found


WEIGHT_SUFFIXES = (".safetensors", ".gguf", ".bin")


def is_checkpoint(path: Path) -> bool:
    """A config and weights, with every indexed shard present."""
    if not (path / "config.json").is_file():
        return False
    for index in path.glob("*.index.json"):
        try:
            shards = set(json.loads(index.read_text()).get("weight_map", {}).values())
        except (OSError, ValueError):
            return False
        if any(not (path / shard).is_file() for shard in shards):
            return False
    return any(p.suffix in WEIGHT_SUFFIXES and p.exists() for p in path.iterdir())


def resolve(task_dir: Path, *, debug: bool = False, env=os.environ) -> Path | None:
    contract = read_contract(task_dir)
    model = contract.get("debug_model") if debug else contract.get("model")
    if not model:
        return None
    for path in candidates(model, models_root=env.get("COMPRESSION_BENCH_MODELS_ROOT"),
                          cache=hf_cache()):
        if is_checkpoint(path):
            return path
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task_dir", type=Path)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)
    found = resolve(args.task_dir, debug=args.debug)
    if found is None:
        contract = read_contract(args.task_dir)
        model = (contract.get("debug_model") if args.debug else contract.get("model")) or {}
        looked = candidates(model, models_root=os.environ.get("COMPRESSION_BENCH_MODELS_ROOT"),
                            cache=hf_cache())
        print("no checkpoint found; looked at: " + ", ".join(map(str, looked)), file=sys.stderr)
        return 1
    print(found)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
