#!/usr/bin/env python3
"""Measure a model's BF16 accuracy on the hidden correctness gate and record it in its tasks.

    python3 compression_bench/scripts/deploy/measure_bf16_accuracy.py --gpu N \\
        compression_tasks/<task> [compression_tasks/<task> ...]

The tasks must share one model and identical accuracy inputs (the capacity
tiers of one model). The BF16 checkpoint (`model_for_task.py`) is run through
the task's own starter `inference.py` and the evaluator's
`accuracy_benchmark.measure` -- the code the Checker and the Verifier score
with -- in the task's runtime image (`render.runtime_image`), on one GPU, once
per split (a model too large for one card runs tensor-parallel over the cards
named, `--gpu 6,7`, without the census). The result is written into every
task's `assets/hidden/accuracy_config.json` as its `bf16` baseline;
run `taskctl render` afterwards. Evidence (every prediction) is kept under
`compression_bench/evolve_space/baselines/`.

Inspect the card first (`nvidia-smi --query-compute-apps=...`): this refuses
nothing on its own.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[3] if len(HERE.parents) > 3 else HERE.parent   # in the container: unused
FINAL_VALIDATE = REPO / "compression_bench" / "harness" / "final_validate"
EVALUATOR = ("accuracy_benchmark.py", "inference_runtime.py", "model_memory.py", "task_policy.py")
SPLITS = ("hidden",)


def inside(work: Path, timeout: float) -> None:
    """In the container, with the evaluator's own runtime and a 1 TiB capacity policy."""
    sys.path.insert(0, str(work / "evaluator"))
    import accuracy_benchmark
    from inference_runtime import SubmittedRuntime
    for split in SPLITS:
        started = time.monotonic()
        result = accuracy_benchmark.measure(work / "submission", work / "assets" / split, SubmittedRuntime,
                                            tokenizer_dir=work / "tokenizer", timeout=timeout,
                                            expected_split=split)
        result["seconds"] = time.monotonic() - started
        (work / "results" / f"{split}.json").write_text(json.dumps(result, indent=1) + "\n")
        print(split, f"{result['accuracy']:.4f}", {k: round(v["accuracy"], 4)
                                                   for k, v in result["benchmarks"].items()}, flush=True)


def host(args) -> int:
    sys.path.insert(0, str(REPO / "compression_tasks" / "_manage"))
    import render
    tasks = [Path(t).resolve() for t in args.tasks]
    contracts = [render.load_contract(t) for t in tasks]
    if len({c["model"]["id"] for c in contracts}) != 1:
        raise SystemExit("the tasks do not share one model")
    for split in SPLITS:
        digests = {json.loads((t / "assets" / split / "accuracy_config.json").read_text())["inputs_sha256"]
                   for t in tasks}
        if len(digests) != 1:
            raise SystemExit(f"the tasks' {split} accuracy inputs differ")
    contract, first = contracts[0], tasks[0]
    model = Path(subprocess.check_output([sys.executable, str(REPO / "compression_bench" / "scripts" / "model_for_task.py"),
                                          str(first)], text=True).strip()).resolve()
    image = render.runtime_image(contract)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    work = REPO / "compression_bench" / "evolve_space" / "baselines" / f"{contract['model']['local_name']}-{stamp}"
    for sub in ("evaluator", "submission", "results"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    for name in EVALUATOR:
        shutil.copy2(FINAL_VALIDATE / name, work / "evaluator" / name)
    shutil.copy2(Path(__file__), work / "evaluator" / "measure_bf16_accuracy.py")
    # The census still runs (a BF16 model is not a submission, so its limit is moot).
    (work / "evaluator" / "evaluator-assets").mkdir(exist_ok=True)
    (work / "evaluator" / "evaluator-assets" / "model_memory_policy.json").write_text(
        json.dumps(dict(render.model_memory_policy(contract), limit_gib=1024)) + "\n")
    (work / "evaluator" / "evaluator-assets" / "task_policy.json").write_text(
        json.dumps(dict(render.task_policy(contract), size_limit_gib=1024)) + "\n")
    cards = [int(g) for g in str(args.gpu).split(",")]
    starter = (render.TEMPLATE / "starters" / f"inference.{contract['starter']}.py").read_text()
    if len(cards) > 1:
        if starter.count("tensor_parallel_size=1") != 1:
            raise SystemExit("the starter does not set tensor_parallel_size=1 exactly once")
        starter = starter.replace("tensor_parallel_size=1", f"tensor_parallel_size={len(cards)}")
        # The census reads exactly one GPU worker (build_fidelity_assets.py does the same).
        runtime_py = work / "evaluator" / "inference_runtime.py"
        old = "        self.memory_policy = model_memory.read_policy(Path(__file__).resolve().parent)\n"
        text = runtime_py.read_text()
        if text.count(old) != 1:
            raise SystemExit("inference_runtime.py does not read its census policy where expected")
        runtime_py.write_text(text.replace(old, "        self.memory_policy = None\n"))
    (work / "submission" / "inference.py").write_text(starter)
    # A Hugging Face cache snapshot links into ../../blobs: mount the whole repository.
    mount, inner = ((model.parent.parent, f"/model/snapshots/{model.name}") if model.parent.name == "snapshots"
                    else (model, "/model"))
    (work / "submission" / "model").symlink_to(inner)
    shutil.copytree(first / "assets" / "tokenizer", work / "tokenizer")
    for split in SPLITS:
        (work / "assets" / split).mkdir(parents=True, exist_ok=True)
        for name in ("accuracy_config.json", "accuracy_inputs.json"):
            shutil.copy2(first / "assets" / split / name, work / "assets" / split / name)
    name = f"compression-bench-bf16-acc-{contract['model']['local_name'].lower()}-{stamp.lower()}"
    command = ["docker", "run", "--rm", "--name", name.replace(".", "-"),
               "--gpus", f'"device={args.gpu}"', "--ipc=host", "--network=none",
               "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
               "-e", "PYTHONDONTWRITEBYTECODE=1",
               "-v", f"{work}:/work", "-v", f"{mount}:/model:ro", "--entrypoint", "python3", image,
               "/work/evaluator/measure_bf16_accuracy.py", "--inside", "/work",
               "--timeout", str(args.timeout)]
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)
    for split in SPLITS:
        result = json.loads((work / "results" / f"{split}.json").read_text())
        bf16 = {"accuracy": result["accuracy"], "chance": result["chance"], "items": result["items"],
                "benchmarks": {name: {k: row[k] for k in ("accuracy", "correct", "items", "chance", "kind")}
                               for name, row in result["benchmarks"].items()},
                "model": contract["model"]["id"], "model_revision": contract["model"]["revision"],
                "runtime": f"starter inference.{contract['starter']}.py, {image}"
                           + (f", tensor_parallel_size={len(cards)}, no census" if len(cards) > 1 else ""),
                "measured_utc": stamp, "seconds": round(result["seconds"], 1),
                "evidence": str((work / "results" / f"{split}.json").relative_to(REPO))}
        for task in tasks:
            path = task / "assets" / split / "accuracy_config.json"
            cfg = json.loads(path.read_text())
            cfg.update(baseline_status="measured", bf16=bf16)
            path.write_text(json.dumps(cfg, indent=2) + "\n")
        print(f"{split}: BF16 accuracy {bf16['accuracy']:.4f} (chance {bf16['chance']:.4f}) "
              f"recorded in {len(tasks)} task(s)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tasks", nargs="*")
    parser.add_argument("--gpu", help="a card, or cards for tensor parallelism (6,7)")
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--inside", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.inside:
        inside(args.inside, args.timeout)
        return 0
    if args.gpu is None or not args.tasks:
        parser.error("--gpu and at least one task are required")
    return host(args)


if __name__ == "__main__":
    sys.exit(main())
