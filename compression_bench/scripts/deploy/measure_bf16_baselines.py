#!/usr/bin/env python3
"""Measure a model's BF16 baselines -- perplexity on both splits and the hidden
correctness gate -- for a model too large for one GPU, and record them in its tasks.

    python3 compression_bench/scripts/deploy/measure_bf16_baselines.py --gpu N --cpu-offload-gb 130 \\
        compression_tasks/<task> [...]

The tasks must share one model and identical inputs (the capacity tiers of one
model). The BF16 checkpoint runs through the evaluator's own measurement code
(`ppl_benchmark.measure`, `accuracy_benchmark.measure`) and the task's starter
engine, in the task's runtime image on one GPU, with two changes the starter
cannot make for a checkpoint larger than the card: vLLM's engine runs in the
worker process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) and `cpu_offload_gb` of the
weights stay in host memory, read over PCIe at every forward pass -- the same
arithmetic, slower. Decode latency is not measured (offload makes it meaningless);
the instruction then states no BF16 latency. `--tensor-parallel N` splits the
model over N cards instead (without the census). Results go to every task given:
`assets/{public,hidden}/ppl_config.json` (`bf16_ppl`) and
`assets/hidden/accuracy_config.json` (`bf16`); run `taskctl render` afterwards.
`--measure-only` prints the measurement beside the recorded values instead (a
runtime change is checked this way before any baseline moves).
Evidence is kept under `compression_bench/evolve_space/baselines/`. baseline_latency.py
reuses `engine_source` and the evaluator file list for its BF16 and GPTQ references.

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
EVALUATOR = ("accuracy_benchmark.py", "ppl_benchmark.py", "inference_runtime.py", "model_memory.py",
             "task_policy.py", "tokenizer_assets.py")

#: The starter's engine, made to hold a checkpoint larger than the card.
PREAMBLE = '''
# BF16 baseline engine (measure_bf16_baselines.py): the starter, with vLLM's engine
# in this process and part of the weights offloaded to host memory.
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
_OFFLOAD_GB = float(os.environ.get("COMPRESSION_BENCH_BF16_CPU_OFFLOAD_GB", "0"))
'''

def engine_source(starter: Path, tensor_parallel: int = 1) -> str:
    """The task's starter with PREAMBLE, `tensor_parallel` cards and the CPU offload of COMPRESSION_BENCH_BF16_CPU_OFFLOAD_GB."""
    text = starter.read_text()
    for old, new in (
        ("tensor_parallel_size=1,", f"tensor_parallel_size={tensor_parallel},"),
        ("INFERENCE_API_VERSION = 1\n", "INFERENCE_API_VERSION = 1\n" + PREAMBLE),
        ("generation_config='vllm', disable_log_stats=True, **kwargs)",
         "generation_config='vllm', disable_log_stats=True, cpu_offload_gb=_OFFLOAD_GB, **kwargs)"),
    ):
        if text.count(old) != 1:
            raise SystemExit(f"the starter {starter.name} changed; this tool expects {old.strip()!r} once")
        text = text.replace(old, new)
    compile(text, "inference.py", "exec")
    return text


def inside(work: Path, timeout: float) -> None:
    sys.path.insert(0, str(work / "evaluator"))
    import accuracy_benchmark
    import ppl_benchmark
    from inference_runtime import SubmittedRuntime
    for split in ("public", "hidden"):
        started = time.monotonic()
        result = ppl_benchmark.measure(work / "submission", work / "assets" / split, work / "tokenizer",
                                       SubmittedRuntime, timeout=timeout, expected_split=split)
        result = {k: v for k, v in result.items() if k != "model_memory"}
        result["seconds"] = time.monotonic() - started
        (work / "results" / f"ppl-{split}.json").write_text(json.dumps(result, indent=1) + "\n")
        print("ppl", split, f"{result['ppl']:.6f}", flush=True)
    started = time.monotonic()
    result = accuracy_benchmark.measure(work / "submission", work / "assets" / "hidden", SubmittedRuntime,
                                        tokenizer_dir=work / "tokenizer", timeout=timeout, expected_split="hidden")
    result = {k: v for k, v in result.items() if k != "model_memory"}
    result["seconds"] = time.monotonic() - started
    (work / "results" / "gate-hidden.json").write_text(json.dumps(result, indent=1) + "\n")
    print("gate hidden", f"{result['accuracy']:.4f}",
          {k: round(v["accuracy"], 4) for k, v in result["benchmarks"].items()}, flush=True)


def host(args) -> int:
    sys.path.insert(0, str(REPO / "compression_tasks" / "_manage"))
    import render
    tasks = [Path(t).resolve() for t in args.tasks]
    contracts = [render.load_contract(t) for t in tasks]
    if len({c["model"]["id"] for c in contracts}) != 1:
        raise SystemExit("the tasks do not share one model")
    for name in ("public/ppl_inputs.json", "hidden/ppl_inputs.json", "hidden/accuracy_inputs.json"):
        if len({(t / "assets" / name).read_bytes() for t in tasks}) != 1:
            raise SystemExit(f"the tasks' {name} differ")
    contract, first = contracts[0], tasks[0]
    model = Path(args.model_dir or subprocess.check_output(
        [sys.executable, str(REPO / "compression_bench" / "scripts" / "model_for_task.py"), str(first)],
        text=True).strip()).resolve()
    image = render.runtime_image(contract)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    work = REPO / "compression_bench" / "evolve_space" / "baselines" / f"{contract['model']['local_name']}-{stamp}"
    for sub in ("evaluator/evaluator-assets", "submission", "results"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    for name in EVALUATOR:
        shutil.copy2(FINAL_VALIDATE / name, work / "evaluator" / name)
    shutil.copy2(HERE, work / "evaluator" / HERE.name)
    (work / "evaluator" / "evaluator-assets" / "model_memory_policy.json").write_text(
        json.dumps(dict(render.model_memory_policy(contract), limit_gib=1024)) + "\n")
    (work / "evaluator" / "evaluator-assets" / "task_policy.json").write_text(
        json.dumps(dict(render.task_policy(contract), size_limit_gib=1024)) + "\n")
    (work / "submission" / "inference.py").write_text(
        engine_source(render.TEMPLATE / "starters" / f"inference.{contract['starter']}.py",
                      args.tensor_parallel))
    if args.tensor_parallel > 1:
        # The resident-storage census needs exactly one GPU worker; a baseline drops it anyway.
        runtime_py = work / "evaluator" / "inference_runtime.py"
        old = "        self.memory_policy = model_memory.read_policy(Path(__file__).resolve().parent)\n"
        text = runtime_py.read_text()
        if text.count(old) != 1:
            raise SystemExit("inference_runtime.py changed; --tensor-parallel expects its policy line once")
        runtime_py.write_text(text.replace(old, "        self.memory_policy = None\n"))
    # A Hugging Face snapshot links into its repository's blob store: bind the repository.
    root = model.parents[1] if model.parent.name == "snapshots" else model
    mounts = ["-v", f"{root}:{root}:ro"]
    (work / "submission" / "model").symlink_to(str(model))
    shutil.copytree(first / "assets" / "tokenizer", work / "tokenizer")
    for split in ("public", "hidden"):
        (work / "assets" / split).mkdir(parents=True, exist_ok=True)
        for name in ("ppl_config.json", "ppl_inputs.json") + (
                ("accuracy_config.json", "accuracy_inputs.json") if split == "hidden" else ()):
            shutil.copy2(first / "assets" / split / name, work / "assets" / split / name)
    name = f"compression-bench-bf16-base-{contract['model']['local_name'].lower()}-{stamp.lower()}".replace(".", "-")
    command = ["docker", "run", "--rm", "--name", name, "--gpus", f'"device={args.gpu}"', "--ipc=host",
               "--network=none", "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
               "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", f"COMPRESSION_BENCH_BF16_CPU_OFFLOAD_GB={args.cpu_offload_gb}",
               "-v", f"{work}:/work", *mounts, "--entrypoint", "python3", image,
               f"/work/evaluator/{HERE.name}", "--inside", "/work", "--timeout", str(args.timeout)]
    runtime = (f"starter inference.{contract['starter']}.py with an in-process engine and "
               f"cpu_offload_gb={args.cpu_offload_gb:g}"
               + (f", tensor_parallel_size={args.tensor_parallel}, no census" if args.tensor_parallel > 1 else "")
               + f", {image}")
    (work / "run.json").write_text(json.dumps({"tasks": [str(t) for t in tasks], "runtime": runtime,
                                               "stamp": stamp, "model": str(model)},
                                              indent=1) + "\n")
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)
    return compare(work) if args.measure_only else record(work)


def compare(work: Path) -> int:
    """Print a completed measurement beside the values its tasks record now; change nothing."""
    run = json.loads((work / "run.json").read_text())
    for task in [Path(t) for t in run["tasks"]]:
        for split in ("public", "hidden"):
            new = json.loads((work / "results" / f"ppl-{split}.json").read_text())["ppl"]
            old = json.loads((task / "assets" / split / "ppl_config.json").read_text()).get("bf16_ppl")
            print(f"{task.name} ppl/{split}: recorded {old} measured {new}"
                  + (f" ({(new / old - 1) * 100:+.4f}%)" if old else ""))
        new = json.loads((work / "results" / "gate-hidden.json").read_text())
        old = json.loads((task / "assets" / "hidden" / "accuracy_config.json").read_text()).get("bf16") or {}
        print(f"{task.name} gate: recorded {old.get('accuracy')} measured {new['accuracy']} "
              f"({ {n: r['correct'] for n, r in new['benchmarks'].items()} } correct)")
    print(f"not recorded; to record: --record {work.relative_to(REPO)}")
    return 0


def record(work: Path) -> int:
    """Write a completed measurement's results into every task it was made for."""
    sys.path.insert(0, str(REPO / "compression_tasks" / "_manage"))
    import render
    run = json.loads((work / "run.json").read_text())
    tasks, runtime, stamp = [Path(t) for t in run["tasks"]], run["runtime"], run["stamp"]
    contract = render.load_contract(tasks[0])
    for task in tasks:
        for split in ("public", "hidden"):
            result = json.loads((work / "results" / f"ppl-{split}.json").read_text())
            path = task / "assets" / split / "ppl_config.json"
            cfg = json.loads(path.read_text())
            cfg.update(baseline_status="measured_new_protocol", bf16_ppl=result["ppl"],
                       baseline_evidence={"runtime": runtime, "measured_utc": stamp,
                                          "seconds": round(result["seconds"], 1),
                                          "evidence": str((work / "results" / f"ppl-{split}.json").relative_to(REPO))})
            path.write_text(json.dumps(cfg, indent=1) + "\n")
        result = json.loads((work / "results" / "gate-hidden.json").read_text())
        path = task / "assets" / "hidden" / "accuracy_config.json"
        cfg = json.loads(path.read_text())
        cfg.update(baseline_status="measured", bf16={
            "accuracy": result["accuracy"], "chance": result["chance"], "items": result["items"],
            "benchmarks": {n: {k: row[k] for k in ("accuracy", "correct", "items", "chance", "kind")}
                           for n, row in result["benchmarks"].items()},
            "model": contract["model"]["id"], "model_revision": contract["model"]["revision"],
            "runtime": runtime, "measured_utc": stamp, "seconds": round(result["seconds"], 1),
            "evidence": str((work / "results" / "gate-hidden.json").relative_to(REPO))})
        path.write_text(json.dumps(cfg, indent=2) + "\n")
        import blocks
        text = (task / "contract.yaml").read_text()
        text, _ = blocks.set_scalar(text, "ppl_baseline_status", '"measured_new_protocol"', indent=2, under="metadata")
        blocks.write_checked(task / "contract.yaml", text)
    print(f"recorded in {len(tasks)} task(s): {work.relative_to(REPO)}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tasks", nargs="*")
    parser.add_argument("--gpu", help="a card, or with --tensor-parallel N, N cards: 6,7")
    parser.add_argument("--tensor-parallel", type=int, default=1,
                        help="split the BF16 model over N cards instead of offloading it to host memory")
    parser.add_argument("--model-dir")
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--record", type=Path, help="record a completed work directory's results")
    parser.add_argument("--measure-only", action="store_true",
                        help="measure and print beside the recorded values; record nothing")
    parser.add_argument("--inside", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.record:
        return record(args.record.resolve())
    if args.inside:
        inside(args.inside, args.timeout)
        return 0
    if args.gpu is None or not args.tasks:
        parser.error("--gpu and at least one task are required")
    import re
    if not re.fullmatch(r"\d+(,\d+)*", args.gpu) or len(args.gpu.split(",")) != args.tensor_parallel:
        parser.error("--gpu must name exactly --tensor-parallel cards")
    return host(args)


if __name__ == "__main__":
    sys.exit(main())
