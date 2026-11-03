#!/usr/bin/env python3
"""Regenerate and verdict the GPTQ baseline of tasks whose BF16 source is downloaded only for the quantization.

    . compression_bench/scripts/env.sh && setsid nohup python3 compression_bench/scripts/baselines/run_nemotron_gptq.py \
        [--task TASK ...] [--gpu-pool 0,1] [--disk PATH] < /dev/null >> nemotron-gptq.log 2>&1 &

Defaults to the two Nemotron settings, which share one BF16 checkpoint. Waits until the disk holding the models
root ($COMPRESSION_BENCH_MODELS_ROOT, default the gitignored compression_bench/models/, where model_for_task.py
looks) has room for the BF16 checkpoint and a margin, and the staging directory (in /dev/shm) room for the packed
outputs; downloads the checkpoint at the revision the contract pins, checks every shard against its LFS sha256,
quantizes every setting into the staging directory, deletes the BF16 copy, then moves each checkpoint into place,
runs its hidden verdict (released_baseline.py) and deletes its weights. A download that would bring the disk under
--guard-gib free is stopped and its partial copy deleted.
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
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE.parent))
import released_models  # noqa: E402

TASKS = ("nemotron3_5_super_vl_120b_a12b_32g", "nemotron3_5_super_vl_120b_a12b_48g")
#: Where model_for_task.py looks first: $COMPRESSION_BENCH_MODELS_ROOT/<local_name>-bf16.
MODELS = Path(os.environ.get("COMPRESSION_BENCH_MODELS_ROOT") or REPO / "compression_bench" / "models")
STAGING = Path(f"/dev/shm/compression-bench-released-gptq-{os.getuid()}")
DRIVER = HERE.parent / "released_baseline.py"
GIB = 1 << 30


def say(*parts):
    print(datetime.now(timezone.utc).strftime("%FT%TZ"), *parts, flush=True)


def free_gib(path: str) -> float:
    return shutil.disk_usage(path).free / GIB


def wait_for_room(args, source_gib: float) -> None:
    while True:
        disk, shm = free_gib(args.disk), free_gib(str(args.staging.parent))
        staged = (sum(p.lstat().st_size for p in args.staging.rglob("*") if p.is_file() and not p.is_symlink()) / GIB
                  if args.staging.is_dir() else 0)
        need_shm = args.staging_gib - staged          # a setting quantized by an earlier attempt already holds its share
        if disk >= source_gib + args.margin_gib and shm >= need_shm:
            say(f"room: {args.disk} {disk:.0f} GiB, {args.staging.parent} {shm:.0f} GiB ({staged:.0f} GiB already staged)")
            return
        say(f"waiting: {args.disk} {disk:.0f} GiB (need {source_gib + args.margin_gib:.0f}), "
            f"{args.staging.parent} {shm:.0f} GiB (need {need_shm:.0f})")
        time.sleep(300)


def download(args, repo_id: str, revision: str, target: Path) -> None:
    code = ("from huggingface_hub import snapshot_download; "
            f"snapshot_download({repo_id!r}, revision={revision!r}, local_dir={str(target)!r}, max_workers=8)")
    process = subprocess.Popen([sys.executable, "-c", code])
    say(f"downloading {repo_id}@{revision} to {target} (pid {process.pid})")
    while process.poll() is None:
        if free_gib(args.disk) < args.guard_gib:
            process.kill()
            process.wait()
            shutil.rmtree(target, ignore_errors=True)
            raise SystemExit(f"stopped: {args.disk} fell under {args.guard_gib} GiB free; the partial copy is deleted")
        time.sleep(30)
    if process.returncode:
        shutil.rmtree(target, ignore_errors=True)
        raise SystemExit(f"download failed ({process.returncode}); the partial copy is deleted")
    shutil.rmtree(target / ".cache", ignore_errors=True)
    import released_identity
    from concurrent.futures import ThreadPoolExecutor
    import released_baseline
    tree = released_identity.tree(repo_id, revision)
    shards = sorted(p for p in tree if p.endswith(".safetensors"))
    with ThreadPoolExecutor(8) as pool:
        digests = dict(zip(shards, pool.map(lambda p: released_baseline.sha256(target / p), shards)))
    bad = [p for p in shards if digests[p] != tree[p]["sha256"]]
    if bad:
        shutil.rmtree(target, ignore_errors=True)
        raise SystemExit(f"sha256 mismatch on {bad}; the copy is deleted")
    say(f"downloaded and verified {len(shards)} shards against their LFS sha256")


def run(*args) -> None:
    say("run", *args)
    subprocess.run([sys.executable, str(DRIVER), *args], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", action="append", help=f"a task sharing the BF16 source (default: {', '.join(TASKS)})")
    parser.add_argument("--gpu-pool", default="0", help="comma-separated GPUs; the first idle one is claimed")
    parser.add_argument("--disk", default=str(MODELS), help="the filesystem the BF16 download lands on")
    parser.add_argument("--staging", type=Path, default=STAGING, help="where the packed checkpoints are written first")
    parser.add_argument("--margin-gib", type=float, default=45, help="free space kept beyond the BF16 checkpoint")
    parser.add_argument("--guard-gib", type=float, default=40, help="stop the download below this much free space")
    parser.add_argument("--staging-gib", type=float, default=85, help="free space the packed outputs need")
    args = parser.parse_args()
    if not os.environ.get("COMPRESSION_BENCH_PY"):
        raise SystemExit("source compression_bench/scripts/env.sh first")
    tasks = args.task or list(TASKS)
    MODELS.mkdir(parents=True, exist_ok=True)
    sources = {(s["repo"], s["revision"]) for s in map(released_models.source_of, tasks)}
    if len(sources) != 1:
        raise SystemExit(f"the tasks do not share one BF16 source: {sorted(sources)}")
    repo_id, revision = sources.pop()
    import yaml
    contract = yaml.safe_load((released_models.task_dir(tasks[0]) / "contract.yaml").read_text())
    target = MODELS / f"{contract['model']['local_name']}-bf16"
    import released_identity
    source_gib = sum(f["size"] for f in released_identity.tree(repo_id, revision).values()) / GIB
    wait_for_room(args, source_gib)
    try:
        if not (target / "config.json").is_file():
            download(args, repo_id, revision, target)
        args.staging.mkdir(exist_ok=True)
        for task in tasks:
            if not (released_models.GPTQ_ROOT / released_models.short(task) / "model").is_dir():
                run("quantize-gptq", task, "--gpu-pool", args.gpu_pool, "--staging", str(args.staging))
    finally:
        if target.exists():
            shutil.rmtree(target)
            say(f"deleted the BF16 copy; {args.disk} {free_gib(args.disk):.0f} GiB free")
    for task in tasks:
        short = released_models.short(task)
        if (args.staging / short).is_dir():
            run("move-gptq", task, "--staging", str(args.staging))
        run("run", task, "--release", "gptq", "--stage", "hidden", "--gpu-pool", args.gpu_pool)
        record = REPO / "evaluation_progress/baselines/released" / short / "gptq/hidden/SUMMARY.json"
        if json.loads(record.read_text()).get("status") == "measured":
            removed = [p for p in (released_models.GPTQ_ROOT / short / "model").rglob("*.safetensors")]
            removed += list((REPO / "compression_bench/evolve_space/baselines/released/work").glob(f"{short}-gptq-*/submission/model/**/*.safetensors"))
            for p in removed:
                p.unlink()
            with (REPO / "compression_bench/evolve_space/baselines/released/weights_deleted.jsonl").open("a") as f:
                f.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "released_weights_deleted": [
                    {"task": task, "release": "gptq", "measured_record": str(record.parent), "deleted": [str(p) for p in removed]}]}) + "\n")
            say(f"{task}: measured; {len(removed)} weight files deleted")
    shutil.rmtree(args.staging, ignore_errors=True)
    run("collect")
    say("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
