#!/usr/bin/env python3
"""A quick outside measurement of one checkpoint, before its weights are deleted.

    python3 compression_bench/scripts/checkpoint_probe.py --task compression_tasks/<task> --snapshot <dir> \\
        --out <probe.json> [--gpu N] [--blocks 32] [--items 8] [--minutes 9]

The snapshot (a submission: `inference.py` and its weights) is loaded through the evaluator's own runtime
(`SubmittedRuntime`) in an offline container of the task's runtime image on a SPARE card -- claimed through
the harness's card claims, never one an evolve or another user holds -- and measured on a subset:

  ppl        the first --blocks public perplexity blocks, scored exactly as the Checker scores them;
             BF16's perplexity on the same blocks comes from the task's measured baseline (the per-block
             evidence `build_fidelity_assets.py` writes under `compression_bench/evolve_space/baselines/`), so
             `reward_estimate = 10 * PPL_bf16 / PPL` is comparable with the real reward (a subset, so noisy)
  accuracy   the first --items items of each hidden gate benchmark, generated greedily and judged by the
             gate's own judge, beside BF16's correctness on the same items (from its measured baseline)

Everything is bounded by --minutes (default 9, so 5-10 minutes in all with the container start). The result
is written after each phase, so a timeout keeps what was measured: `{"status": "complete" | "partial" |
"failed", ...}`. The hidden items stay on the host and in the evaluator's memory, the way the verdict has them;
the container has no network.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2] if len(HERE.parents) > 2 else HERE.parent     # inside the container the script sits in /work
FINAL_VALIDATE = REPO / "compression_bench" / "harness" / "final_validate"
EVALUATOR = ("ppl_benchmark.py", "accuracy_benchmark.py", "inference_runtime.py", "model_memory.py",
             "task_policy.py", "own_compression.py", "tokenizer_assets.py")


# --- inside the container -----------------------------------------------------------------------

def inside(work: Path) -> int:
    sys.path.insert(0, str(work / "evaluator"))
    spec = json.loads((work / "spec.json").read_text())
    result = {"status": "partial", "blocks": spec["blocks"], "items": spec["items"], "started": time.time()}
    out = work / "result.json"

    def save():
        result["seconds"] = round(time.time() - result["started"], 1)
        out.write_text(json.dumps(result, indent=1) + "\n")

    try:
        import accuracy_benchmark
        import ppl_benchmark
        from inference_runtime import SubmittedRuntime
        from transformers import AutoTokenizer

        cfg_ppl, rows = ppl_benchmark.load_inputs(work / "ppl", work / "tokenizer", expected_split="public")
        protocol, prefix, _ = ppl_benchmark.protocol_for("public")
        cfg_acc, items = accuracy_benchmark.load_inputs(work / "accuracy", expected_split="hidden")
        tokenizer = AutoTokenizer.from_pretrained(str(work / "tokenizer"), local_files_only=True)
        began = time.time()
        with SubmittedRuntime(work / "submission", max_model_len=accuracy_benchmark.MAX_MODEL_LEN, max_num_seqs=32,
                              gpu_memory_utilization=0.9, timeout=spec["timeout"]) as runtime:
            result["load_seconds"] = round(time.time() - began, 1)
            nll = []
            for offset in range(0, spec["blocks"], 16):
                batch = rows[offset:offset + 16]
                for ids, logprobs in zip(batch, runtime.token_logprobs(batch), strict=True):
                    nll.append(-sum(logprobs[prefix:]))
            per_block = ppl_benchmark.CONTEXT - prefix
            bf16 = spec["bf16_nll"][:len(nll)]
            ppl = math.exp(sum(nll) / (len(nll) * per_block))
            bf16_ppl = math.exp(sum(bf16) / (len(bf16) * per_block))
            result["ppl"] = {"blocks": len(nll), "ppl": ppl, "bf16_ppl_same_blocks": bf16_ppl,
                             "ppl_fidelity": bf16_ppl / ppl, "reward_estimate": 10 * bf16_ppl / ppl,
                             "nll_per_block": nll, "bf16_nll_per_block": bf16}
            save()
            for b, bench in enumerate(cfg_acc["benchmarks"] if spec["items"] else []):
                picked = [i for i, item in enumerate(items) if item["b"] == b][:spec["items"]]
                outputs = runtime.generate([items[i]["prompt"] for i in picked], max_new_tokens=bench["max_new_tokens"],
                                           vocab_size=cfg_acc["vocab_size"])
                texts = {n: tokenizer.decode(ids, skip_special_tokens=True) for n, ids in enumerate(outputs)}
                scored = accuracy_benchmark.score({"benchmarks": [dict(bench, items=len(picked))]},
                                                  [dict(items[i], b=0) for i in picked], texts)
                row = scored["benchmarks"][bench["name"]]
                result.setdefault("accuracy", {})[bench["name"]] = {
                    "correct": row["correct"], "items": row["items"],
                    "bf16_correct": sum(spec["bf16_correct"][bench["name"]][:row["items"]])}
                save()
        result["status"] = "complete"
    except Exception as exc:                                  # noqa: BLE001
        result["status"] = "partial" if "ppl" in result else "failed"
        result["error"] = f"{type(exc).__name__}: {str(exc)[-600:]}"
    save()
    return 0


# --- on the host ---------------------------------------------------------------------------------

def reference(task: Path) -> tuple[list[float], dict]:
    """BF16's per-block NLLs (public perplexity baseline) and per-item correctness (hidden gate baseline)."""
    ppl_cfg = json.loads((task / "assets" / "public" / "ppl_config.json").read_text())
    acc_cfg = json.loads((task / "assets" / "hidden" / "accuracy_config.json").read_text())
    root = task.parents[1]                       # the checkout the task belongs to: its baselines are there
    windows = json.loads((root / ppl_cfg["baseline_evidence"]["result"]).read_text())["windows"]
    try:
        correct = json.loads((root / acc_cfg["bf16"]["evidence"]).read_text())["correct_items"]
    except OSError:
        correct = {}                             # a checkout without the accuracy baselines: no accuracy comparison
    return [w["nll"] for w in sorted(windows, key=lambda w: w["block"])], correct


def probe(task: Path, snapshot: Path, out: Path, *, gpu: int | None, blocks: int, items: int, minutes: float) -> dict:
    sys.path.insert(0, str(REPO / "compression_tasks" / "_manage"))
    sys.path.insert(0, str(REPO / "compression_bench" / "harness"))
    import render
    from checkpoint_janitor import PROBE_LOCK
    from lifecycle import gpus
    contract = render.load_contract(task)
    started = time.time()
    if gpu is None:
        gpu = gpus.pick()
        if gpu is None:
            return {"status": "failed", "error": "no spare GPU"}
    claim = gpus.Claim(gpu, run_id=f"probe-{snapshot.parent.name[:30]}-{snapshot.name}")
    if not claim.acquire():
        return {"status": "failed", "error": f"GPU {gpu} is claimed"}
    lock = snapshot / PROBE_LOCK                    # tells the janitor to leave this snapshot alone
    lock.write_text(f"{os.getpid()}\n")
    work = Path(tempfile.mkdtemp(prefix="probe-", dir=REPO / "compression_bench" / "evolve_space"))
    name = f"compression-bench-probe-{work.name}".lower().replace(".", "-")[:100]
    try:
        bf16_nll, bf16_correct = reference(task)
        for module in EVALUATOR:
            (work / "evaluator").mkdir(exist_ok=True)
            shutil.copy2(FINAL_VALIDATE / module, work / "evaluator" / module)
        evaluator_assets = work / "evaluator" / "evaluator-assets"
        evaluator_assets.mkdir()
        (evaluator_assets / "model_memory_policy.json").write_text(
            json.dumps(dict(render.model_memory_policy(contract), limit_gib=1024)) + "\n")
        ppl_cfg = json.loads((task / "assets" / "public" / "ppl_config.json").read_text())
        (evaluator_assets / "task_policy.json").write_text(
            json.dumps(dict(render.task_policy(contract), size_limit_gib=1024, ppl_protocol=ppl_cfg["protocol"])) + "\n")
        shutil.copytree(task / "assets" / "public", work / "ppl", ignore=shutil.ignore_patterns("decode*", "accuracy*"))
        shutil.copytree(task / "assets" / "hidden", work / "accuracy",
                        ignore=shutil.ignore_patterns("decode*", "ppl*"))
        shutil.copytree(task / "assets" / "tokenizer", work / "tokenizer")
        shutil.copy2(HERE, work / "probe.py")
        (work / "spec.json").write_text(json.dumps({
            "blocks": blocks, "items": items, "timeout": int(minutes * 60), "bf16_nll": bf16_nll,
            "bf16_correct": bf16_correct}))
        command = ["docker", "run", "--rm", "--name", name, "--gpus", f'"device={gpu}"', "--ipc=host", "--network=none",
                   "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp", "-e", "PYTHONDONTWRITEBYTECODE=1",
                   "-e", "VLLM_NO_USAGE_STATS=1", "-v", f"{work}:/work", "-v", f"{snapshot}:/work/submission:ro",
                   "--entrypoint", "python3", render.runtime_image(contract), "/work/probe.py", "--inside", "/work"]
        output = ""
        try:
            got = subprocess.run(command, timeout=minutes * 60 + 90, capture_output=True, text=True)
            output = (got.stderr or got.stdout or "")[-800:]
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        result = json.loads((work / "result.json").read_text()) if (work / "result.json").exists() else {
            "status": "failed", "error": "the container produced no result: " + output.strip()}
        if result.get("status") == "partial" and "error" not in result:
            result["error"] = f"stopped after {minutes:g} minutes"
        result.pop("started", None)
        result.update(gpu=gpu, probed_utc=time.strftime("%FT%TZ", time.gmtime()), wall_seconds=round(time.time() - started, 1),
                      snapshot=str(snapshot), task=task.name)
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        shutil.rmtree(work, ignore_errors=True)
        lock.unlink(missing_ok=True)
        claim.release()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inside")
    parser.add_argument("--task")
    parser.add_argument("--snapshot")
    parser.add_argument("--out")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--blocks", type=int, default=32)
    parser.add_argument("--items", type=int, default=8)
    parser.add_argument("--minutes", type=float, default=9)
    args = parser.parse_args()
    if args.inside:
        return inside(Path(args.inside))
    result = probe(Path(args.task).resolve(), Path(args.snapshot).resolve(), Path(args.out), gpu=args.gpu,
                   blocks=args.blocks, items=args.items, minutes=args.minutes)
    print(json.dumps({k: v for k, v in result.items() if k in ("status", "ppl", "accuracy", "error", "wall_seconds", "load_seconds")}, indent=1))
    return 0 if result.get("status") in ("complete", "partial") else 1


if __name__ == "__main__":
    raise SystemExit(main())
