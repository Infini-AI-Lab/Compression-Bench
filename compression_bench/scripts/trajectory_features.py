#!/usr/bin/env python3
"""One row of outcome and process features per finished run, from its lineage's trajectory, metrics and journal.

    python3 compression_bench/scripts/trajectory_features.py [--out DIR] [--runs FILE | --latest] [--group selected|other|all]

Reads `compression_bench/evolve_space/evolve/*/<run>/METRICS.json` and the lineage's `TRAJECTORY/<run>/steps.jsonl.gz`
(`compression_output/<lineage>/`), and writes `features.json`, `features.csv` and `selected_runs.json` to DIR
(default `evaluation_progress/trajectory_analysis/`, the directory every other analysis script reads). Every finished
run with a final verdict (SCORED or ZERO) gets a row. Its `group` is `selected` when the run is one the analysis
selects, else `other`: the run ids listed in `--runs FILE` (a JSON list of ids, `{"runs": [id or {"run": id}, ...]}`,
or one id per line); with `--latest`, the latest finished run of each agent on each task; with neither, every
finished run. `selected_runs.json` lists the selected runs (task, model, capacity, agent, run id) for the scripts that
follow (insight_data.py, gate_accuracy.py, version_ledger.py, verified_results_table.py).
Process features come from what the agent did: tool calls, its own messages, the files it wrote, and the outputs it
read; none of them uses the hidden verdict, so a feature can be compared with the outcome without circularity.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_common import (DATA, EVOLVE, agent_key, created, model_key, read_selection, short_task,  # noqa: E402
                             steps, utc_seconds)

#: What the agent's commands and files contain; each is a yes/no per run.
METHOD = {
    "gptq": r"\bgptq\b|hessian|cholesky|obq\b|ldlq",
    "rotation": r"hadamard|\brht\b|quarot|spinquant|incoheren",
    "awq_scale": r"\bawq\b|smoothquant|salient|channel.?scal",
    "prune": r"\bprun|reap\b|expert.?drop|drop.?expert|neuron.?compact",
    "codebook_vq": r"codebook|\bvq\b|vector.?quant|trellis|e8\b|aqlm|quip",
    "mixed_precision": r"knapsack|bit.?alloc|mixed.?precision|sensitiv|per.?layer.?bit|tier",
    "distill": r"distill|\bkd\b|kl.?div|fine.?tun|lora|qat\b",
    "kernel": r"triton|cuda.?kernel|\bkernel",
    "vllm_plugin": r"register_quantization_config|quantizationconfig|vllm",
}
CALIBRATION = {"fineweb": r"fineweb", "ultrachat": r"ultrachat", "wikitext": r"wikitext", "self_generated": r"self.?(generated|sample|continuation)|bf16.?(sample|generated|continuation)|sampled from (the )?(bf16|source)"}
ACCURACY_WORDS = r"math-?500|humaneval|gsm8?k|triviaqa|mmlu|hellaswag|arc[-_ ]|boolq|accuracy|exact.?match|pass@1|strict.?match"
PAPER = r"/opt/compression-papers|compression-papers|search\.py"


def row(run_dir: Path, group: str) -> dict | None:
    try:
        m = json.loads((run_dir / "METRICS.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    final = (m.get("verdicts") or {}).get("final") or {}
    if final.get("status") not in ("SCORED", "ZERO"):
        return None
    d = final.get("detail") or {}
    acc = d.get("accuracy") or {}
    task = m.get("task") or {}
    if not task.get("capacity_gib"):          # not a run of a compression task
        return None
    task_id, cap = task["task_id"], float(task["capacity_gib"])
    agent_model = (m.get("agent") or {}).get("agent_model", "")
    package = ((d.get("package") or {}).get("total_gib"))
    ppl = (d.get("ppl") or {}).get("ppl")
    bf16 = (d.get("ppl") or {}).get("bf16_ppl")
    r = {
        "run": m["run"]["run_id"], "group": group, "task_id": task_id,
        "model": model_key(task_id), "cap_gib": cap,
        "agent": agent_key(agent_model), "status": final["status"], "reward": (m.get("verdict") or {}).get("reward") or 0.0,
        "failed_stage": (m.get("verdict") or {}).get("failed_stage"),
        "ppl": ppl, "bf16_ppl": bf16, "ppl_fidelity": (bf16 / ppl) if ppl and bf16 else None,
        "acc": acc.get("accuracy"), "bf16_acc": (m.get("verdict") or {}).get("bf16_accuracy"),
        "retention": (m.get("verdict") or {}).get("accuracy_retention"),
        "acc_error": acc.get("error"),
        "acc_by": {k: v.get("correct") for k, v in (acc.get("benchmarks") or {}).items()},
        "package_gib": package, "utilization": (package / cap) if package else None,
        "decode_s": (m.get("verdict") or {}).get("decode_latency_seconds"),
    }
    usage = (m.get("agent") or {}).get("usage") or {}
    r.update(turns=usage.get("turns"), tool_calls=usage.get("tool_calls"), out_tokens=usage.get("output_tokens"),
             launches=(m.get("agent") or {}).get("launches"), agent_hours=((m.get("agent") or {}).get("seconds") or 0) / 3600)
    started_utc = (m.get("timeline") or {}).get("agent_started_utc")
    started = utc_seconds(started_utc) if started_utc else None
    # public checks: the Checker's feedback an agent could act on
    full = [c for c in ((m.get("public_checks") or {}).get("checks") or []) if c.get("action") == "full-check"]
    ok = [c for c in full if c.get("valid")]
    r["public_full_checks"] = len(full)
    r["public_valid_checks"] = len(ok)
    r["first_valid_h"] = ((utc_seconds(ok[0]["utc"]) - started) / 3600) if ok and started else None
    r["public_reward_first"] = ok[0].get("public_reward") if ok else None
    r["public_reward_last"] = ok[-1].get("public_reward") if ok else None
    r["public_reward_best"] = max((c.get("public_reward") or 0 for c in ok), default=None)
    r["public_minus_hidden"] = (r["public_reward_last"] - r["reward"]) if ok and r["public_reward_last"] is not None and r["reward"] > 0 else None
    r["versions"] = len([v for v in m.get("versions") or [] if v.get("source") == "agent"])
    r["verdict_latency_s"] = (m.get("timing") or {}).get("phases", {}).get("verdict_seconds")
    # process features from the trajectory
    trajectory = steps(m["run"]["run_id"])
    r["has_trajectory"] = bool(trajectory)
    calls = [s for s in trajectory if s["kind"] == "tool_call"]
    results = {s["id"]: s for s in trajectory if s["kind"] == "tool_result"}
    texts = [s.get("text", "") for s in trajectory if s["kind"] in ("message", "thinking")]
    inputs = " ".join(str(c.get("input", "")) for c in calls)
    blob_say = "\n".join(texts)
    r["thinking_chars"] = sum(len(s.get("text", "")) for s in trajectory if s["kind"] == "thinking")
    r["bash_calls"] = sum(1 for c in calls if c.get("tool") == "Bash")
    r["tool_errors"] = sum(1 for s in results.values() if str(s.get("is_error")).lower() == "true")
    r["files_written"] = sum(1 for c in calls if c.get("tool") in ("Write", "Edit"))
    r["paper_calls"] = sum(1 for c in calls if re.search(PAPER, str(c.get("input", "")), re.I))
    r["paper_skill_loaded"] = any(s["kind"] == "input" and "compression-papers" in s.get("text", "")[:400] for s in trajectory)
    r["paper_ids_read"] = sorted({x for c in calls if re.search(PAPER, str(c.get("input", "")), re.I) for x in re.findall(r"\b(\d{4}\.\d{4,5})\b", str(c.get("input", "")))})
    acc_calls = [c for c in calls if c.get("tool") == "Bash" and re.search(ACCURACY_WORDS, str(c.get("input", "")), re.I)
                 and not re.search(PAPER, str(c.get("input", "")), re.I)]
    r["self_accuracy_calls"] = len(acc_calls)
    r["self_accuracy_math500"] = bool(re.search(r"math-?500", inputs, re.I))
    r["mentions_gate"] = bool(re.search(r"accuracy gate|retention|correctness (gate|check)|hidden (accuracy|gate)", blob_say, re.I))
    r["mentions_tradeoff"] = len(re.findall(r"trade-?off|headroom|budget|knapsack|marginal|per.?bit|bits per|bpw|bits/", blob_say, re.I))
    r["mentions_cap"] = len(re.findall(r"\bcap\b|capacity|limit|\b\d+(\.\d+)? ?gib\b", blob_say, re.I))
    haystack = inputs + "\n" + blob_say
    for k, rx in METHOD.items():
        r["m_" + k] = bool(re.search(rx, haystack, re.I))
    for k, rx in CALIBRATION.items():
        r["c_" + k] = bool(re.search(rx, inputs, re.I))
    # the agent's own final note
    notes = [v.get("note") or "" for v in m.get("versions") or [] if v.get("source") == "agent"]
    r["final_note"] = notes[-1] if notes else ""
    r["first_note"] = notes[0] if notes else ""
    return r


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(DATA))
    pick = parser.add_mutually_exclusive_group()
    pick.add_argument("--runs", type=Path, help="the run ids to select (default: every finished run)")
    pick.add_argument("--latest", action="store_true", help="select the latest finished run of each agent on each task")
    parser.add_argument("--group", default="all", choices=["selected", "other", "all"], help="which rows to write")
    args = parser.parse_args()
    rows = [got for got in (row(metrics.parent, "other") for metrics in sorted(EVOLVE.glob("*/*/METRICS.json"))) if got]
    if args.runs:
        wanted = read_selection(args.runs)
        missing = sorted(wanted - {r["run"] for r in rows})
        if missing:
            print(f"not a finished run with a verdict, skipped: {', '.join(missing)}", file=sys.stderr)
    elif args.latest:
        latest = {}
        for r in rows:
            k = (r["task_id"], r["agent"])
            if k not in latest or created(r["run"]) > created(latest[k]):
                latest[k] = r["run"]
        wanted = set(latest.values())
    else:
        wanted = {r["run"] for r in rows}
    for r in rows:
        r["group"] = "selected" if r["run"] in wanted else "other"
    selected = sorted((r for r in rows if r["group"] == "selected"), key=lambda r: (r["task_id"], r["agent"], r["run"]))
    rows = [r for r in rows if args.group in ("all", r["group"])]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "selected_runs.json").write_text(json.dumps({
        "about": "The runs the trajectory analysis selects (trajectory_features.py); read by the scripts that follow it.",
        "runs": [{"task": short_task(r["task_id"]), "task_id": r["task_id"], "model": r["model"], "cap": int(r["cap_gib"])
                  if float(r["cap_gib"]).is_integer() else r["cap_gib"], "agent": r["agent"], "run": r["run"]}
                 for r in selected]}, indent=1) + "\n")
    (out / "features.json").write_text(json.dumps(rows, indent=1, default=str) + "\n")
    flat = [{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items() if k not in ("final_note", "first_note")} for r in rows]
    with (out / "features.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for r in flat for k in r}))
        writer.writeheader()
        writer.writerows(flat)
    print(f"{len(rows)} runs ({len(selected)} selected) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
