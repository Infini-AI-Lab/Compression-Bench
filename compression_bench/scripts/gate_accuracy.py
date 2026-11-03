#!/usr/bin/env python3
"""Hidden multi-benchmark gate accuracy of every selected run and checkpoint, from correct-answer counts.

    python3 compression_bench/scripts/gate_accuracy.py [--data DIR]      # writes DIR/gate_accuracy.json

Gate accuracy is the mean accuracy over the three hidden benchmarks (GSM8K = math, HumanEval = code, TriviaQA =
knowledge; 50 items each), in %, recomputed here from each verdict's correct counts (`accuracy.json: benchmarks`),
never from a stored ratio. A submission passes the gate when its gate accuracy is at least 0.8 x the BF16 model's
(ρ = 0.8, chance 0). BF16's counts come from each task's `assets/hidden/accuracy_config.json`.

Runs: `DIR/selected_runs.json` (trajectory_features.py; DIR defaults to evaluation_progress/trajectory_analysis/).
Per run: the final verdict and every measured checkpoint verdict, the latter from `METRICS.json` and from the lineage's
`checkpoint_verdicts.jsonl` (verdicts measured after the run). A verdict without counts (an accuracy stage that
never finished) keeps its status and has no accuracy. The output holds counts and percentages only, no items.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_common import DATA, PROTOTYPES, REPO, lineage_checkpoint_verdicts, selected_runs, task_dir  # noqa: E402

BENCH = ("math", "code", "knowledge")
BENCH_NAME = {"math": "GSM8K", "code": "HumanEval", "knowledge": "TriviaQA"}
RHO = 0.8


def gate_pct(correct: dict, items: dict) -> float:
    """Mean of the per-benchmark accuracies, in %."""
    return 100 * sum(correct[b] / items[b] for b in BENCH) / len(BENCH)


def counts(benchmarks: dict | None):
    if not benchmarks or any(benchmarks.get(b, {}).get("correct") is None for b in BENCH):
        return None
    correct = {b: benchmarks[b]["correct"] for b in BENCH}
    items = {b: benchmarks[b]["items"] for b in BENCH}
    return {"correct": correct, "items": items, "accuracy_pct": gate_pct(correct, items),
            "by_benchmark_pct": {b: 100 * correct[b] / items[b] for b in BENCH}}


def bf16_reference(task: str) -> dict:
    cfg = json.loads((task_dir(task) / "assets" / "hidden" / "accuracy_config.json").read_text())
    ref = counts(cfg["bf16"]["benchmarks"])
    ref["gate_pct"] = RHO * ref["accuracy_pct"]
    return ref


def checkpoint_rows(run_dir: Path) -> dict[str, dict]:
    """version -> the last measured checkpoint verdict of this run (METRICS.json, then the lineage's later ones)."""
    metrics = json.loads((run_dir / "METRICS.json").read_text())
    rows = {c["version"]: c for c in metrics["verdicts"]["checkpoints"] if c.get("measured")}
    rows.update(lineage_checkpoint_verdicts(PROTOTYPES / metrics["run"]["lineage_id"], metrics["run"]["run_id"],
                                            ("SCORED", "ZERO")))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA, help="the analysis directory (selected_runs.json in, output out)")
    args = parser.parse_args()
    out = {"rho": RHO, "benchmarks": BENCH_NAME, "bf16": {}, "runs": []}
    for r in selected_runs(args.data):
        if r["model"] not in out["bf16"]:
            out["bf16"][r["model"]] = bf16_reference(r["task"])
        run_dir = Path(r["run_dir"])
        final = json.loads((run_dir / "METRICS.json").read_text())["verdicts"]["final"]
        detail = final.get("detail") or {}
        row = {k: r[k] for k in ("task", "model", "cap", "agent", "run")}
        row["status"] = final.get("status")
        row["passed"] = bool((detail.get("score") or {}).get("reward"))
        row["failed_stage"] = final.get("failed_stage")
        row["final"] = counts((detail.get("accuracy") or {}).get("benchmarks"))
        cks = []
        for version, c in sorted(checkpoint_rows(run_dir).items(), key=lambda kv: int(kv[0].lstrip("v"))):
            evidence = REPO / (c.get("evidence") or "") / "accuracy.json"
            got = counts(json.loads(evidence.read_text()).get("benchmarks")) if evidence.is_file() else None
            cks.append({"version": version, "status": c.get("status"), "package_gib": c.get("package_gib"),
                        "same_as_final": bool(c.get("same_as_final")), **({"accuracy": got} if got else {})})
        row["checkpoints"] = cks
        out["runs"].append(row)
    for model, ref in out["bf16"].items():
        print(f"{model}: BF16 {ref['accuracy_pct']:.2f}%, gate {ref['gate_pct']:.2f}%, by benchmark "
              + ", ".join(f"{BENCH_NAME[b]} {ref['by_benchmark_pct'][b]:.0f}%" for b in BENCH))
    (args.data / "gate_accuracy.json").write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", args.data / "gate_accuracy.json", len(out["runs"]), "runs,",
          sum(len(r["checkpoints"]) for r in out["runs"]), "checkpoint verdicts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
