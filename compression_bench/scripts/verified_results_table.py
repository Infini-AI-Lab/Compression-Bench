#!/usr/bin/env python3
"""The results table, every cell from its recorded hidden verdict, with the gate's correct counts behind each accuracy.

    python3 compression_bench/scripts/verified_results_table.py [--data DIR] [--baselines DIR]
                                                     # -> DIR/verified_results.md, DIR/verified_results.json

Accuracy is the hidden correctness gate's multi-benchmark accuracy: the mean of the accuracies on 50 GSM8K (math),
50 HumanEval (code) and 50 TriviaQA (knowledge) items, which with 50 items each is the total correct out of 150
(values in steps of 2/3 of a point). L is the decode time in seconds (median round mean of the 127 intervals between
the 128 output tokens); TPS is 127/L, computed before rounding.

Rows per task: BF16, the task's frozen reference (`tests/evaluator-assets`, what the gate compares against); BF16 as
measured by baseline_latency.py, when its record exists; the GPTQ baseline, from its regenerated hidden verdict
(released_baseline.py); each selected run (`DIR/selected_runs.json`, trajectory_features.py), from its
METRICS.json final verdict; the released quantizations measured by released_baseline.py
(`<baselines>/released/RESULTS.json`). A baseline whose record is missing is left out. DIR
defaults to evaluation_progress/trajectory_analysis/, the baselines root to evaluation_progress/baselines/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_common import (AGENT_KEY, AGENT_NAME, BASELINES, DATA, MODEL_LABEL, REPO, selected_runs,  # noqa: E402
                             short_task, task_dir)

BENCH = ("code", "knowledge", "math")
INTERVALS = 127   # decode intervals between the 128 output tokens of a decode request


def label(task: str) -> str:
    slug, _, cap = task.rpartition("_")
    return f"{MODEL_LABEL.get(slug, slug)} / {cap.rstrip('g')} GiB" if cap.endswith("g") else task


def counts(benchmarks: dict) -> dict:
    return {k: (benchmarks.get(k) or {}).get("correct") for k in BENCH}


def cell(ppl, by: dict, L, passed, source, note=""):
    total = sum(v for v in by.values() if v is not None) if all(v is not None for v in by.values()) else None
    return {"ppl": ppl, "correct": by, "correct_total": total, "accuracy_pct": 100 * total / 150 if total is not None else None,
            "decode_seconds": L, "decode_tps": INTERVALS / L if L else None,
            "passed": passed, "source": source, "note": note}


def agent_cells(data: Path) -> dict:
    found = {}
    for r in selected_runs(data):
        metrics = Path(r["run_dir"]) / "METRICS.json"
        m = json.loads(metrics.read_text())
        task = short_task(m["task"]["task_id"])
        final = (m.get("verdicts") or {}).get("final") or {}
        d = final.get("detail") or {}
        acc = d.get("accuracy") or {}
        c = cell((d.get("ppl") or {}).get("ppl"), counts(acc.get("benchmarks") or {}),
                 (m.get("verdict") or {}).get("decode_latency_seconds"),
                 final.get("status") == "SCORED", str(metrics.relative_to(REPO)),
                 "" if final.get("status") == "SCORED" else f"not passed ({(m.get('verdict') or {}).get('failed_stage') or final.get('status')})")
        c["run"] = r["run"]
        found.setdefault(task, []).append((m["agent"]["agent_model"], c))
    return found


def bf16_cells(task: str, baselines: Path) -> list[tuple[str, dict]]:
    assets = task_dir(task) / "tests" / "evaluator-assets"
    ref = json.loads((assets / "accuracy_config.json").read_text())["bf16"]
    rows = [("BF16 (frozen task reference)",
             cell(json.loads((assets / "ppl_config.json").read_text())["bf16_ppl"], counts(ref["benchmarks"]), None, True,
                  str((assets / "accuracy_config.json").relative_to(REPO)), "the task's frozen reference: what the gate compares against"))]
    rec = baselines / "latency" / "bf16" / task
    if (rec / "SUMMARY.json").is_file() and (rec / "gate-hidden.json").is_file():
        s = json.loads((rec / "SUMMARY.json").read_text())
        g = json.loads((rec / "gate-hidden.json").read_text())
        rows.append(("BF16 (measured)", cell((s.get("ppl_verification") or {}).get("ppl"), counts(g["benchmarks"]),
                                             s.get("decode_seconds"), True, str(rec), "baseline_latency.py record")))
    return rows


def gptq_cells(task: str, baselines: Path) -> list[tuple[str, dict]]:
    regenerated = baselines / "released" / task / "gptq" / "hidden" / "SUMMARY.json"
    if not regenerated.is_file():
        return []
    s = json.loads(regenerated.read_text())
    m = s["measurements"]
    return [("Compression baseline (GPTQ)", cell(
        m["ppl"], {k: (m["accuracy_by_benchmark"].get(k) or {}).get("correct") for k in BENCH}, m["decode_latency_seconds"],
        m.get("accuracy_retention") is not None and (s.get("gates") or {}).get("correctness_gate") is True
        and "resident_capacity" not in (s.get("failed_gates") or []),
        str(regenerated.parent), "regenerated, run through the hidden verdict"))]


def released_cells(task: str, baselines: Path) -> list[tuple[str, dict]]:
    results = baselines / "released" / "RESULTS.json"
    if not results.is_file():
        return []
    out = []
    for r in json.loads(results.read_text())["rows"]:
        if r["task"] != task or r["kind"] != "released" or r.get("status") != "measured":
            continue
        by = {k: (r["accuracy_by_benchmark"].get(k) or {}).get("correct") for k in BENCH}
        gates = r.get("gates") or {}
        other = [g for g in (r.get("failed_gates") or []) if g != "own_compression"]
        note = ("not passed: own-compression gate (a released quantization); passes every other gate"
                if gates.get("own_compression") is False and not other else "not passed: " + ", ".join(r.get("failed_gates") or []))
        label = f"Released: {r['release']}"
        out.append((label, cell(r["ppl"], by, r["decode_latency_seconds"], False, r["record"], note)))
    return out


def fmt(c: dict) -> str:
    acc = "–" if c["accuracy_pct"] is None else f"{c['accuracy_pct']:.2f}"
    by = " / ".join("–" if c["correct"][k] is None else str(c["correct"][k]) for k in BENCH)
    tot = "–" if c["correct_total"] is None else f"{c['correct_total']}/150"
    ppl = "–" if c["ppl"] is None else f"{c['ppl']:.3f}"
    L = "–" if c["decode_seconds"] is None else f"{c['decode_seconds']:.3f}"
    tps = "–" if c["decode_tps"] is None else f"{c['decode_tps']:.1f}"
    return f"{ppl} | {acc} | {tot} ({by}) | {L} | {tps} | {'yes' if c['passed'] else 'no'}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA, help="the analysis directory (selected_runs.json in, output out)")
    parser.add_argument("--baselines", type=Path, default=BASELINES, help="the baseline records' root")
    args = parser.parse_args()
    agents = agent_cells(args.data)
    order = {model: i for i, model in enumerate(AGENT_KEY)}
    names = {model: AGENT_NAME[key] for model, key in AGENT_KEY.items()}
    lines = ["# Results, cell by cell", "",
             "Every value from its recorded hidden verdict (`compression_bench/scripts/verified_results_table.py`). **Acc.** "
             "is the hidden correctness gate's multi-benchmark accuracy, the mean accuracy on 50 HumanEval (code), 50 "
             "TriviaQA (knowledge) and 50 GSM8K (math) items; with 50 items each it is the total correct out of 150. "
             f"**L** is the decode time (s); **TPS** is {INTERVALS}/L, computed before rounding. The gate compares against "
             "the task's frozen BF16 reference. A not-passed run's measurements are shown as measured.", ""]
    data = {}
    for task in sorted(agents):
        rows = bf16_cells(task, args.baselines) + gptq_cells(task, args.baselines)
        runs = sorted(agents[task], key=lambda mc: (order.get(mc[0], len(order)), mc[0], mc[1]["run"]))
        rows += [(f"{names.get(model, model)} ({c['run']})" if sum(m == model for m, _ in runs) > 1 else names.get(model, model), c)
                 for model, c in runs]
        rows += released_cells(task, args.baselines)
        data[task] = {name: c for name, c in rows}
        lines += [f"## {label(task)}", "", "| Method / agent | PPL | Acc. (%) | correct (code / knowledge / math) | L (s) | TPS | passed | note |",
                  "|---|--:|--:|---|--:|--:|---|---|"]
        lines += [f"| {name} | {fmt(c)} | {c['note']} |" for name, c in rows]
        lines.append("")
    (args.data / "verified_results.md").write_text("\n".join(lines))
    (args.data / "verified_results.json").write_text(json.dumps(data, indent=1) + "\n")
    print(f"{sum(len(v) for v in data.values())} cells -> {args.data / 'verified_results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
