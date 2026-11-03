#!/usr/bin/env python3
"""Package the finished repeatability trials and memory profiles into portable evidence; never runs an agent or model.

    finalize_studies.py [--decision-events FILE --event ID ...]

Requires every planned trial to have finished (evidence_studies.py, study_status.py). Writes, to the evidence study's
output directory: repeat_verification_evidence.json (aggregate verification per original and repeat run, no hidden
items or generations), original_storage.json (the original runs' official package and resident-storage census),
complete_results.csv, final_summary.json, README.md and evidence_manifest.json. With --decision-events (the
decision_events.json of agent_decisions.py) the named events' recorded alternatives, local validation and reward
feedback are copied to recorded_alternatives.json as observational context.
"""
import argparse
import csv
import json
from pathlib import Path

from common import file_sha256, read_json, utc_now, write_json
import evidence_studies as study
import study_status


def select(d, keys):
    return {k: d[k] for k in keys if k in d}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--decision-events")
    p.add_argument("--event", action="append", default=[])
    a = p.parse_args()
    out = study.OUT
    assert study_status.refresh(), "wait for every planned trial to end; never launch replacements"
    state = read_json(out / "repeat_state.json")
    trials = len(state["jobs"])
    assert all(j["status"] == "finished" and j["returncode"] == 0 for j in state["jobs"])
    rows = read_json(out / "repeat_results.json")["rows"]
    assert sum(r["kind"] == "repeat" for r in rows) == trials and len(rows) == 2 * trials
    costs = {r["run_id"]: r[study_status.COST] for r in csv.DictReader((out / "provider_costs.csv").open())}
    verifications, table, original_storage = [], [], []
    for row in rows:
        src = Path(row["source"])
        m = read_json(src)
        d = m["verdicts"]["final"]["detail"]
        decode = select(d["decode"], ["valid", "complete", "input_tokens", "output_tokens", "concurrency", "measured_requests",
                                      "engine_startup_seconds", "warmup_seconds", "median_round_mean_seconds", "round_mean_seconds",
                                      "round_cv", "round_spread_fraction", "requests"])
        census = select(d["census"], ["max_resident_bytes", "limit_gib", "passed", "storage_count", "worker_bytes", "bytes_by_dtype"])
        bench = d.get("accuracy", {}).get("benchmarks", {})
        cost = float(costs[row["run_id"]]) if costs.get(row["run_id"]) else None
        verifications.append({
            "trial": row["kind"], "agent": row["agent"], "task": row["task"], "run_id": row["run_id"],
            "metrics_source": str(src), "metrics_sha256": file_sha256(src), "status": row["status"],
            "package": d.get("package"), "census": census,
            "ppl": select(d.get("ppl", {}), ["valid", "ppl", "bf16_ppl", "predicted_tokens", "blocks", "ppl_split", "ppl_protocol"]),
            "accuracy": {"items": d.get("accuracy", {}).get("items"), "mean_accuracy": d.get("accuracy", {}).get("accuracy"), "benchmarks": bench},
            "decode": decode,
            "score": select(d.get("score", {}), ["valid", "scoring_version", "capacity_protocol", "ppl_protocol", "ppl_split",
                                                  "accuracy_protocol", "accuracy_split", "accuracy_gate_pass", "accuracy_gate_floor",
                                                  "accuracy_retention", "reward", "ppl_fidelity"]),
            "timing_source": str(src.parent / "TIMING.json"), "timing_sha256": file_sha256(src.parent / "TIMING.json"),
            "usage": m["agent"]["usage"], study_status.COST: cost,
            "credential_values_recorded": False, "hidden_prompts_predictions_and_answers_recorded": False})
        flat = dict(row)
        for k in ("code", "knowledge", "math"):
            flat[k + "_accuracy_percent"] = 100 * bench[k]["accuracy"] if k in bench else None
        flat[study_status.COST] = cost
        table.append(flat)
        if row["kind"] == "original":
            original_storage.append({"agent": row["agent"], "task": row["task"], "run_id": row["run_id"],
                                     "official_resident_storage": census, "package": d.get("package"),
                                     "note": "The official resident-storage census is not a peak-device-memory measurement."})
    write_json(out / "repeat_verification_evidence.json", {"scope": "Post-design aggregate verification only; no hidden items or generations.", "runs": verifications})
    write_json(out / "original_storage.json", {"scope": "Official storage measurements of the original runs; not loading/prefill/decode peak profiles.", "runs": original_storage})
    with (out / "complete_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(table)
    if a.decision_events:
        events = {e["event_id"]: e for e in json.loads(Path(a.decision_events).read_text())}
        write_json(out / "recorded_alternatives.json", {
            "scope": "Recorded local/public feedback for the named alternatives; observational, not a controlled verification.",
            "source": a.decision_events,
            "observations": [{k: events[i].get(k) for k in ("event_id", "run", "agent", "task", "alternatives", "local_validation", "reward_feedback")}
                             for i in a.event]})
    variation = read_json(out / "repeat_variation.json")
    consistency = {metric: {"consistent": sum(r["ranking_consistent"] for r in variation["rankings"] if r["metric"] == metric),
                            "tasks": sum(r["metric"] == metric for r in variation["rankings"])}
                   for metric in ("verification_ppl", "mean_accuracy", "tps")}
    memory = read_json(out / "memory_results.json") if (out / "memory_results.json").exists() else {"profiles": []}
    preservation = study.protected() == study.plan()["protected_sha256"]
    assert preservation
    summary = {"finalized_utc": utc_now(), "repeatability": {
        "trials": trials, "original_passed": sum(r["passed"] for r in rows if r["kind"] == "original"),
        "repeat_passed": sum(r["passed"] for r in rows if r["kind"] == "repeat"),
        "pass_agreement": sum(c["pass_agreement"] for c in variation["comparisons"]),
        "ranking_consistency": consistency,
        "repeat_estimated_cost_usd": sum(r[study_status.COST] or 0 for r in table if r["kind"] == "repeat"),
        "limitation": "One repeat per condition; descriptive paired variation, not population reliability or causal attribution."},
        "memory": {"measured_profiles": sum(p.get("status") == "measured" for p in memory["profiles"])},
        "protected_files_unchanged": preservation}
    write_json(out / "final_summary.json", summary)
    write_readme(out, table)
    # Integrity manifest covers small portable evidence; full nsys reports stay in the work directory.
    files = [p for p in out.rglob("*") if p.is_file() and p.name != "evidence_manifest.json"]
    write_json(out / "evidence_manifest.json", {"files": {str(p.relative_to(out)): file_sha256(p) for p in sorted(files)},
                                                "protected_files_unchanged": preservation})
    print(json.dumps(summary, indent=2))


def write_readme(out, rows):
    pct = lambda v: "n/a" if v is None else f"{v:.0f}"
    usd = lambda v: "n/a" if v is None else f"{v:.3f}"
    text = '''# Repeatability and memory evidence

Original and repeated outcomes of each planned trial. Accuracy columns are percentages on 50 HumanEval / 50 TriviaQA /
50 GSM8K items; PPL and all accuracies are post-design verification. TPS is the 127-token-interval statistic. API
dollars are provider-rate estimates from token counts (`provider_pricing.json`), not invoices.

| Task | Agent | Trial | PPL | HumanEval (%) | TriviaQA (%) | GSM8K (%) | Mean accuracy (%) | Pass | TPS | API $ estimate |
|---|---|---|--:|--:|--:|--:|--:|---|--:|--:|
'''
    for r in rows:
        ppl = "n/a" if r["verification_ppl"] is None else f"{r['verification_ppl']:.5f}"
        mean = "n/a" if r["mean_accuracy"] is None else f"{100*r['mean_accuracy']:.2f}"
        tps = "n/a" if r["tps"] is None else f"{r['tps']:.2f}"
        text += (f"| {r['task']} | {r['agent']} | {r['kind']} | {ppl} | {pct(r['code_accuracy_percent'])} | "
                 f"{pct(r['knowledge_accuracy_percent'])} | {pct(r['math_accuracy_percent'])} | {mean} | {r['passed']} | "
                 f"{tps} | {usd(r[study_status.COST])} |\n")
    text += '''
## Files

- `complete_results.csv`: every original/repeat outcome with per-benchmark accuracy and cost estimates.
- `repeat_verification_evidence.json`: aggregate verification, nine-request timing measurements, official storage
  measurements and source digests. Hidden prompts, predictions and answers are excluded.
- `repeat_variation.json`: paired changes, ranges, sample SD and ranking agreement; `plan.json` the frozen launch controls.
- `memory/`, `memory_results.json`, `peak_memory.csv`: three repetitions per profiled artifact with separate loading,
  prefill and decode phases, fixed cache capacity, allocation lifetimes and sampled device memory.
- `original_storage.json`: the original runs' official package/resident measurements.
- `final_summary.json`, `STATUS.md`; `evidence_manifest.json` records hashes of the portable evidence.

Rebuild with `python compression_bench/scripts/studies/finalize_studies.py`; it reads existing results and launches nothing.
'''
    (out / "README.md").write_text(text)


if __name__ == "__main__":
    main()
