#!/usr/bin/env python3
"""Refresh the repeatability and memory evidence and its status page (STATUS.md in the study's output directory).

    study_status.py [--watch]

Reads evidence_studies.py's records and memory_study.py's profiles. API cost is re-estimated from token counts when
`provider_pricing.json` is in the output directory: {"rates": {"<agent>": {"input": $, "cached_input": $,
"output": $}}} in USD per million tokens (cache-creation tokens are priced as input). `--watch` refreshes every
minute until every trial has ended. Never launches an agent, a verifier or a model.
"""
import argparse
import csv
import statistics
import time

from common import read_json, utc_now, write_json
import evidence_studies as study

COST = "provider_estimated_cost_usd"


def memory_rows():
    rows = []
    for p in sorted((study.OUT / "memory").glob("*/*/rep*/profile_summary.json")):
        d = read_json(p)
        if d["status"] != "measured":
            continue
        native = d["native_cache_and_compute_buffers"]
        for phase in d["measurement"]["phases"]:
            workers = phase["workers"]
            row = {"task": d["task"], "artifact": d["artifact"], "repetition": d["repetition"], "phase": phase["phase"],
                   "resident_model_bytes": max(w["resident_model"]["bytes"] for w in workers),
                   "kv_cache_bytes": native.get("CUDA0_KV_bytes"), "recurrent_state_bytes": native.get("CUDA0_RS_bytes"),
                   "runtime_compute_buffer_bytes": native.get("CUDA0_compute_bytes"),
                   "sampled_peak_device_bytes": phase["sampled_peak_device_bytes"],
                   "cuda_traced_peak_live_bytes": d["cuda_trace"]["phases"][phase["phase"]]["cuda_traced_peak_live_bytes"],
                   "torch_peak_allocated_bytes": max(w["torch_peak_allocated_bytes"] for w in workers),
                   "torch_peak_reserved_bytes": max(w["torch_peak_reserved_bytes"] for w in workers), "source": str(p)}
            rows.append(row)
    if rows:
        with (study.OUT / "peak_memory.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    return rows


def fmt(x, digits=3):
    return "pending" if x is None else f"{x:.{digits}f}"


def gib(x):
    return "n/a" if x is None else f"{x/2**30:.3f}"


def costs(rows):
    pricing_file = study.OUT / "provider_pricing.json"
    pricing = read_json(pricing_file) if pricing_file.exists() else None
    for row in rows:
        rate = pricing and pricing["rates"].get(row["agent"])
        row[COST] = None if not rate else (
            ((row.get("input_tokens") or 0) + (row.get("cache_creation_input_tokens") or 0)) * rate["input"]
            + (row.get("cache_read_input_tokens") or 0) * rate["cached_input"]
            + (row.get("output_tokens") or 0) * rate["output"]) / 1_000_000
    with (study.OUT / "provider_costs.csv").open("w", newline="") as f:
        keys = ("kind", "agent", "task", "run_id", COST, "api_cost_usd", "api_cost_basis", "cli_reported_cost_usd",
                "input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return pricing is not None


def variation(rows):
    comparisons, rankings = [], []
    for old in [r for r in rows if r["kind"] == "original"]:
        new = next((r for r in rows if r["kind"] == "repeat" and (r["task"], r["agent"]) == (old["task"], old["agent"])), None)
        if not new:
            continue
        pair = {"task": old["task"], "agent": old["agent"], "original_run": old["run_id"], "repeat_run": new["run_id"], "pass_agreement": old["passed"] == new["passed"]}
        for metric in ("verification_ppl", "mean_accuracy", "tps", COST):
            values = [old[metric], new[metric]]
            pair[metric] = None if any(v is None for v in values) else {
                "original": values[0], "repeat": values[1], "delta": values[1] - values[0],
                "min": min(values), "max": max(values), "mean": statistics.mean(values), "sample_sd_n2": statistics.stdev(values)}
        comparisons.append(pair)
    for task in sorted({r["task"] for r in rows}):
        for metric, higher in (("verification_ppl", False), ("mean_accuracy", True), ("tps", True)):
            original = [r for r in rows if r["task"] == task and r["kind"] == "original" and r[metric] is not None]
            repeat = [r for r in rows if r["task"] == task and r["kind"] == "repeat" and r[metric] is not None]
            if len(original) == len(repeat) >= 2:
                order = lambda group: [r["agent"] for r in sorted(group, key=lambda r: r[metric], reverse=higher)]
                tie = lambda group: len({r[metric] for r in group}) < len(group)
                rankings.append({"task": task, "metric": metric, "original": order(original), "repeat": order(repeat),
                                 "ranking_consistent": order(original) == order(repeat),
                                 "original_tie": tie(original), "repeat_tie": tie(repeat)})
    write_json(study.OUT / "repeat_variation.json", {"updated_utc": utc_now(), "comparisons": comparisons, "rankings": rankings,
                "limitation": "one repeat per agent/task; n=2 per condition is descriptive, not an estimate of population reliability or a significance test"})
    return comparisons, rankings


def refresh():
    study.collect()
    state = read_json(study.OUT / "repeat_state.json")
    rows = read_json(study.OUT / "repeat_results.json")["rows"]
    priced = costs(rows)
    mem = memory_rows()
    comparisons, rankings = variation(rows)
    unchanged = study.protected() == study.plan()["protected_sha256"]
    write_json(study.OUT / "study_status.json", {"updated_utc": utc_now(), "repeat_jobs": state["jobs"],
                "repeat_results_recorded": sum(r["kind"] == "repeat" for r in rows),
                "memory_profiles_measured": len({(r["task"], r["artifact"], r["repetition"]) for r in mem}),
                "protected_files_unchanged": unchanged})
    lines = ["# Repeatability and memory evidence", "",
             f"Updated {utc_now()}. Hash-guarded files unchanged: {unchanged}.", "",
             "## Repeatability", "",
             "Quality numbers are post-design verification, distinct from the design-time public reward feedback kept "
             "in each run. Accuracy columns are correct/50 for HumanEval / TriviaQA / GSM8K; mean accuracy averages the "
             "three; TPS = 127/L." + (" API $ is a provider-rate estimate from `provider_pricing.json` and token counts, "
                                      "not an invoice." if priced else ""), "",
             "| Task | Agent | Trial | Verification PPL | HumanEval | TriviaQA | GSM8K | Mean accuracy (%) | Status | TPS | API $ estimate |",
             "|---|---|---|--:|--:|--:|--:|--:|---|--:|--:|"]
    for agent, task, _ in study.originals():
        for kind in ("original", "repeat"):
            r = next((r for r in rows if (r["agent"], r["task"], r["kind"]) == (agent, task, kind)), None)
            if r:
                counts = ["unmeasured" if r[k] is None else f"{r[k]}/50" for k in ("code_correct", "knowledge_correct", "math_correct")]
                lines.append(f"| {task} | {agent} | {kind} | {fmt(r['verification_ppl'])} | {' | '.join(counts)} | {fmt(100*r['mean_accuracy'] if r['mean_accuracy'] is not None else None,2)} | {r['status']} | {fmt(r['tps'],1)} | {fmt(r[COST])} |")
            else:
                job = next(j for j in state["jobs"] if j["id"] == f"{agent}-{task}")
                lines.append(f"| {task} | {agent} | repeat | pending | pending | pending | pending | pending | {job['status']} | pending | pending |")
    lines += ["", f"Completed repeat-result pairs: {len(comparisons)}/{len(state['jobs'])}. "
              "`repeat_variation.json` records paired deltas, ranges and sample SD at n=2, and per-task PPL, accuracy "
              "and TPS ranking agreement when every repeat of a task is available."]
    if comparisons:
        lines += ["", "| Task | Agent | Repeat − original PPL | Accuracy change (percentage points) | TPS change | API $ estimate change | Same pass status |",
                  "|---|---|--:|--:|--:|--:|---|"]
        for pair in comparisons:
            delta = lambda key, factor=1: fmt(pair[key]["delta"]*factor) if pair[key] is not None else "unmeasured"
            lines.append(f"| {pair['task']} | {pair['agent']} | {delta('verification_ppl')} | {delta('mean_accuracy',100)} | {delta('tps')} | {delta(COST)} | {pair['pass_agreement']} |")
    if rankings:
        lines += ["", "| Task | Metric (best first) | Original ranking | Repeated ranking | Consistent |",
                  "|---|---|---|---|---|"]
        for row in rankings:
            same = "tie; not a strict ranking" if row["original_tie"] or row["repeat_tie"] else str(row["ranking_consistent"])
            lines.append(f"| {row['task']} | {row['metric']} | {' > '.join(row['original'])} | {' > '.join(row['repeat'])} | {same} |")
    lines += ["", "## Peak memory", "",
              "memory_study.py profiles each artifact in three independent cold processes: batch one, the task's first "
              "measured 2,048-token decode prompt, 128 output tokens with EOS ignored, a 4,096-token context and one "
              "sequence. Loading includes runtime initialization; prefill ends at the first generated token; decode "
              "covers the remaining 127 intervals. Nsight Systems traces CUDA allocation lifetimes under NVTX phase "
              "markers (driver/context overhead excluded); NVML samples total device-used memory every 10 ms (included, "
              "but sub-sample transients can be missed). Profiling overhead means these are not latency measurements, "
              "and peak device memory does not replace the resident-storage gate.", "",
              "| Artifact | Phase | Repetitions | Resident weights (GiB) | KV cache (GiB) | Recurrent state (GiB) | CUDA-traced peak live (GiB), min–max | Sampled device peak (GiB), min–max |",
              "|---|---|--:|--:|--:|--:|---|---|"]
    for artifact in sorted({r["artifact"] for r in mem}):
        for phase in ("loading", "prefill", "decode"):
            group = [r for r in mem if r["artifact"] == artifact and r["phase"] == phase]
            if not group:
                continue
            value = group[0]
            bounds = lambda k: f"{min(r[k] for r in group)/2**30:.3f}–{max(r[k] for r in group)/2**30:.3f}"
            lines.append(f"| {artifact} | {phase} | {len(group)} | {gib(value['resident_model_bytes'])} | {gib(value['kv_cache_bytes'])} | {gib(value['recurrent_state_bytes'])} | {bounds('cuda_traced_peak_live_bytes')} | {bounds('sampled_peak_device_bytes')} |")
    lines += ["", "Raw allocation events, 10 ms device samples, probe/launch hashes and per-repetition records are under "
              "`memory/`; full `.nsys-rep`/SQLite traces stay under the study work directory listed in each record."]
    (study.OUT / "STATUS.md").write_text("\n".join(lines) + "\n")
    return all(j["status"] not in ("queued", "running") for j in state["jobs"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--watch", action="store_true")
    a = p.parse_args()
    while not refresh() and a.watch:
        time.sleep(60)
