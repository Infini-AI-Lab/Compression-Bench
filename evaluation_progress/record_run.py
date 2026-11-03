#!/usr/bin/env python3
"""Record one finished run as one row of `evaluation_progress/<task>.csv`.

    evaluation_progress/record_run.py --run <run root>
    evaluation_progress/record_run.py --all          # every finished production run

The row is MLE-Bench's ServerBench shape (`rank, task, system, arm, model,
agent, effort, method, <task metrics>, gates_missed, verdict, evolve_hours,
started_utc, measured_utc, run_id, version, versions_published, backend, source,
evidence`) with compression's metrics and the agent's cost. Reads the run's
journal, its agent stream (token usage, priced by `lifecycle.pricing`) and the
lineage's `FINAL.json`; measures nothing. Idempotent (one row per run id,
replaced in place), atomic (a temporary file and a rename) and serialized with
an fcntl lock; `rank` is recomputed on every write.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import fcntl
import os
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "compression_bench" / "harness"))
from lifecycle.record import TERMINAL, Journal  # noqa: E402
from lifecycle.report import usage_totals  # noqa: E402
from lifecycle.utils import read_json  # noqa: E402

FIELDS = ("rank", "task", "system", "arm", "model", "agent", "effort", "method",
          "reward", "ppl", "bf16_ppl", "accuracy", "bf16_accuracy", "accuracy_retention",
          "package_gib", "resident_gib", "decode_latency_s", "gates_missed", "verdict",
          "scoring_version", "evolve_hours", "agent_hours", "started_utc", "measured_utc",
          "run_id", "version", "versions_published", "cost_usd", "cost_basis",
          "backend", "source", "evidence")
DIGITS = {"reward": 4, "ppl": 4, "bf16_ppl": 4, "accuracy": 4, "bf16_accuracy": 4,
          "accuracy_retention": 4, "package_gib": 3, "resident_gib": 3,
          "decode_latency_s": 3, "evolve_hours": 2, "agent_hours": 2, "cost_usd": 2}


def _utc(text: str | None):
    return datetime.datetime.strptime(text[:19], "%Y-%m-%dT%H:%M:%S") if text else None


def _fmt(key: str, value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.{DIGITS.get(key, 4)}f}"
    return str(value)


def gates_missed(final: dict) -> str:
    """Every gate the verdict failed, in words (empty for a scored run)."""
    missed = []
    if final.get("failed_stage"):
        missed.append(f"stage {final['failed_stage']} failed"
                      + (f": {str(final.get('error'))[:120]}" if final.get("error") else ""))
    if final.get("accuracy_gate_pass") is False:
        missed.append(f"correctness: kept {final.get('accuracy_retention', 0):.3f} of BF16's "
                      f"above-chance accuracy < {final.get('accuracy_gate_floor')}")
    return "; ".join(missed)


def method_of(events: list[dict], judged: str | None) -> str:
    """What the judged submission is, in the agent's own words: its version note."""
    notes = [(e.get("version"), e.get("note") or "") for e in events if e.get("event") == "version_captured"]
    # The judged version's note, or the latest before it that describes a method rather
    # than a documentation or clean-up change ("README wording only", "no weight change").
    upto = [n for v, n in notes[:[v for v, _ in notes].index(judged) + 1]] if judged in [v for v, _ in notes] \
        else [n for _, n in notes]
    plain = [n for n in upto if not re.search(r"\b(README|docstring|wording|documentation|no (code|weight)s? change|"
                                              r"unchanged|clean(ed|up)|pycache)\b", n, re.I)]
    note = (plain or upto or [""])[-1]
    return f"agent's note: {' '.join(str(note).split())[:300]}" if note else ""


def row_for(run_root: Path) -> dict:
    events = list(Journal(run_root))
    if not events:
        raise SystemExit(f"{run_root} has no journal")
    created = next((e for e in events if e.get("event") == "run_created"), {})
    launches = [e for e in events if e.get("event") == "agent_launched"]
    first = launches[0] if launches else {}
    ended = next((e for e in reversed(events) if e.get("event") in
                  ("agent_phase_ended", "cancelled")), None) \
        or next((e for e in reversed(events) if e.get("event") == "agent_finished"), {})
    states = [e.get("to") for e in events if e.get("event") == "state"]
    run_id = created.get("run_id")
    lineage_dir = Path(created.get("lineage_dir") or "")
    final = (read_json(lineage_dir / "FINAL.json", {}) or {}) if lineage_dir.name else {}
    if final.get("run_id") != run_id:
        final = {}
    model = first.get("agent_model") or created.get("agent_model") or ""
    effort = first.get("agent_effort") or created.get("agent_effort") or ""
    started, stopped = _utc(first.get("utc")), _utc(ended.get("utc"))
    usage = usage_totals(run_root / "agent.stream.jsonl")
    if final:
        verdict = final.get("status")
        source = "checkpoint" if any(e.get("event") == "final_verdict_from_checkpoint"
                                     for e in events) else "verdict"
    else:
        terminal = [s for s in states if s in TERMINAL]
        verdict = terminal[-1] if terminal else "NOT_RUN"
        source = "run_summary"
    return {
        "task": created.get("task_id"),
        "system": f"{model.rsplit('/', 1)[-1]} {effort}".strip(), "arm": "candidate",
        "model": model, "agent": first.get("agent_cli") or created.get("agent_cli"),
        "effort": effort, "method": method_of(events, final.get("matches_version")),
        "reward": final.get("reward"), "ppl": final.get("ppl"), "bf16_ppl": final.get("bf16_ppl"),
        "accuracy": final.get("accuracy"), "bf16_accuracy": final.get("bf16_accuracy"),
        "accuracy_retention": final.get("accuracy_retention"),
        "package_gib": final.get("package_gib"), "resident_gib": final.get("resident_gib"),
        "decode_latency_s": final.get("decode_latency_seconds"),
        "gates_missed": gates_missed(final) if final else "",
        "verdict": verdict, "scoring_version": final.get("scoring_version"),
        "evolve_hours": created.get("budget_hours"),
        "agent_hours": (stopped - started).total_seconds() / 3600 if started and stopped else None,
        "started_utc": first.get("utc"), "measured_utc": final.get("verified_utc"),
        "run_id": run_id,
        "version": final.get("matches_version"),
        "versions_published": sum(e.get("event") == "version_captured" for e in events),
        "cost_usd": usage.get("cost_usd"), "cost_basis": usage.get("cost_basis"),
        "backend": "container", "source": source, "evidence": final.get("evidence"),
    }


def ranked(rows: list[dict]) -> list[dict]:
    """Rank by reward within each scoring version (a scored, nonzero row only;
    ties share a rank); the newest version's rows first."""
    newest: dict[str, str] = {}
    for row in rows:
        version = row.get("scoring_version") or ""
        newest[version] = max(newest.get(version, ""), row.get("measured_utc") or "")
    for version in newest:
        scored = sorted((r for r in rows if (r.get("scoring_version") or "") == version
                         and r.get("verdict") == "SCORED" and r.get("reward")),
                        key=lambda r: -float(r["reward"]))
        previous, rank = None, 0
        for position, row in enumerate(scored, 1):
            if row["reward"] != previous:
                rank, previous = position, row["reward"]
            row["rank"] = str(rank)
    for row in rows:
        if row.get("verdict") != "SCORED" or not row.get("reward"):
            row["rank"] = ""
    order = sorted(newest, key=lambda v: newest[v], reverse=True)
    return sorted(rows, key=lambda r: (order.index(r.get("scoring_version") or ""),
                                       int(r["rank"]) if r.get("rank") else 10**6,
                                       r.get("started_utc") or ""))


def upsert(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # The directory itself is the lock: no lock file is left beside the CSVs.
    lock = os.open(csv_path.parent, os.O_RDONLY)
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        rows = []
        if csv_path.exists():
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = [r for r in csv.DictReader(handle) if r.get("run_id") != row["run_id"]]
        rows.append({k: _fmt(k, row.get(k)) for k in FIELDS})
        fd, temporary = tempfile.mkstemp(dir=csv_path.parent, prefix=f".{csv_path.name}.")
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            for r in ranked(rows):
                writer.writerow({k: r.get(k, "") for k in FIELDS})
        os.replace(temporary, csv_path)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        os.close(lock)


def finished_runs() -> list[Path]:
    """Every production run whose journal has reached a terminal state."""
    runs = []
    for journal in sorted((REPO / "compression_bench" / "evolve_space" / "evolve").glob("*/*/journal.jsonl")):
        states = [e.get("to") for e in Journal(journal.parent) if e.get("event") == "state"]
        if any(s in TERMINAL for s in states):
            runs.append(journal.parent)
    return runs


def record(run_root: Path, out_dir: Path) -> dict:
    row = row_for(run_root.resolve())
    if not row["task"] or not row["run_id"]:
        raise SystemExit(f"{run_root}: the journal names no task or run id")
    target = out_dir / f"{row['task']}.csv"
    upsert(target, row)
    print(f"recorded {row['run_id']} in {target} ({row['verdict']}, reward {row['reward']})")
    return row


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--run", type=Path)
    target.add_argument("--all", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args(argv)
    for run_root in ([args.run] if args.run else finished_runs()):
        record(run_root, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
