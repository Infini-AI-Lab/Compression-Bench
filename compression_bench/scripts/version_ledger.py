#!/usr/bin/env python3
"""Every version of every selected run with what each measurement stage recorded about it, stage by stage.

    python3 compression_bench/scripts/version_ledger.py [--data DIR]      # writes DIR/version_ledger.json

Runs: the selected runs of `DIR/selected_runs.json` (trajectory_features.py), nothing else; needs DIR/gate_accuracy.json
(gate_accuracy.py). DIR defaults to evaluation_progress/trajectory_analysis/. Per version (a `./version` snapshot of the
run, `METRICS.json: versions`):

  reward_feedback the public Checker's full check the agent ran for that content (PPL on the public BF16-continued
                 blocks, reward 10 * PPL_bf16 / PPL, package size): the feedback the agent saw during design. Matched
                 to the version by the version's own `public_check` record; `same_bytes` says whether the checked
                 submission had the version's byte count.
  probe          the harness's measurement of the snapshot during the run, never shown to the agent (the first 32
                 public PPL blocks; 8 items of each hidden benchmark): a post-capture diagnostic, not a verdict.
  verdict        the full hidden final verification of that version (`checkpoint_verdicts.jsonl`, METRICS.json),
                 with per-benchmark correct counts (gate_accuracy.json): GSM8K = math, HumanEval = code, TriviaQA =
                 knowledge, 50 items each.

Each run also lists every reward-feedback call in time order (`checks`), so checks of content the agent never snapshotted
are kept. The stages are never merged: a version can carry all three, one, or none. No hidden item, prompt or
generation is written.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_common import DATA, PROTOTYPES, lineage_checkpoint_verdicts, selected_runs  # noqa: E402


def check_row(c: dict) -> dict:
    keep = ("utc", "seconds", "action", "valid", "error", "submission_bytes", "total_gib", "public_ppl", "public_bf16_ppl",
            "public_reward", "ppl_increase_percent", "decode_latency_seconds", "limit_gib")
    return {k: c.get(k) for k in keep if c.get(k) is not None}


def probe_row(path: Path) -> dict | None:
    if not path.is_file():
        return None
    p = json.loads(path.read_text())
    out = {"status": p.get("status"), "utc": p.get("probed_utc")}
    if p.get("ppl"):
        out.update(blocks=p["ppl"].get("blocks"), ppl=p["ppl"].get("ppl"), bf16_ppl=p["ppl"].get("bf16_ppl_same_blocks"))
    if p.get("accuracy"):
        out["accuracy"] = {b: {k: v.get(k) for k in ("correct", "items", "bf16_correct")} for b, v in p["accuracy"].items()
                           if isinstance(v, dict)}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA, help="the analysis directory (inputs and output)")
    args = parser.parse_args()
    gate_accuracy = json.loads((args.data / "gate_accuracy.json").read_text())
    gate = {r["run"]: r for r in gate_accuracy["runs"]}
    bf16 = gate_accuracy["bf16"]
    runs = []
    for sel in selected_runs(args.data):
        run_dir = Path(sel["run_dir"])
        metrics = json.loads((run_dir / "METRICS.json").read_text())
        lineage = PROTOTYPES / metrics["run"]["lineage_id"]
        verdicts = {c["version"]: c for c in metrics["verdicts"].get("checkpoints") or [] if c.get("measured")}
        verdicts.update(lineage_checkpoint_verdicts(lineage, sel["run"]))
        counts = {c["version"]: c.get("accuracy") for c in gate[sel["run"]]["checkpoints"]}
        checks = [check_row(c) for c in metrics["public_checks"]["checks"]]
        versions = []
        for v in metrics["versions"]:
            pc = v.get("public_check")
            match = next((c for c in checks if pc and c.get("utc") == pc.get("utc")), None)
            row = {"version": v["version"], "captured_utc": v.get("captured_utc"), "note": v.get("note"),
                   "total_bytes": v.get("total_bytes"), "weight_bytes": v.get("weight_bytes"),
                   "manifest_identity": v.get("manifest_identity"),
                   "reward_feedback": None if not pc else {**{k: pc.get(k) for k in ("utc", "public_ppl", "public_bf16_ppl",
                                                                                  "public_reward", "ppl_increase_percent",
                                                                                  "total_gib", "decode_latency_seconds")},
                                                       "same_bytes": bool(match and match.get("submission_bytes") == v.get("total_bytes"))},
                   "probe": probe_row(lineage / "CHECKPOINT_NOTES" / f"{v['version']}.probe.json"),
                   "verdict": None}
            if v["version"] in verdicts:
                c = verdicts[v["version"]]
                row["verdict"] = {k: c.get(k) for k in ("status", "same_as_final", "ppl", "accuracy", "accuracy_retention",
                                                        "accuracy_gate_pass", "package_gib", "resident_gib",
                                                        "decode_latency_seconds", "failed_stage", "infra_error")}
                row["verdict"]["counts"] = (counts.get(v["version"]) or {}).get("correct")
            versions.append(row)
        final = metrics["verdict"]
        runs.append({"task": sel["task"], "model": sel["model"], "cap": sel["cap"], "agent": sel["agent"], "run": sel["run"],
                     "lineage": metrics["run"]["lineage_id"],
                     "agent_window_utc": [metrics["timeline"].get("agent_started_utc"), metrics["timeline"].get("agent_ended_utc")],
                     "final": {"status": final.get("status"), "ppl": final.get("ppl"), "bf16_ppl": final.get("bf16_ppl"),
                               "counts": (gate[sel["run"]].get("final") or {}).get("correct"),
                               "failed_stage": gate[sel["run"]].get("failed_stage")},
                     "versions": versions, "checks": checks})
    out = {"about": __doc__.split("\n\n")[0].strip(), "bf16_hidden_counts": {m: b["correct"] for m, b in bf16.items()},
           "runs": runs}
    (args.data / "version_ledger.json").write_text(json.dumps(out, indent=1) + "\n")
    n = sum(len(r["versions"]) for r in runs)
    print(f"{len(runs)} runs, {n} versions: reward feedback {sum(bool(v['reward_feedback']) for r in runs for v in r['versions'])}, "
          f"probe {sum(bool(v['probe']) for r in runs for v in r['versions'])}, "
          f"verdict {sum(bool(v['verdict']) for r in runs for v in r['versions'])}; "
          f"{sum(len(r['checks']) for r in runs)} checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
