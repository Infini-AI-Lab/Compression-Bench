#!/usr/bin/env python3
"""Within-run design changes and what reward feedback measured for each.

    python3 compression_bench/scripts/design_pairs.py --pairs PAIRS.json [--data DIR]      # writes DIR/design_pairs.json

Each pair is one change an agent made between two candidates of the same selected run, both measured by the public
Checker's full check (reward feedback: PPL on the task's public BF16-continued blocks, the same tokens for every
candidate of a task). The change is the one the agent's version notes (`version_ledger.json`) or trajectory name;
both rewards are looked up in the run's recorded checks (`checks`, matched to 1e-4), which give the measured package
size and public PPL. A pair is a within-run comparison under the same size limit, not a controlled experiment:
`same_measured_size` is true only when the two packages differ by under 0.01 GiB, and `single_factor` is false where
the agent changed more than one thing at once. `decision` is what the agent did with the second candidate: kept (the
run continued from it), reverted (it went back to the first). Excess cross-entropy (excess CE) is ln(PPL / PPL_BF16),
nats per token; PPL above BF16 is 100 (PPL / PPL_BF16 - 1).

PAIRS.json is the reader's annotation of the runs: a list of objects
    {"factor": "rotation", "run": "<run id>", "change": "Hadamard rotation on the expert weights",
     "reward_before": 8.73, "reward_after": 8.83, "decision": "kept" | "reverted", "single_factor": true,
     "source": "v1 -> v2 notes", "label": "optional short label for a figure"}
with the two public rewards as the run's checks recorded them. DIR (default evaluation_progress/trajectory_analysis/)
must hold version_ledger.json (version_ledger.py).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_common import DATA  # noqa: E402

FIELDS = ("factor", "run", "change", "reward_before", "reward_after", "decision", "single_factor", "source")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", type=Path, required=True, help="the annotated design changes (format above)")
    parser.add_argument("--data", type=Path, default=DATA, help="the analysis directory (version_ledger.json in, output out)")
    args = parser.parse_args()
    specs = json.loads(args.pairs.read_text())
    specs = specs.get("pairs", specs) if isinstance(specs, dict) else specs
    ledger = json.loads((args.data / "version_ledger.json").read_text())
    runs = {r["run"]: r for r in ledger["runs"]}
    out = []
    for spec in specs:
        missing = [k for k in FIELDS if k not in spec]
        if missing or spec["decision"] not in ("kept", "reverted"):
            raise SystemExit(f"pair {spec}: missing {missing} or a decision other than kept/reverted")
        if spec["run"] not in runs:
            raise SystemExit(f"run {spec['run']} is not in version_ledger.json")
        run = runs[spec["run"]]

        def check(reward):
            hits = [c for c in run["checks"] if c.get("public_reward") is not None and abs(c["public_reward"] - reward) < 1e-4]
            if not hits:
                raise SystemExit(f"{run['run']}: no reward feedback at {reward}")
            c = hits[0]
            return {"utc": c["utc"], "public_reward": c["public_reward"], "public_ppl": c["public_ppl"],
                    "public_bf16_ppl": c["public_bf16_ppl"], "package_gib": c["total_gib"],
                    "ppl_above_bf16_pct": 100 * (c["public_ppl"] / c["public_bf16_ppl"] - 1),
                    "excess_ce": math.log(c["public_ppl"] / c["public_bf16_ppl"])}
        b, a = check(spec["reward_before"]), check(spec["reward_after"])
        out.append({"factor": spec["factor"], "agent": run["agent"], "model": run["model"], "cap": run["cap"],
                    "task": run["task"], "run": run["run"], "change": spec["change"], "label": spec.get("label"),
                    "before": b, "after": a, "decision": spec["decision"], "single_factor": bool(spec["single_factor"]),
                    "same_measured_size": abs(a["package_gib"] - b["package_gib"]) < 0.01,
                    "excess_ce_change_pct": 100 * (a["excess_ce"] / b["excess_ce"] - 1), "source": spec["source"]})
    doc = {"about": __doc__.split("\n\n")[1].strip().replace("\n", " "), "stage": "reward feedback", "pairs": out}
    (args.data / "design_pairs.json").write_text(json.dumps(doc, indent=1) + "\n")
    by = {}
    for p in out:
        by.setdefault(p["factor"], []).append(p["excess_ce_change_pct"] < 0)
    print(len(out), "pairs;", {k: f"{sum(v)}/{len(v)} lower" for k, v in by.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
