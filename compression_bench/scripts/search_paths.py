#!/usr/bin/env python3
"""Each run's public full checks as (position in the agent budget, package size / cap, public reward).

    python3 compression_bench/scripts/search_paths.py [--out evaluation_progress/trajectory_analysis] [--runs RUN,RUN]

Reads `public-checks/checks.jsonl` of the run directory: every check that was valid and returned a public reward.
Without `--runs`, every run of OUT/features.json (trajectory_features.py) is written into OUT/search_paths.json; with
it, only those runs are (re)written and the other entries stay.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_common import DATA, run_dir, utc_seconds  # noqa: E402


def path_of(run: str) -> list | None:
    root = run_dir(run)
    if root is None:
        return None
    ident = json.loads((root / "identity.json").read_text())
    start, span = utc_seconds(ident["created_at"]), float(ident["budget_hours"]) * 3600
    out = []
    for line in (root / "public-checks" / "checks.jsonl").read_text().splitlines():
        c = json.loads(line)
        r = c.get("response") or {}
        if r.get("valid") and r.get("public_reward") is not None and r.get("limit_gib"):
            out.append([(utc_seconds(c["utc_start"]) - start) / span, r["total_gib"] / r["limit_gib"], r["public_reward"]])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(DATA))
    parser.add_argument("--runs", help="comma-separated run ids (default: every run of features.json)")
    args = parser.parse_args()
    target = Path(args.out) / "search_paths.json"
    data = json.loads(target.read_text()) if target.is_file() else {}
    runs = (args.runs.split(",") if args.runs
            else [r["run"] for r in json.loads((Path(args.out) / "features.json").read_text())])
    for run in runs:
        got = path_of(run)
        if got is not None:
            data[run] = got
            print(run, len(got), "checks")
    target.write_text(json.dumps(data, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
