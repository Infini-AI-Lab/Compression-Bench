#!/usr/bin/env python3
"""How each agent used the offline compression-paper collection, from its trajectory.

    python3 compression_bench/scripts/paper_usage.py [--catalog catalog.json] [--out evaluation_progress/trajectory_analysis]

For every run in `features.json` (see trajectory_features.py) this reads the trajectory's tool calls, keeps those that
touch `/opt/compression-papers` (the collection's `search.py` and `read.py`, or a direct read of a paper's text), and
records, with the position in the run (0 to 1): each search query, each paper opened (`read.py ID`, with `--page`,
`--pages` or `--grep`, or a direct file read), and the papers the searches listed. The catalogue gives each paper's
title and categories: by default `paper_catalog.json` beside this script, a copy of the collection's own
(/opt/compression-papers/catalog.json in the agent image).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_common import DATA, steps as trajectory_steps, utc_seconds  # noqa: E402

ID = r"(?:\d{4}\.\d{4,5}|[a-z]+-[A-Za-z0-9_.-]+)"
READ = re.compile(r"read\.py\s+(" + ID + r")(?:\s+(--pages|--page\s+\d+|--grep\s+(?:'[^']*'|\"[^\"]*\"|\S+)|--pdf))?")
DIRECT = re.compile(r"/opt/compression-papers/papers/(" + ID + r")\.(?:txt|pdf)")
SEARCH = re.compile(r"search\.py\s+(?:--\w+\s+\S+\s+)*(\"[^\"]*\"|'[^']*')")
LISTED = re.compile(r"^\s*(\d{4}\.\d{4,5}|[a-z]+-[A-Za-z0-9_.-]+)\s+\(?\d{4}\)?", re.M)


def usage(run: str, catalog: dict) -> dict | None:
    steps = trajectory_steps(run)
    if not steps:
        return None
    first, last = utc_seconds(steps[0]["t"]), utc_seconds(steps[-1]["t"])
    span = max(last - first, 1.0)
    results = {s["id"]: s for s in steps if s["kind"] == "tool_result" and "id" in s}
    events, listed = [], set()
    for s in steps:
        if s["kind"] != "tool_call":
            continue
        raw = json.dumps(s.get("input"), ensure_ascii=False)
        if "compression-papers" not in raw:
            continue
        where = (utc_seconds(s["t"]) - first) / span
        text = (s.get("input") or {}).get("command", raw) if isinstance(s.get("input"), dict) else raw
        for q in SEARCH.findall(text):
            events.append({"at": where, "kind": "search", "query": q.strip("\"'")})
            out = str(results.get(s.get("id"), {}).get("output", ""))
            listed.update(LISTED.findall(out))
        for pid, mode in READ.findall(text):
            events.append({"at": where, "kind": "read", "paper": pid, "mode": (mode or "page 1").split()[0]})
        for pid in DIRECT.findall(text):
            events.append({"at": where, "kind": "read", "paper": pid, "mode": "direct"})
    read = [e["paper"] for e in events if e["kind"] == "read"]
    distinct = sorted(set(read))
    cats = {}
    for p in distinct:
        for c in (catalog.get(p, {}).get("categories") or ["(not in catalogue)"]):
            cats[c] = cats.get(c, 0) + 1
    return {"searches": sum(1 for e in events if e["kind"] == "search"), "reads": len(read), "distinct_read": len(distinct),
            "papers_read": distinct, "papers_listed": len(listed), "by_category": cats, "events": events}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default=str(Path(__file__).resolve().parent / "paper_catalog.json"))
    parser.add_argument("--out", default=str(DATA))
    args = parser.parse_args()
    catalog = {p["id"]: p for p in json.load(open(args.catalog))["papers"]}
    rows = json.load(open(Path(args.out) / "features.json"))
    out = {}
    for r in rows:
        got = usage(r["run"], catalog)
        if got:
            out[r["run"]] = got
    (Path(args.out) / "paper_usage.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(out)} runs -> {args.out}/paper_usage.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
