#!/usr/bin/env python3
"""Write one methods report from the lineages' own evidence.

    python3 compression_bench/scripts/methods_report.py [--since YYYYMMDDTHHMM] [--runs FILE] [--out FILE]

For every lineage under `compression_output/` that has a final verdict (`FINAL.json`), created at or after
`--since` and, with `--runs` (run ids, as trajectory_features.py reads them), judged on one of those runs, the report
gives the
verdict (reward, perplexity, accuracy retention against the correctness gate, package size) and the method as the
agent recorded it: the first version's note (the base method), the judged version's note, the opening of the
README the agent shipped with the judged version, and the path of that version's code. The method text is the
agent's own claim, quoted; nothing here is checked against the code. The weights are not in the repository, the
code and notes are.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_common import AGENT_NAME, DATA, PROTOTYPES, REPO, agent_key, read_selection, short_task  # noqa: E402


def lines(path: Path) -> list[dict]:
    out = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def readme_opening(version_dir: Path, limit: int = 900) -> str:
    for name in ("README.md", "readme.md", "NOTES.md"):
        path = version_dir / "files" / name
        if path.is_file():
            text = re.sub(r"\n{3,}", "\n\n", path.read_text(encoding="utf-8", errors="replace")).strip()
            return text[:limit] + (" …" if len(text) > limit else "")
    return ""


def quote(text: str) -> str:
    return "\n".join("> " + part for part in (text or "(none recorded)").splitlines())


def section(lineage: Path) -> tuple[str, str, str] | None:
    final = json.loads((lineage / "FINAL.json").read_text()) if (lineage / "FINAL.json").is_file() else None
    meta = json.loads((lineage / "LINEAGE.json").read_text()) if (lineage / "LINEAGE.json").is_file() else {}
    if not final or not meta:
        return None
    versions = lines(lineage / "versions.jsonl")
    notes = {v["version"]: v.get("note") or "" for v in versions if v.get("source") == "agent"}
    judged = final.get("matches_version") or (versions[-1]["version"] if versions else "")
    base = notes.get(next(iter(notes), ""), "")
    judged_note = notes.get(judged, "") or (list(notes.values())[-1] if notes else "")
    version_dir = next(iter(sorted(lineage.glob(f"*_{judged}"))), None) if judged else None
    model = meta.get("agent_model") or "?"
    agent = AGENT_NAME.get(agent_key(model), model)
    task = short_task(meta.get("task_id", "?"))
    status = final.get("status")
    retention = final.get("accuracy_retention")
    head = f"### {task} - {agent} - " + (f"reward {final['reward']:.2f}" if status == "SCORED" and final.get("reward") is not None else f"ZERO ({final.get('failed_stage') or 'gate'})")
    facts = [f"run `{final.get('run_id')}`", f"PPL {final['ppl']:.3f}" + (f" vs BF16 {final['bf16_ppl']:.3f}" if final.get("bf16_ppl") else "") if final.get("ppl") else "no PPL",
             f"accuracy {final['accuracy']:.3f}" + (f" ({retention:.0%} of BF16" + (", gate " + ("passed" if final.get("accuracy_gate_pass") else "FAILED") + ")") if retention is not None else "")
             if final.get("accuracy") is not None else "no accuracy",
             f"package {final['package_gib']:.2f} GiB" if final.get("package_gib") else "no package size",
             f"scoring {final.get('scoring_version') or 'n/a'}", f"judged version {judged}"]
    body = [head, "", "; ".join(facts) + ".", "", "Base method (first version's note):", quote(base), "",
            f"Judged version {judged}'s note:", quote(judged_note)]
    opening = readme_opening(version_dir) if version_dir else ""
    if opening:
        body += ["", "Opening of the README the agent shipped:", quote(opening)]
    if version_dir:
        body += ["", f"Code and notes: `{version_dir.relative_to(REPO)}/files/` (weights are not in the repository)."]
    return meta.get("created_utc", ""), task, "\n".join(body) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", default="", help="only lineages created on or after this time (YYYYMMDDTHHMM)")
    parser.add_argument("--runs", type=Path, help="only lineages whose final verdict is of one of these runs")
    parser.add_argument("--out", default=str(DATA / f"methods_{time.strftime('%Y%m%d')}.md"))
    args = parser.parse_args()
    wanted = read_selection(args.runs) if args.runs else None
    rows = []
    for lineage in sorted(p for p in PROTOTYPES.iterdir() if p.is_dir() and p.name[:1].isdigit()):
        if lineage.name[:13] < args.since:
            continue
        if wanted is not None:
            final = lineage / "FINAL.json"
            if not final.is_file() or json.loads(final.read_text()).get("run_id") not in wanted:
                continue
        got = section(lineage)
        if got:
            rows.append(got)
    rows.sort(key=lambda r: (r[1], r[0]))
    text = ["# Methods of the finished runs", "",
            "Generated by `compression_bench/scripts/methods_report.py` from each lineage's `FINAL.json`, `versions.jsonl` and the "
            "judged version's shipped files. The method text is the agent's own note, quoted; it is not verified against the "
            "code. Checkpoints are not stored; the code and notes are under `compression_output/`.", "",
            f"{len(rows)} runs with a final verdict, grouped by task. A run that was killed or cancelled before its verdict has no entry "
            "(its lineage still holds its version notes).", ""]
    task = None
    for _created, name, body in rows:
        if name != task:
            text += [f"## {name}", ""]
            task = name
        text += [body]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(text), encoding="utf-8")
    print(f"{len(rows)} runs -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
