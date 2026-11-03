#!/usr/bin/env python3
"""Keep every checkpoint of a few chosen runs for a case study, past the janitor and the run's own cleanup.

    python3 compression_bench/scripts/case_study_keep.py [--list compression_bench/evolve_space/scheduler/case_studies.json] [--dry-run]

Each run named in the list (by run id) has every version snapshot of its lineage
(`compression_output/.objects/<lineage>/vN/`) hard-linked into
`compression_output/.case_studies/<lineage>/vN/`. A hard link costs nothing while the snapshot exists; when
`checkpoint_janitor` (a live run keeps only its newest snapshot) or the run's harvest deletes the snapshot, the
case-study copy keeps the data. The scheduler calls `keep` before it cleans, every housekeeping step, so a
snapshot is linked while it is still the newest. Weights only: the notes, manifests and verdicts already stay in
the lineage. A file is linked once; a run that has ended and been harvested needs nothing more.

A version whose snapshot the harness skipped (the disk was under its snapshot floor) has no snapshot to link. While it
is the newest version and the run's `submission/` still matches its MANIFEST.json file for file (size and mtime), the
submission is copied instead with `cp --reflink=always`: no space until the agent rewrites its files.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LIST = REPO / "compression_bench" / "evolve_space" / "scheduler" / "case_studies.json"
KEPT = REPO / "compression_output" / ".case_studies"


def lineage_of(run_dir: Path) -> Path | None:
    for line in (run_dir / "journal.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event.get("event") == "run_created" and event.get("lineage_dir"):
            return Path(event["lineage_dir"])
    return None


def keep(run_dir: Path, kept_root: Path = KEPT, dry_run: bool = False) -> list[Path]:
    """Hard-link every snapshot file of the run's lineage not yet linked; returns the new links."""
    lineage = lineage_of(run_dir)
    if lineage is None:
        return []
    objects = lineage.parent / ".objects" / lineage.name
    made = []
    for source in (sorted(objects.rglob("*")) if objects.is_dir() else []):
        if not source.is_file() or source.is_symlink():
            continue
        target = kept_root / lineage.name / source.relative_to(objects)
        if target.exists():
            continue
        made.append(target)
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, target)
            except FileNotFoundError:   # deleted between the listing and the link
                made.pop()
    return made


def keep_skipped(run_dir: Path, kept_root: Path = KEPT, dry_run: bool = False) -> list[Path]:
    """Reflink-copy the submission as the newest version when the harness skipped that version's snapshot."""
    lineage = lineage_of(run_dir)
    if lineage is None or not (lineage / "versions.jsonl").exists():
        return []
    lines = [json.loads(line) for line in (lineage / "versions.jsonl").read_text().splitlines() if line.strip()]
    agent = [v for v in lines if v.get("source") == "agent"]
    if not agent or agent[-1].get("snapshot") != "skipped":
        return []
    version = agent[-1]
    target = kept_root / lineage.name / version["version"]
    if target.exists():
        return []
    manifest = json.loads((lineage / version["name"] / "MANIFEST.json").read_text())
    submission = run_dir / "submission"
    for entry in manifest:
        path = submission / entry["path"]
        try:
            stat = path.stat()
        except OSError:
            return []
        if stat.st_size != entry["size"] or stat.st_mtime_ns != entry["mtime_ns"]:
            return []                   # the agent has changed it since the version: no faithful copy
    if dry_run:
        return [target]
    staging = target.with_name(target.name + ".partial")
    made = []
    for entry in manifest:
        dest = staging / entry["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["cp", "--reflink=always", "--preserve=timestamps", str(submission / entry["path"]), str(dest)],
                       check=True, timeout=600)
        made.append(target / entry["path"])
    (staging / "KEPT_FROM_SUBMISSION.json").write_text(json.dumps(
        {"version": version["version"], "why": "the harness skipped this version's snapshot",
         "matched": "every file's size and mtime equal MANIFEST.json"}, indent=1) + "\n")
    staging.rename(target)
    return made


def runs(list_path: Path = LIST) -> list[Path]:
    """The listed runs that still exist under compression_bench/evolve_space/evolve/."""
    try:
        wanted = json.loads(list_path.read_text()).get("runs", [])
    except (OSError, json.JSONDecodeError):
        return []
    found = []
    for run_id in wanted:
        found += sorted((REPO / "compression_bench" / "evolve_space" / "evolve").glob(f"*/{run_id}"))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", default=str(LIST))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for run_dir in runs(Path(args.list)):
        made = keep(run_dir, dry_run=args.dry_run)
        copied = keep_skipped(run_dir, dry_run=args.dry_run)
        print(f"{run_dir.name}: {len(made)} new links, {len(copied)} files copied from the submission")
    return 0


if __name__ == "__main__":
    sys.exit(main())
