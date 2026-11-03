#!/usr/bin/env python3
"""Delete the weights of finished checkpoints of LIVE runs, keeping what is under development.

    python3 compression_bench/scripts/checkpoint_janitor.py [--once] [--dry-run] [--interval 300]
        [--cold-min 90] [--min-mb 200] [--min-group-gb 2] [--force] [--probe-wait-min 45] [--run RUN_DIR ...]

The rule: checkpoints of a running evolve are not kept on disk, except the one being developed. Every
`--interval` seconds, for each live run (a journal and no SUMMARY.json):

  version snapshots   `compression_output/.objects/<lineage>/vN/`: every agent version but the
                      newest has its weight files deleted. Before that, what matters is copied to
                      `<lineage>/CHECKPOINT_NOTES/vN.json`: the agent's note, size, composition and the
                      public Checker's result for that exact version (public PPL and reward, decode
                      latency), all of which `VERSION.json` already carries, and the quick outside
                      measurement (`vN.probe.json`, `checkpoint_probe.py`, started by the scheduler) a
                      superseded version waits for up to --probe-wait-min (not with --force). The janitor
                      itself uses no GPU.
  workspace weights   weight files (>= --min-mb) under the run's `work/`, grouped by directory. The
                      group written most recently, and any group written in the last --cold-min
                      minutes, is under development and stays; every other group is deleted after
                      its file list is appended to `<run>/cleaned_checkpoints.jsonl`. A group under --min-group-gb in
                      total (calibration data, small test models) is not a checkpoint worth deleting.

A directory whose name holds backup, best or final is the agent's own safety copy and is kept.
`submission/` is never touched (it is what the Verifier measures), and neither is anything the harness
delivers. A run held for credit (`CREDIT_HOLD.json`, its container paused) or kept after its driver died
(`KEPT_AFTER_FAILURE.json`, scheduler --keep-on-failure) is not touched at all. The agent's process is
not signalled; a file deleted under an open handle just stays alive until closed. A run's final checkpoint verdicts then cover what is left (the newest version and the final
submission); the cleaned ones were measured by the agent's own Checker calls, recorded in the notes.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "compression_bench" / "harness"))

from lifecycle.agent import CREDIT_HOLD_FILE  # noqa: E402
from lifecycle.boundary import WEIGHT_SUFFIXES  # noqa: E402
from lifecycle.utils import parse_utc, read_json, utc_now, write_json_atomically  # noqa: E402

#: The checkouts whose live runs are cleaned: this one and any named in COMPRESSION_BENCH_SHARED_CHECKOUTS
#: (paths separated by `:`) that share the host's disk.
REPOS = [REPO, *(Path(p) for p in os.environ.get("COMPRESSION_BENCH_SHARED_CHECKOUTS", "").split(":") if p)]
#: Never looked at in a run's work/ (the harness's own, the CLI's home and its transcripts).
SKIP_PARTS = frozenset({".agent-home", ".versions", "__pycache__"})
#: A directory the agent itself named a safety copy (a rollback of its submission) is kept.
PROTECTED_WORDS = ("backup", "best", "final")
#: Safety copies are kept only up to this many bytes of weight files per run, the most recently written first;
#: an agent that names every build "...-best" would otherwise keep the whole disk.
PROTECTED_BUDGET_BYTES = 60 * 2**30
#: Weight files outside the group being written, per run, whatever their age: the oldest groups go first past
#: this. Idleness alone is no limit -- an agent can `touch` a directory to keep it "hot".
RUN_BUDGET_BYTES = 100 * 2**30
#: Whole words of a directory name that mark the agent's INPUT data (calibration blocks, teacher or reference
#: activations), not a checkpoint. Kept while the disk is healthy; a short disk cleans them too (`protect_data=False`).
DATA_WORDS = frozenset({"data", "dataset", "datasets", "calib", "calibration", "teacher", "ref", "reference", "refs",
                        "blocks", "corpus", "tok", "tokens", "train"})
#: A live run whose driver died and that the scheduler keeps as it was (--keep-on-failure): not touched either.
KEPT_FILE = "KEPT_AFTER_FAILURE.json"
#: An accuracy figure the agent's version note quotes, kept in the checkpoint's note.
ACCURACY_MENTION = re.compile(
    r"((?:MATH-?500|GSM8K|HumanEval|TriviaQA|MMLU|acc(?:uracy)?|pass@1)[^;,\n]{0,50}?\d+(?:\.\d+)?\s*%)", re.I)
README_NAMES = ("README.md", "NOTES.md", "DESIGN.md", "README.txt")
#: checkpoint_probe.py's lock file in a snapshot it is reading, and how long such a lock can be live.
PROBE_LOCK = ".probing"
PROBE_LOCK_MAX_S = 25 * 60
#: A superseded checkpoint waits at most this long for its quick outside measurement.
PROBE_WAIT_S = 45 * 60

#: Where the agent's container sees its workspace and /tmp (`/app/work/.tmp`); a process of the container shows
#: these paths in /proc/<pid>/fd and /proc/<pid>/maps, which the host reads.
CONTAINER_PATHS = {"/app/work/": "", "/tmp/": ".tmp/"}


def open_relative_paths(prefixes=None) -> set[str]:
    """Paths (relative to a run's work/) that some process of this user has open or memory-mapped right now.
    Reading does not change a file's mtime, so a directory a running job reads would otherwise look idle.
    Matching is by relative name across all runs: over-cautious, never under."""
    prefixes = prefixes or CONTAINER_PATHS
    found = set()
    uid = os.getuid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != uid:
                continue
        except OSError:
            continue
        names = []
        try:
            fds = list((entry / "fd").iterdir())
        except OSError:
            fds = []
        for fd in fds:
            try:
                names.append(os.readlink(fd))
            except OSError:
                continue                    # that descriptor closed meanwhile; the others still count
        try:
            for line in (entry / "maps").read_text(errors="replace").splitlines():
                parts = line.split(None, 5)
                if len(parts) == 6:
                    names.append(parts[5].strip())
        except OSError:
            pass
        for name in names:
            for prefix, rel in prefixes.items():
                if name.startswith(prefix):
                    found.add(rel + name[len(prefix):])
    return found


def in_use(group: Path, work: Path, open_paths: set[str]) -> bool:
    rel = group.relative_to(work).as_posix() + "/"
    return any(path.startswith(rel) for path in open_paths)


def looks_like_model(group: Path) -> bool:
    """A checkpoint, whatever its name: it has a model config or `model*.safetensors` shards."""
    try:
        if (group / "config.json").exists():
            return True
        return any(p.name.startswith("model") and p.suffix == ".safetensors" for p in group.iterdir())
    except OSError:
        return False


def is_data_dir(group: Path, work: Path) -> bool:
    if looks_like_model(group):
        return False                        # e.g. "ref6-best-tok": a checkpoint, not input data
    for part in group.relative_to(work).parts:
        if DATA_WORDS & set(re.split(r"[^a-z]+", part.lower())):
            return True
    return False


def live_runs(repos=None) -> list[Path]:
    runs = []
    for repo in (repos or REPOS):
        for journal in sorted((Path(repo) / "compression_bench" / "evolve_space" / "evolve").glob("*/*/journal.jsonl")):
            root = journal.parent
            if not (root / "SUMMARY.json").exists() and (root / "identity.json").exists():
                runs.append(root)
    return runs


def prototypes_of(root: Path) -> Path:
    """`<repo>/compression_output` of the checkout a run directory belongs to."""
    return root.parents[4] / "compression_output"


def weight_files(folder: Path, min_bytes: int = 0) -> list[Path]:
    out = []
    for path in folder.rglob("*"):
        try:
            if not path.is_file() or path.is_symlink() or path.suffix.lower() not in WEIGHT_SUFFIXES:
                continue
            if SKIP_PARTS & set(path.relative_to(folder).parts):
                continue
            if path.stat().st_size >= min_bytes:
                out.append(path)
        except OSError:
            continue                    # a file that vanished while we looked
    return out


def newest_mtime(files) -> float:
    """The latest write among files, ignoring any that disappeared meanwhile (0 if none is left)."""
    latest = 0.0
    for f in files:
        try:
            latest = max(latest, f.stat().st_mtime)
        except OSError:
            continue
    return latest


def size_of(files) -> int:
    total = 0
    for f in files:
        try:
            total += f.stat().st_size
        except OSError:
            continue
    return total



def probing(snapshot: Path) -> bool:
    """A probe holds the snapshot (a lock file younger than any probe can run); a stale lock is ignored."""
    try:
        return time.time() - (snapshot / PROBE_LOCK).stat().st_mtime < PROBE_LOCK_MAX_S
    except OSError:
        return False


def design_of(lineage: Path, row: dict, snapshot: Path) -> dict:
    """What the checkpoint is: the agent's note, its quantization config, its code files and a README."""
    files = lineage / row["name"] / "files"
    design = {"note": row.get("note")}
    config = {}
    for candidate in (files / "model" / "config.json", snapshot / "model" / "config.json", snapshot / "config.json"):
        config = read_json(candidate, {}) or {}
        if config:
            break
    quant = config.get("quantization_config") or {k: v for k, v in config.items() if "quant" in k.lower()}
    summary = {}
    for key, value in (quant or {}).items():
        if isinstance(value, dict) and len(value) > 8:          # per-module tables: their distinct values
            summary[key] = dict(collections.Counter(str(v) for v in value.values()).most_common(8))
        else:
            summary[key] = value if len(str(value)) < 300 else str(value)[:300]
    if summary:
        design["quantization_config"] = summary
    for name in README_NAMES:
        for base in (files, snapshot):
            path = base / name
            if path.is_file():
                design["readme_excerpt"] = path.read_text(encoding="utf-8", errors="replace")[:2500]
                break
        if "readme_excerpt" in design:
            break
    if files.is_dir():
        design["code_files"] = sorted(
            ((p.relative_to(files).as_posix(), p.stat().st_size) for p in files.rglob("*.py") if p.is_file()))[:40]
    return design


def agent_measured(record: dict, row: dict) -> dict:
    """The agent's own measurements: the public Checker's result for exactly this version, and any
    accuracy figure its note quotes."""
    out = {"public_check": record.get("public_check")}
    mentions = ACCURACY_MENTION.findall(row.get("note") or "")
    if mentions:
        out["accuracy_in_note"] = mentions[:6]
    return out


def agent_versions(root: Path, prototypes: Path | None = None):
    """(lineage dir, agent version rows oldest first) of a run's lineage."""
    identity = read_json(root / "identity.json", {}) or {}
    lineage_id = identity.get("lineage_id")
    if not lineage_id:
        return None, "", []
    prototypes = Path(prototypes) if prototypes else prototypes_of(root)
    lineage = prototypes / lineage_id
    rows = []
    try:
        for line in (lineage / "versions.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        return lineage, lineage_id, []
    return lineage, lineage_id, [r for r in rows if r.get("source") == "agent"]


def needs_probe(lineage: Path, version: str) -> bool:
    """No outside measurement of this version yet (a failed one for want of a card is not recorded)."""
    return not (lineage / "CHECKPOINT_NOTES" / f"{version}.probe.json").exists()


def eligible_since(agent: list[dict], index: int) -> float:
    """When the version stopped being the newest: the capture time of the next agent version."""
    try:
        return parse_utc(agent[index + 1]["captured_utc"]).timestamp()
    except (IndexError, KeyError, ValueError):
        return time.time()


def probe_candidates(root: Path, prototypes: Path | None = None) -> list[dict]:
    """Superseded snapshots that still hold weights and have not been probed, oldest first."""
    lineage, lineage_id, agent = agent_versions(root, prototypes)
    out = []
    objects = (Path(prototypes) if prototypes else prototypes_of(root)) / ".objects" / lineage_id
    for index, row in enumerate(agent[:-1]):                       # the newest is under development
        snapshot = objects / row["version"]
        if snapshot.is_dir() and not probing(snapshot) and weight_files(snapshot) and needs_probe(lineage, row["version"]):
            out.append({"run": root, "lineage_id": lineage_id, "version": row["version"], "snapshot": snapshot,
                        "eligible_since": eligible_since(agent, index), "task_id": (read_json(root / "identity.json", {}) or {}).get("task_id")})
    return out


def clean_snapshots(root: Path, *, dry_run: bool, prototypes: Path | None = None, force: bool = False,
                    wait_s: float = PROBE_WAIT_S) -> list[dict]:
    """All agent snapshots of the run's lineage but the newest lose their weights (notes first)."""
    lineage, lineage_id, agent = agent_versions(root, prototypes)
    if not agent:
        return []
    prototypes = Path(prototypes) if prototypes else prototypes_of(root)
    keep = {agent[-1]["version"]}                               # the newest is under development
    done = []
    for index, row in enumerate(agent):
        version = row["version"]
        snapshot = prototypes / ".objects" / lineage_id / version
        if version in keep or not snapshot.is_dir():
            continue
        if probing(snapshot):
            continue                                    # a quick outside measurement is reading it
        weights = weight_files(snapshot)
        if not weights:
            continue
        if not force and needs_probe(lineage, version) and time.time() - eligible_since(agent, index) < wait_s:
            continue                                    # give the quick outside measurement its chance first
        freed = size_of(weights)
        record = read_json(lineage / row["name"] / "VERSION.json", {}) or {}
        probe = read_json(lineage / "CHECKPOINT_NOTES" / f"{version}.probe.json", None)
        measured = agent_measured(record, row)
        have_ppl = bool((measured.get("public_check") or {}).get("public_ppl")) or bool(probe and probe.get("ppl"))
        note = {"kind": "checkpoint_note", "lineage": lineage_id, "version": version,
                "run_id": row.get("run_id"), "captured_utc": row.get("captured_utc"),
                "size": {"total_bytes": row.get("total_bytes"), "files": row.get("files"), "weight_bytes_removed": freed,
                         "composition": record.get("composition")},
                "design": design_of(lineage, row, snapshot),
                "measured_by_agent": measured,
                "measured_outside": probe,
                "missing": [name for name, ok in (("ppl_and_reward", have_ppl),
                                                  ("accuracy", bool(probe and probe.get("accuracy")) or "accuracy_in_note" in measured))
                            if not ok],
                "manifest_identity": record.get("manifest_identity"),
                "cleaned_utc": utc_now(),
                "why": "not the checkpoint under development; its weights were deleted, this is what was recorded"}
        if not dry_run:
            write_json_atomically(lineage / "CHECKPOINT_NOTES" / f"{version}.json", note)
            for w in weights:
                try:
                    w.unlink()
                except FileNotFoundError:
                    pass
        done.append({"version": version, "freed": freed})
    return done


def clean_workspace(root: Path, *, cold_s: float, min_bytes: int, dry_run: bool, now: float,
                    min_group_bytes: int = 0, protect_data: bool = True,
                    budget_bytes: int = PROTECTED_BUDGET_BYTES, run_budget_bytes: int = RUN_BUDGET_BYTES,
                    open_paths: set[str] | None = None) -> list[dict]:
    work = root / "work"
    if not work.is_dir():
        return []
    groups: dict[Path, list[Path]] = {}
    for path in weight_files(work, min_bytes):
        groups.setdefault(path.parent, []).append(path)
    if not groups:
        return []
    newest = {g: newest_mtime(files) for g, files in groups.items()}
    current = max(newest, key=newest.get)                       # being written now: under development
    # Which protected-name groups fit the budget (newest first); the rest are cleaned like any other.
    kept_safety, spent = set(), 0
    for group in sorted((g for g in groups if g != current and any(
            word in part.lower() for part in g.relative_to(work).parts for word in PROTECTED_WORDS)),
            key=newest.get, reverse=True):
        size = size_of(groups[group])
        if spent + size <= budget_bytes:
            kept_safety.add(group)
            spent += size
    done = []
    sizes = {g: size_of(files) for g, files in groups.items()}

    def delete(group, files, why):
        entries = []
        for f in sorted(files):
            try:
                entries.append({"path": str(f.relative_to(root)), "bytes": f.stat().st_size, "mtime": int(f.stat().st_mtime)})
            except OSError:
                continue
        freed = sum(e["bytes"] for e in entries)
        if not dry_run:
            with (root / "cleaned_checkpoints.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"utc": utc_now(), "group": str(group.relative_to(root)), "files": len(entries),
                                         "bytes": freed, "idle_min": int((now - newest[group]) / 60), "why": why,
                                         "entries": entries}) + "\n")
            for f in files:
                try:
                    f.unlink()
                except FileNotFoundError:
                    pass                        # the agent removed it first
        done.append({"group": str(group.relative_to(root)), "files": len(entries), "freed": freed,
                     "idle_min": int((now - newest[group]) / 60), "why": why})

    open_paths = open_relative_paths() if open_paths is None else open_paths

    def kept(group):
        return (group == current or group in kept_safety or (protect_data and is_data_dir(group, work))
                or in_use(group, work, open_paths))              # a running job has it open or mapped

    gone = set()
    for group, files in sorted(groups.items()):
        if group == current or now - newest[group] < cold_s or kept(group) or sizes[group] < min_group_bytes:
            continue
        delete(group, files, "idle")
        gone.add(group)
    # The run budget: past it, the oldest groups go whatever their age (the current one and data stay).
    standing = [g for g in groups if g not in gone and not kept(g) and sizes[g] >= min_group_bytes]
    total = sum(sizes[g] for g in standing)
    for group in sorted(standing, key=newest.get):
        if total <= run_budget_bytes:
            break
        delete(group, groups[group], "run budget")
        total -= sizes[group]
    return done


def sweep(args) -> dict:
    """One pass over the live runs (or `args.run`): {run name: what was cleaned, errors, GiB freed}.
    `args.force` deletes without waiting for a probe and takes input data too (a short disk);
    `args.budget_scale` shrinks the safety-copy and run budgets."""
    now = time.time()
    summary = {}
    runs = [Path(r).resolve() for r in args.run] if args.run else live_runs()
    for root in runs:
        if (root / CREDIT_HOLD_FILE).exists() or (root / KEPT_FILE).exists():
            # Held for credit (lifecycle/agent.py credit_hold), or kept after its driver died (scheduler
            # --keep-on-failure): to be resumed as it was.
            summary[root.name] = {"snapshots": [], "workspace": [], "errors": [], "freed_gib": 0.0, "credit_hold": True}
            continue
        snaps, work, errors = [], [], []
        for name, step in (("snapshots", lambda: clean_snapshots(root, dry_run=args.dry_run, force=args.force,
                                                                  wait_s=args.probe_wait_s)),
                           ("workspace", lambda: clean_workspace(
                               root, cold_s=args.cold_min * 60, min_bytes=args.min_mb * 2**20, dry_run=args.dry_run,
                               now=now, min_group_bytes=int(args.min_group_gb * 2**30), protect_data=not args.force,
                               budget_bytes=int(PROTECTED_BUDGET_BYTES * args.budget_scale),
                               run_budget_bytes=int(RUN_BUDGET_BYTES * args.budget_scale)))):
            try:                         # one failing step or run must never stop the others
                (snaps if name == "snapshots" else work).extend(step())
            except Exception as exc:      # noqa: BLE001
                errors.append(f"{name}: {type(exc).__name__}: {str(exc)[:200]}")
                traceback.print_exc(file=sys.stderr)
        summary[root.name] = {"snapshots": snaps, "workspace": work, "errors": errors,
                              "freed_gib": round((sum(s["freed"] for s in snaps) + sum(w["freed"] for w in work)) / 2**30, 1)}
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--interval", type=float, default=300)
    parser.add_argument("--cold-min", type=float, default=90,
                        help="minutes without a write before a group of weight files is finished")
    parser.add_argument("--min-mb", type=float, default=200)
    parser.add_argument("--force", action="store_true", help="do not wait for a checkpoint's outside measurement")
    parser.add_argument("--probe-wait-min", type=float, default=PROBE_WAIT_S / 60,
                        help="minutes a superseded checkpoint waits for its quick outside measurement")
    parser.add_argument("--min-group-gb", type=float, default=2,
                        help="a directory of weight files smaller than this in total is left alone")
    parser.add_argument("--run", action="append", help="a run directory (default: every live run)")
    args = parser.parse_args()
    args.probe_wait_s = args.probe_wait_min * 60
    args.budget_scale = 1.0
    while True:
        try:
            result = sweep(args)
        except Exception as exc:          # noqa: BLE001 - the loop never dies; the next sweep tries again
            print(f"{utc_now()} JANITOR ERROR {type(exc).__name__}: {str(exc)[:300]}", flush=True)
            result = {}
        for name, row in result.items():
            for error in row.get("errors", []):
                print(f"{utc_now()} JANITOR ERROR {name}: {error}", flush=True)
            if row["freed_gib"]:
                print(f"{utc_now()} {'WOULD FREE' if args.dry_run else 'freed'} {row['freed_gib']} GiB in {name}: "
                      f"{len(row['snapshots'])} snapshot(s), {len(row['workspace'])} workspace group(s)", flush=True)
                for w in row["workspace"]:
                    print(f"    {w['group']}: {w['files']} files, {w['freed'] / 2**30:.1f} GiB, idle {w['idle_min']} min", flush=True)
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
