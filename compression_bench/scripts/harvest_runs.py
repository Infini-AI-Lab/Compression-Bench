#!/usr/bin/env python3
"""Keep every metric of finished runs, then free their weights (lifecycle/harvest.py).

    harvest_runs.py --run <run dir> [--run ...]      # one or more runs
    harvest_runs.py --all                             # every finished run (a sweep)
        [--dry-run] [--keep N] [--metrics-only]

The driver does this itself at the end of every run; this is for a run whose
driver did not get there, for a periodic sweep, and to re-collect after a
verdict re-run.
A live run (not in a terminal state, or no SUMMARY.json yet) is skipped.
`--dry-run` prints what would be freed and deletes nothing; `--keep N` is the
number of measured weight checkpoints a lineage keeps besides the unmeasured
ones (default COMPRESSION_BENCH_KEEP_CHECKPOINTS, 0); `--metrics-only` writes
METRICS.json and deletes nothing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

from lifecycle import harvest  # noqa: E402
from lifecycle.utils import REPO  # noqa: E402


def finished_runs() -> list[Path]:
    """Every run directory of every namespace (`evolve_space/<namespace>/<task>/<run>/`)."""
    space = REPO / "compression_bench" / "evolve_space"
    return sorted(path.parent for path in space.glob("*/*/*/journal.jsonl"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", type=Path, default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep", type=int)
    parser.add_argument("--metrics-only", action="store_true")
    args = parser.parse_args(argv)
    runs = [path.resolve() for path in args.run] + (finished_runs() if args.all else [])
    if not runs:
        parser.error("name --run or --all")
    total, failed = 0, 0
    for root in runs:
        try:
            metrics = harvest.harvest(root, clean_weights=not args.metrics_only,
                                      keep=args.keep, dry_run=args.dry_run)
        except harvest.NotFinished as exc:
            print(f"skip  {root.name}: {exc}")
            continue
        except Exception as exc:                            # noqa: BLE001
            failed += 1
            print(f"FAIL  {root.name}: {type(exc).__name__}: {exc}")
            continue
        cleanup = metrics.get("cleanup") or {}
        freed = cleanup.get("freed_bytes") or 0
        total += freed
        verdict = metrics["verdict"]
        print(f"{'would' if args.dry_run else 'done '} {root.name}: verdict {verdict.get('status')} "
              f"reward {verdict.get('reward')}; {len(metrics['versions'])} versions, "
              f"{len(metrics['public_checks']['checks'])} public checks "
              f"({metrics['public_checks']['source']}); "
              f"{'would free' if args.dry_run else 'freed'} {freed / 2**30:.1f} GiB, "
              f"checkpoints kept {cleanup.get('checkpoints_kept')}")
    print(f"total {'to free' if args.dry_run else 'freed'}: {total / 2**30:.1f} GiB")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
