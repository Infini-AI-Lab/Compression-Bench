#!/usr/bin/env python3
"""Find (and remove) a second copy of the task's BF16 checkpoint in a live run's workspace.

    python3 compression_bench/scripts/bf16_copy_guard.py --run <run root> [--remove]

The checkpoint is mounted read-only at `/input/model`; a run needs no second copy of it, and one (about 50 GiB for a 27B
model, about 230 GiB for a 120B one) fills the disk. A file in the run's `work/` or `submission/` counts as a
copy of a shard when it has the shard's exact size and the same first and last MiB (SHA-256 of both); a modified, requantized
or repacked tensor file never matches. A hard link to the original (same device and inode) costs nothing and is left alone.
Only the identical copy is removed; the agent can read the original in place. Nothing else is touched.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "harness"))

CHUNK = 1 << 20
MIN_BYTES = 64 << 20          # a shard smaller than this is not worth guarding


def fingerprint(path: Path, size: int) -> tuple[str, str]:
    with path.open("rb") as handle:
        head = hashlib.sha256(handle.read(CHUNK)).hexdigest()
        handle.seek(max(0, size - CHUNK))
        tail = hashlib.sha256(handle.read(CHUNK)).hexdigest()
    return head, tail


def reference_index(model_dir: Path) -> dict[int, dict]:
    """{size: {fingerprint: (device, inode)}} of the checkpoint's weight shards (links followed)."""
    index: dict[int, dict] = {}
    for path in sorted(Path(model_dir).iterdir()) if Path(model_dir).is_dir() else ():
        real = path.resolve()
        if not real.is_file() or real.suffix.lower() not in (".safetensors", ".bin", ".pt", ".pth", ".gguf"):
            continue
        stat = real.stat()
        if stat.st_size < MIN_BYTES:
            continue
        index.setdefault(stat.st_size, {})[fingerprint(real, stat.st_size)] = (stat.st_dev, stat.st_ino)
    return index


def find_copies(run_root: Path, index: dict[int, dict]) -> list[tuple[Path, int]]:
    found = []
    if not index:
        return found
    for scope in ("work", "submission"):
        for folder, _dirs, names in os.walk(Path(run_root) / scope):
            for name in names:
                path = Path(folder) / name
                try:
                    stat = path.lstat()
                    if not path.is_file() or path.is_symlink() or stat.st_size not in index:
                        continue
                    original = index[stat.st_size].get(fingerprint(path, stat.st_size))
                except OSError:
                    continue
                if original is not None and (stat.st_dev, stat.st_ino) != original:
                    found.append((path, stat.st_size))
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--remove", action="store_true")
    args = parser.parse_args(argv)
    from lifecycle import record
    created = record.Journal(args.run).last("run_created") or {}
    copies = find_copies(args.run, reference_index(Path(created["model_dir"])))
    for path, size in copies:
        print(f"{'removing' if args.remove else 'copy'} {path} ({size / 2**30:.1f} GiB)")
        if args.remove:
            path.unlink(missing_ok=True)
    print(f"{len(copies)} copies, {sum(s for _p, s in copies) / 2**30:.1f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
