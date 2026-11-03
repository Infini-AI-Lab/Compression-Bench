#!/usr/bin/env python3
"""Is a built image what this checkout's sources would build?

    image_freshness.py digest <image key>      # the inputs digest of compression_bench/environments/images.json[key]
    image_freshness.py check  [image key ...]  # each built image's label equals its digest

Every image in `compression_bench/environments/images.json` that the deploy script
builds carries `LABEL compression_bench.image_inputs_sha256=<digest>`, where the digest
covers every file of its build context (paths and bytes) and, for an image
built FROM another listed image, that image's digest too. A run refuses an
image whose label is missing or differs: an image built from another Dockerfile,
dataset or paper library is not the environment the task names.
Stdlib only.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
IMAGES = REPO / "compression_bench" / "environments" / "images.json"
LABEL = "compression_bench.image_inputs_sha256"
ALWAYS_EXCLUDED = ("__pycache__", "*.pyc", ".pytest_cache")


def manifest() -> dict:
    data = json.loads(IMAGES.read_text())
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def _excluded(relative: Path, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(part, pattern) for part in relative.parts for pattern in patterns)


def digest(key: str) -> str:
    images = manifest()
    entry = images[key]
    if entry.get("pulled"):
        raise ValueError(f"{key} is pulled by tag, not built from this checkout")
    context = REPO / entry["context"]
    patterns = ALWAYS_EXCLUDED + tuple(entry.get("exclude", ()))
    h = hashlib.sha256()
    for path in sorted(p for p in context.rglob("*") if p.is_file()):
        relative = path.relative_to(context)
        if _excluded(relative, patterns):
            continue
        h.update(relative.as_posix().encode() + b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                h.update(chunk)
        h.update(b"\0")
    for arg, base in sorted(entry.get("build_args", {}).items()):
        if base in images:
            h.update(f"{arg}={digest(base)}".encode())
    return h.hexdigest()


def label_of(tag: str) -> str | None:
    done = subprocess.run(["docker", "image", "inspect", "--format",
                           "{{ index .Config.Labels \"" + LABEL + "\" }}", tag],
                          capture_output=True, text=True)
    if done.returncode != 0:
        return None
    value = done.stdout.strip()
    return value if value and value != "<no value>" else ""


def check(keys: list[str]) -> list[str]:
    problems = []
    images = manifest()
    for key in keys or [k for k, v in images.items() if not v.get("pulled")]:
        entry = images[key]
        if entry.get("pulled"):
            if label_of(entry["tag"]) is None:
                problems.append(f"{entry['tag']} is not present (docker pull {entry['tag']})")
            continue
        label = label_of(entry["tag"])
        if label is None:
            problems.append(f"{entry['tag']} is not built (compression_bench/scripts/deploy/01-build-images.sh)")
        elif label != digest(key):
            problems.append(f"{entry['tag']} is stale: label {label[:12] or 'missing'} != inputs "
                            f"{digest(key)[:12]}; rebuild it")
    return problems


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "digest":
        print(digest(argv[1]))
        return 0
    if argv and argv[0] == "check":
        problems = check(argv[1:])
        for problem in problems:
            print(problem)
        if not problems:
            print("images fresh")
        return 1 if problems else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
