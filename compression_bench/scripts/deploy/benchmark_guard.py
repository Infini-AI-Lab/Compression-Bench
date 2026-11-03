#!/usr/bin/env python3
"""Keep the evaluation benchmark out of everything the agent's container ships.

    python3 compression_bench/scripts/deploy/benchmark_guard.py scan     # rewrite library-exclude.txt
    python3 compression_bench/scripts/deploy/benchmark_guard.py check    # fail if a listed path is missing
                                                                 # or an unlisted file matches
    python3 compression_bench/scripts/deploy/benchmark_guard.py image TAG   # scan a built image: every
                                                                 # environment, package and file

The hidden evaluation's datasets are `build_accuracy_assets.BENCHMARKS` (GSM8K,
HumanEval and TriviaQA), every split of each (`GUARD_FILES`): training on any
split is training on the benchmark. Papers quote examples of such data.
A file matches when it holds any of their questions: the first ten words of
the question (`QUESTION_FIELD`), lower-cased, as a word 10-gram of the file's
text; questions shorter than ten words are not looked for. Matching files, and
files named after benchmark or C4 data, are listed in
`compression_bench/environments/agent-base/library-exclude.txt`; the agent-base build
withholds every paper listed there from `/opt/compression-papers`.
`prepare_datasets.py` uses the same matcher to drop rows from the training
datasets.

Needs network once (the Hub), pyarrow and huggingface_hub; a deploy tool.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
REPO = _HERE.parents[3] if len(_HERE.parents) > 3 else _HERE.parent   # in an image: unused
LIBRARY = REPO / "compression_bench" / "environments" / "agent-base" / "library"
EXCLUDE = REPO / "compression_bench" / "environments" / "agent-base" / "library-exclude.txt"
DATA_SUFFIXES = (".json", ".jsonl", ".tsv", ".csv", ".parquet", ".arrow", ".txt")
NAME_PATTERNS = tuple(f"*{name}*{suffix}" for name in ("mathqa", "math_qa", "math-qa", "codemmlu",
                                                       "code_mmlu", "sciq", "gsm8k", "gsm-8k",
                                                       "humaneval", "human_eval", "human-eval",
                                                       "trivia_qa", "triviaqa", "trivia-qa")
                      for suffix in DATA_SUFFIXES) + ("c4-validation*", "c4-train*", "*c4*.json.gz")
TEXT_SUFFIXES = (".json", ".jsonl", ".tsv", ".csv", ".txt", ".parquet", ".arrow", ".yaml", ".yml",
                 ".py", ".md", ".jinja", ".tex")
MAX_BYTES = 512 * 2**20
N = 10
_WORD = re.compile(r"[a-z0-9]+")


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def benchmark_questions() -> list[tuple[str, int]]:
    """(question, signature words) for every question of every split the gate draws from."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import build_accuracy_assets as assets
    out = []
    for name in assets.BENCHMARKS:
        field = assets.QUESTION_FIELD[name]
        out += [(str(row[field]), N) for row in assets.rows_of(name, assets.GUARD_FILES[name])]
    return out


def gate_questions() -> list[tuple[str, int]]:
    """(question, signature words) for the questions the hidden gate actually holds:
    `build_accuracy_assets`' sampling, reproduced."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import build_accuracy_assets as assets
    out = []
    for name in sorted(assets.BENCHMARKS):
        field = assets.QUESTION_FIELD[name]
        out += [(str(row[field]), N) for row in assets.gate_rows(name)]
    return out


def package_source(path: str) -> bool:
    """An installed Python package's own source: runtime code, not data."""
    return path.endswith(".py") and ("/site-packages/" in path or "/dist-packages/" in path)


def fingerprint(keys: set[tuple[str, ...]]) -> str:
    import hashlib
    return hashlib.sha256("\n".join(sorted(" ".join(k) for k in keys)).encode()).hexdigest()


def signatures(questions) -> set[tuple[str, ...]]:
    """A question's first n words (N unless given as `(question, n)`); shorter ones are not looked for."""
    keys = set()
    for item in questions:
        text, n = item if isinstance(item, tuple) else (item, N)
        w = words(text)
        if len(w) >= n:
            keys.add(tuple(w[:n]))
    return keys


_LENGTHS: dict[int, tuple[set, set[int]]] = {}


def matches(text: str, keys: set[tuple[str, ...]]) -> int:
    # The key lengths are computed once per key set, not once per text.
    held = _LENGTHS.get(id(keys))
    if held is None or held[0] is not keys:
        held = _LENGTHS[id(keys)] = (keys, {len(k) for k in keys})
    w = words(text)
    return sum(1 for n in held[1] for i in range(len(w) - n + 1) if tuple(w[i:i + n]) in keys)


def file_text(path: Path) -> str:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return json.dumps(pq.read_table(path).to_pylist())
    return path.read_text(errors="ignore")


def scan(root: Path, keys: set[tuple[str, ...]]) -> dict[str, str]:
    found = {}
    for directory, _, names in os.walk(root):
        for name in names:
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if any(fnmatch.fnmatch(name.lower(), p) for p in NAME_PATTERNS):
                found[relative] = "benchmark data by name"
                continue
            if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > MAX_BYTES:
                continue
            try:
                hits = matches(file_text(path), keys)
            except Exception:  # noqa: BLE001 - unreadable data cannot be checked, so it goes
                found[relative] = "unreadable data file"
                continue
            if hits:
                found[relative] = f"{hits} benchmark question(s)"
    # A paper's text extract and its PDF hold the same text: the PDF goes too.
    for relative in [r for r in found if r.endswith(".txt")]:
        for sibling in (root / relative).parent.glob(Path(relative).stem + ".*"):
            name = sibling.relative_to(root).as_posix()
            found.setdefault(name, f"same document as {Path(relative).name}")
    return found


def read_exclude() -> list[str]:
    if not EXCLUDE.is_file():
        return []
    return [line.split("\t", 1)[0] for line in EXCLUDE.read_text().splitlines()
            if line and not line.startswith("#")]


IMAGE_ROOTS = ("/opt", "/app", "/usr/local/lib/python3.12/dist-packages", "/usr/lib/python3/dist-packages")


def scan_image(tag: str) -> int:
    """Run this matcher inside `tag` (no network) over everything the agent can read."""
    import subprocess
    import tempfile
    keys = {"all": sorted(" ".join(k) for k in signatures(benchmark_questions())),
            "gate": sorted(" ".join(k) for k in signatures(gate_questions()))}
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "keys.json").write_text(json.dumps(keys))
        got = subprocess.run(
            ["docker", "run", "--rm", "--network", "none", "--entrypoint", "python3",
             "-v", f"{Path(__file__).resolve()}:/guard/benchmark_guard.py:ro",
             "-v", f"{tmp}/keys.json:/guard/keys.json:ro", tag,
             "/guard/benchmark_guard.py", "dirs", *IMAGE_ROOTS], capture_output=True, text=True)
    print(got.stdout, end="")
    print(got.stderr[-2000:], end="", file=sys.stderr)
    return got.returncode


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("scan", "check", "image", "dirs"))
    parser.add_argument("targets", nargs="*")
    parser.add_argument("--root", type=Path, default=LIBRARY)
    args = parser.parse_args(argv)
    if args.command == "image":
        return scan_image(args.targets[0])
    if args.command == "dirs":
        # Inside an image: signatures come from the mounted keys file. Installed
        # packages' own source can share code with questions a code benchmark drew
        # from it (benchmarks quote real projects); that is reported, and fails only
        # when one of the questions the hidden gate holds is among them.
        loaded = json.loads(Path("/guard/keys.json").read_text())
        keys, gate = ({tuple(k.split()) for k in loaded[name]} for name in ("all", "gate"))
        hits = {}
        for root in map(Path, args.targets):
            if root.is_dir():
                hits.update({f"{root}/{k}": v for k, v in scan(root, keys).items()})
        for path in [p for p, why in hits.items() if package_source(p) and "question" in why]:
            if not matches(file_text(Path(path)), gate):
                hits[path] = "package source sharing code with a non-gate question"
        held = {p: why for p, why in hits.items()
                if why not in ("unreadable data file", "package source sharing code with a non-gate question")}
        for path, why in sorted(hits.items()):
            print(f"{path}\t{why}")
        print(f"{len(held)} file(s) hold benchmark questions; {len(hits) - len(held)} unreadable or "
              "package-source file(s) reported", file=sys.stderr)
        return 1 if held else 0
    found = scan(args.root, signatures(benchmark_questions()))
    if args.command == "scan":
        lines = ["# Library files the agent-base build withholds: they hold the evaluation benchmark.",
                 "# Written by compression_bench/scripts/deploy/benchmark_guard.py scan; path<TAB>reason."]
        lines += [f"{path}\t{why}" for path, why in sorted(found.items())]
        EXCLUDE.write_text("\n".join(lines) + "\n")
        print(f"{len(found)} file(s) listed in {EXCLUDE.relative_to(REPO)}")
        return 0
    listed = set(read_exclude())
    missing = sorted(set(found) - listed)
    for path in missing:
        print(f"UNLISTED {path}: {found[path]}")
    print(f"{len(found)} matching, {len(listed)} listed, {len(missing)} unlisted")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
