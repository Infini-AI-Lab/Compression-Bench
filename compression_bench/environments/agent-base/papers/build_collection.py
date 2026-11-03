#!/usr/bin/env python3
"""Assemble /opt/compression-papers from the restored library (image build time).

    build_collection.py --library DIR --exclude FILE --skill DIR --out DIR [--supplement FILE]

Takes the papers only, never author source code: every paper of the
Awesome-LLM-Compression catalogue (`external/awesome-llm-compression/papers.json`)
that was retrieved, every paper of `references/papers.json` the catalogue
does not list, and every paper of the arXiv supplement (`supplement.json` next
to this script: `supplement/arxiv-<id>.pdf|.txt` in the library, restored by
`scripts/deploy/prepare_references.py`) whose id is not already taken, minus
every paper whose PDF or text extract the benchmark guard lists
(`library-exclude.txt`). Writes

    papers/<id>.pdf, papers/<id>.txt   one flat directory; id = the arXiv id, or
                                       the catalogue's own id for a paper without one
    catalog.json                       id, title, year, categories, url, files
    index.sqlite                       SQLite FTS5 (BM25) over title and full text
    skill/                             the compression-papers skill (SKILL.md, scripts/)

and refuses a duplicate id, a missing file or a SKILL.md whose paper count is
not the collection's. Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import posixpath
import re
import shutil
import sys
import urllib.parse
from collections import Counter
from pathlib import Path, PurePosixPath

CATALOGUE = "external/awesome-llm-compression/papers.json"
REFERENCES = "references/papers.json"
SOURCE = "https://github.com/HuangOwen/Awesome-LLM-Compression"
#: The pinned arXiv supplement (tracked); its files live in the library's SUPPLEMENT_DIR.
SUPPLEMENT = Path(__file__).resolve().parent / "supplement.json"
SUPPLEMENT_DIR = "supplement"
SUPPLEMENT_SOURCE = "https://arxiv.org"

#: The references/papers.json entries whose file the catalogue does not use: their
#: arXiv ids and titles (the entries carry neither). One the catalogue lists by
#: its arXiv id is taken from the catalogue.
REFERENCE_PAPERS = {
    "albert": ("1909.11942", "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations"),
    "d2quant": ("2602.02546", "D2Quant: Accurate Low-bit Post-Training Weight Quantization for LLMs"),
    "gptaq": ("2504.02692", "GPTAQ: Efficient Finetuning-Free Quantization for Asymmetric Calibration"),
    "qtip": ("2406.11235", "QTIP: Quantization with Trellises and Incoherence Processing"),
    "sinq": ("2509.22944", "SINQ: Sinkhorn-Normalized Quantization for Calibration-Free Low-Precision LLM Weights"),
    "spinquant": ("2405.16406", "SpinQuant: LLM Quantization with Learned Rotations"),
}
#: references/papers.json categories, as the catalogue's section names.
REFERENCE_CATEGORIES = {"quantization": "Quantization", "pruning": "Pruning and Sparsity",
                        "distillation": "Distillation", "low_rank": "Other", "parameter_sharing": "Other"}


def paper_id(raw: str) -> str:
    """`arxiv-2210.17323` -> `2210.17323`; `openreview-X` stays; a URL id -> `<site>-<last path part>`."""
    if raw.startswith("arxiv-"):
        return raw[len("arxiv-"):]
    if raw.startswith("http"):
        url = urllib.parse.urlparse(raw)
        site = url.netloc.split(".")[-2]
        stem = re.sub(r"\.(html|pdf)$", "", PurePosixPath(url.path.rstrip("/")).name, flags=re.I)
        return f"{site}-" + re.sub(r"[^a-z0-9.]+", "-", stem.lower()).strip("-.")
    return raw


def library_path(base: str, path: str) -> str:
    return posixpath.normpath(posixpath.join(base, path))


def supplement_rows(supplement: Path) -> list[dict]:
    """The supplement's papers as collection rows (sources relative to the library)."""
    data = json.loads(supplement.read_text())
    if data.get("schema_version") != 1:
        raise SystemExit(f"{supplement}: unsupported schema_version {data.get('schema_version')}")
    rows = []
    for entry in data["papers"]:
        arxiv = entry["arxiv_id"]
        stem = f"{SUPPLEMENT_DIR}/arxiv-{arxiv}"
        rows.append({"id": arxiv, "title": entry["title"], "year": entry["year"],
                     "categories": list(entry["categories"]), "url": entry["abs_url"],
                     "source": {"pdf": f"{stem}.pdf", "text": f"{stem}.txt"}})
    return rows


def collect(library: Path, excluded: set[str], supplement: Path | None = SUPPLEMENT) -> tuple[list[dict], int]:
    catalogue = json.loads((library / CATALOGUE).read_text())
    references = json.loads((library / REFERENCES).read_text())["papers"]
    rows, withheld = [], 0
    for entry in catalogue["papers"]:
        if entry["status"] != "available":
            continue
        pdf, text = library_path("", entry["pdf_path"]), library_path("", entry["text_path"])
        if {pdf, text} & excluded:
            withheld += 1
            continue
        rows.append({"id": paper_id(entry["id"]), "title": entry["title"], "year": entry["year"],
                     "categories": list(entry["upstream_categories"]), "url": entry["url"],
                     "source": {"pdf": pdf, "text": text}})
    taken = {row["source"]["pdf"] for row in rows}
    listed = {paper_id(entry["id"]) for entry in catalogue["papers"]}
    for entry in references:
        pdf = library_path("references", entry["pdf_path"])
        text = library_path("references", entry["text_path"])
        if pdf in taken:
            continue
        if entry["name"] not in REFERENCE_PAPERS:
            raise SystemExit(f"{REFERENCES}: {entry['name']} is not in the catalogue and has no arXiv id here")
        arxiv, title = REFERENCE_PAPERS[entry["name"]]
        if arxiv in listed:            # the catalogue holds its own copy of the paper
            continue
        if {pdf, text} & excluded:
            withheld += 1
            continue
        categories = list(dict.fromkeys(REFERENCE_CATEGORIES[c] for c in entry["categories"]
                                        if c in REFERENCE_CATEGORIES))
        rows.append({"id": arxiv, "title": title, "year": 2000 + int(arxiv[:2]), "categories": categories,
                     "url": entry["url"], "source": {"pdf": pdf, "text": text}})
    if supplement is not None:
        taken = {row["id"].lower() for row in rows} | {i.lower() for i in listed}
        for row in supplement_rows(supplement):
            if row["id"].lower() in taken:       # the catalogue or the references hold the paper
                continue
            if set(row["source"].values()) & excluded:
                withheld += 1
                continue
            rows.append(row)
    return sorted(rows, key=lambda r: r["id"]), withheld


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, required=True)
    parser.add_argument("--skill", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, default=SUPPLEMENT,
                        help="the pinned arXiv supplement (default: supplement.json next to this script)")
    args = parser.parse_args(argv)
    excluded = {line.split("\t", 1)[0] for line in args.exclude.read_text().splitlines()
                if line and not line.startswith("#")}
    rows, withheld = collect(args.library, excluded, args.supplement)

    ids = [row["id"] for row in rows]
    sys.path.insert(0, str(args.skill / "scripts"))
    sys.dont_write_bytecode = True
    import paperlib
    bad = [i for i in ids if not paperlib.ID_PATTERN.fullmatch(i)]
    counts = Counter(i.lower() for i in ids)
    duplicates = sorted(i for i in ids if counts[i.lower()] > 1)
    if bad or duplicates:
        raise SystemExit(f"invalid ids {bad[:5]}, duplicate ids {duplicates[:5]}")
    for row in rows:
        for kind in ("pdf", "text"):
            source = args.library / row["source"][kind]
            if not source.is_file() or source.stat().st_size == 0:
                raise SystemExit(f"{row['id']}: missing {kind} {row['source'][kind]}")

    stated = set(re.findall(r"(\d+) research papers", (args.skill / "SKILL.md").read_text()))
    if stated != {str(len(rows))}:
        raise SystemExit(f"SKILL.md states {sorted(stated)} research papers; the collection holds {len(rows)}")

    out = args.out
    (out / "papers").mkdir(parents=True, exist_ok=False)
    catalog = []
    for row in rows:
        files = {"pdf": f"papers/{row['id']}.pdf", "text": f"papers/{row['id']}.txt"}
        shutil.copyfile(args.library / row["source"]["pdf"], out / files["pdf"])
        shutil.copyfile(args.library / row["source"]["text"], out / files["text"])
        catalog.append({"id": row["id"], "title": row["title"], "year": row["year"],
                        "categories": row["categories"], "url": row["url"], "files": files})
    head = {"source": SOURCE}
    if any(row["source"]["pdf"].startswith(SUPPLEMENT_DIR + "/") for row in rows):
        head["supplement"] = SUPPLEMENT_SOURCE
    (out / paperlib.CATALOG).write_text(json.dumps({**head, "papers": catalog}, indent=1,
                                                   ensure_ascii=False) + "\n", encoding="utf-8")
    indexed = paperlib.create_index(out / paperlib.INDEX, (
        (row["id"], row["title"], (out / row["files"]["text"]).read_text(encoding="utf-8", errors="replace"))
        for row in catalog))
    shutil.copytree(args.skill, out / "skill", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    print(f"{len(catalog)} papers ({indexed} indexed), {withheld} withheld by the benchmark guard -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
