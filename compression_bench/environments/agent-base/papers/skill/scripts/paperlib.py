"""The paper collection's layout, shared by search.py and read.py. Stdlib only.

    <root>/catalog.json        one row per paper: id, title, year, categories, url, files
    <root>/index.sqlite        SQLite FTS5 full-text index over title and text
    <root>/papers/<id>.pdf     the paper
    <root>/papers/<id>.txt     its text extract

<root> is /opt/compression-papers unless COMPRESSION_PAPERS_ROOT names another.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path

DEFAULT_ROOT = "/opt/compression-papers"
CATALOG = "catalog.json"
INDEX = "index.sqlite"
#: The FTS5 table: (id UNINDEXED, title, body); BM25 weights per column.
TABLE = "fts"
TOKENIZER = "porter unicode61 remove_diacritics 2"
WEIGHTS = (0.0, 5.0, 1.0)
#: A page is one page of the text extract (form-feed separated); a page longer
#: than this, or an extract without page breaks, is cut at line ends into parts
#: of at most this many characters.
PAGE_CHARS = 12000
ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def root() -> Path:
    return Path(os.environ.get("COMPRESSION_PAPERS_ROOT") or DEFAULT_ROOT)


def load_catalog(base: Path | None = None) -> list[dict]:
    base = base or root()
    path = base / CATALOG
    if not path.is_file():
        raise SystemExit(f"no paper catalogue at {path}")
    return json.loads(path.read_text(encoding="utf-8"))["papers"]


def normalize_id(text: str) -> str:
    """`arXiv:YYMM.NNNNNv2`, `arxiv-YYMM.NNNNN`, `YYMM.NNNNN.pdf` -> `YYMM.NNNNN`."""
    value = text.strip()
    value = re.sub(r"^(?:arxiv[:\-]|https?://arxiv\.org/(?:abs|pdf)/)", "", value, flags=re.I)
    value = re.sub(r"\.(?:pdf|txt)$", "", value, flags=re.I)
    value = re.sub(r"^(\d{4}\.\d{4,5})v\d+$", r"\1", value)
    return value


def find(papers: list[dict], text: str) -> dict | None:
    wanted = normalize_id(text).lower()
    for paper in papers:
        if paper["id"].lower() == wanted:
            return paper
    return None


def connect(base: Path | None = None) -> sqlite3.Connection:
    path = (base or root()) / INDEX
    if not path.is_file():
        raise SystemExit(f"no search index at {path}")
    return sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)


def create_index(path: Path, rows) -> int:
    """Write the index from (id, title, text) rows; used when the image is built."""
    path.unlink(missing_ok=True)
    db = sqlite3.connect(path)
    db.execute(f"CREATE VIRTUAL TABLE {TABLE} USING fts5(id UNINDEXED, title, body, "
               f"tokenize='{TOKENIZER}')")
    count = 0
    for paper_id, title, text in rows:
        db.execute(f"INSERT INTO {TABLE} (id, title, body) VALUES (?, ?, ?)",
                   (paper_id, title, compact(text)))
        count += 1
    db.execute(f"INSERT INTO {TABLE}({TABLE}) VALUES ('optimize')")
    db.commit()
    db.execute("VACUUM")
    db.close()
    return count


def compact(text: str) -> str:
    return " ".join(text.split())


def _parts(body: str) -> list[str]:
    out, chunk, size = [], [], 0
    lines = []
    for line in body.split("\n"):
        while len(line) > PAGE_CHARS:
            lines.append(line[:PAGE_CHARS])
            line = line[PAGE_CHARS:]
        lines.append(line)
    for line in lines:
        if chunk and size + len(line) + 1 > PAGE_CHARS:
            out.append("\n".join(chunk).strip("\n"))
            chunk, size = [], 0
        chunk.append(line)
        size += len(line) + 1
    if chunk:
        out.append("\n".join(chunk).strip("\n"))
    return [part for part in out if part.strip()]


def pages(text: str) -> list[str]:
    """The extract as pages: trailing spaces and runs of blank lines removed,
    each page's common left margin removed."""
    out = []
    for piece in text.split("\f"):
        kept, blank = [], False
        for line in piece.splitlines():
            line = line.rstrip()
            if not line.strip():
                if kept and not blank:
                    kept.append("")
                blank = True
                continue
            kept.append(line)
            blank = False
        body = _dedent(kept).strip("\n")
        if body.strip():
            out.extend(_parts(body))
    return out or [""]


def _dedent(lines: list[str]) -> str:
    """Remove the page's left margin: the indentation that all but the few
    least indented lines (a margin note, a page header) share."""
    indents = sorted(len(line) - len(line.lstrip()) for line in lines if line.strip())
    if not indents:
        return ""
    margin = indents[len(indents) // 20]
    return "\n".join(line[min(margin, len(line) - len(line.lstrip())):] for line in lines)


def read_text(paper: dict, base: Path | None = None) -> str:
    return ((base or root()) / paper["files"]["text"]).read_text(encoding="utf-8", errors="replace")
