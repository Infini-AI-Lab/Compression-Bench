#!/usr/bin/env python3
"""Stage the training datasets the agent's container carries at /opt/datasets.

    python3 compression_bench/scripts/deploy/prepare_datasets.py            # download, filter, write
    python3 compression_bench/scripts/deploy/prepare_datasets.py --verify   # check the staged copy

Pinned public datasets, none of them an input of any live task's evaluation.
Training data:

  wikitext-103   Salesforce/wikitext, wikitext-103-raw-v1 train: Wikipedia text
  ultrachat-200k HuggingFaceH4/ultrachat_200k, train_sft: multi-turn chat
  fineweb        HuggingFaceFW/fineweb, sample-10BT: English web text

FineWeb and C4 both come from Common Crawl, so every FineWeb document that shares
its URL or any word 13-gram (lowercased alphanumeric words) with any document of
the C4 validation split -- all eight files, every block any task's evaluation
draws from among them -- is dropped before anything is written (`c4_overlap`).

Check data -- a test split of a public benchmark with reference answers, for an
agent to test its submission's answers on; the instruction's notice forbids
training, fine-tuning or calibrating on it:

  math-500       HuggingFaceH4/MATH-500, test: competition math problems

Rows holding an item of the hidden evaluation -- the correctness gate's
benchmarks, every split (`benchmark_guard.matches`) -- are dropped, and so is
every training row sharing a word 13-gram with a document of the public or
hidden perplexity blocks. Those documents are not distributed: the rows they rule out
are listed by row index in `perplexity_overlap.json` beside this script,
written from the documents with

    python3 compression_bench/scripts/deploy/prepare_datasets.py \
        --pin-perplexity-overlap <build_fidelity_assets output>/source/*/documents.jsonl

The output goes to `compression_bench/environments/agent-base/datasets/` (gitignored), with
`DATASETS.json` recording each source, revision, file, the rows kept and
dropped, and the SHA-256 of every file written; `01-build-images.sh` runs this
before agent-base is hashed and built. Needs network once, pyarrow and
huggingface_hub.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import benchmark_guard  # noqa: E402

REPO = HERE.parents[2]
OUT = REPO / "compression_bench" / "environments" / "agent-base" / "datasets"
#: name -> (repository, revision, files, license, content); the instruction's table reads this.
SOURCES = {
    "wikitext-103": ("Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3",
                     ["wikitext-103-raw-v1/train-00000-of-00002.parquet",
                      "wikitext-103-raw-v1/train-00001-of-00002.parquet"],
                     "CC BY-SA 3.0 / GFDL", "wikitext-103-raw-v1 train: Wikipedia articles, raw text"),
    "ultrachat-200k": ("HuggingFaceH4/ultrachat_200k", "8049631c405ae6576f93f445c6b8166f76f5505a",
                       ["data/train_sft-00000-of-00003-a3ecf92756993583.parquet",
                        "data/train_sft-00001-of-00003-0a1804bcb6ae68c6.parquet",
                        "data/train_sft-00002-of-00003-ee46ed25cfae92c6.parquet"],
                       "MIT", "train_sft: multi-turn user/assistant conversations"),
    "fineweb": ("HuggingFaceFW/fineweb", "9bb295ddab0e05d785b879661af7260fed5140fc",
                ["sample/10BT/000_00000.parquet"],
                "ODC-By 1.0 (and the Common Crawl terms of use)",
                "sample-10BT: 241,114 English web documents from Common Crawl (`text`, `url`, `dump`, "
                "`token_count`), the first 250,000 of its first file without every one that shares its "
                "URL or a word 13-gram with a document of the C4 validation split"),
}
#: Only the first rows of these files are taken (the file is 2 GiB).
ROW_LIMITS = {"fineweb": 250_000}
#: Columns kept, when not all.
COLUMNS = {"fineweb": ["text", "id", "dump", "url", "date", "language_score", "token_count"]}
#: Sources filtered against the C4 validation split (`c4_overlap`).
C4_FILTERED = {"fineweb"}
C4_REPOSITORY, C4_REVISION = "allenai/c4", "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
C4_FILES = [f"en/c4-validation.{i:05d}-of-00008.json.gz" for i in range(8)]
NGRAM = 13
#: The same shape; test splits, never training data (the manifest records `use`).
CHECK_SOURCES = {
    "math-500": ("HuggingFaceH4/MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
                 ["test.jsonl"],
                 "MIT (MATH; the PRM800K subset)",
                 "test: 500 competition math problems with worked `solution` and final `answer` (LaTeX)"),
}
ALL_SOURCES = {**SOURCES, **CHECK_SOURCES}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _words(text: str) -> list[str]:
    import re
    return re.findall(r"[a-z0-9]+", text.lower())


def _url(url) -> str:
    url = (url or "").strip().lower()
    for prefix in ("https://", "http://"):
        url = url.removeprefix(prefix)
    return url.removeprefix("www.").rstrip("/")


class C4Overlap:
    """Every URL and word 13-gram of the C4 validation split, to drop what shares one.

    13-grams are 64-bit rolling hashes over word ids; a word C4 never uses cannot be
    in a C4 13-gram, so a window holding one never matches."""

    MULT = 0x9E3779B97F4A7C15

    def __init__(self, documents=None, cache: Path | None = None):
        """`documents`: (url, text) pairs; by default every C4 validation document."""
        import numpy as np
        self.np, self.vocab, self.urls, self.documents, self.files = np, {}, set(), 0, []
        chunks = []
        for url, text in (self._c4(cache) if documents is None else documents):
            self.documents += 1
            if _url(url):
                self.urls.add(_url(url))              # a document without a URL claims none: "" would match every row
            ids = np.fromiter((self.vocab.setdefault(w, len(self.vocab) + 1) for w in _words(text)),
                              dtype=np.uint64)
            if len(ids) >= NGRAM:
                chunks.append(self._hashes(ids))
        self.grams = np.unique(np.concatenate(chunks)) if chunks else np.zeros(0, dtype=np.uint64)
        print(f"C4 validation: {self.documents} documents, {len(self.urls)} URLs, "
              f"{len(self.grams)} distinct word {NGRAM}-grams", flush=True)

    def _c4(self, cache):
        import gzip
        from huggingface_hub import hf_hub_download
        for file in C4_FILES:
            path = hf_hub_download(C4_REPOSITORY, file, repo_type="dataset", revision=C4_REVISION,
                                   cache_dir=cache)
            self.files.append({"file": file, "sha256": sha256(Path(path))})
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    doc = json.loads(line)
                    yield doc.get("url"), doc["text"]

    def _hashes(self, ids):
        np = self.np
        with np.errstate(over="ignore"):
            h = np.zeros(len(ids) - NGRAM + 1, dtype=np.uint64)
            for k in range(NGRAM):
                h = h * np.uint64(self.MULT) + ids[k:len(ids) - NGRAM + 1 + k]
        return h

    def overlapping(self, rows: list[dict]) -> list[bool]:
        """Per row: its URL or one of its word 13-grams is in the C4 validation split."""
        np = self.np
        unknown = np.uint64(1 << 63)
        hits = [_url(row.get("url")) in self.urls for row in rows]
        hashes, owner = [], []
        for i, row in enumerate(rows):
            ids = np.fromiter((self.vocab.get(w, unknown) for w in _words(row.get("text") or "")), dtype=np.uint64)
            if len(ids) < NGRAM:
                continue
            windows = self._hashes(ids)
            # A window holding a word C4 never uses is no C4 13-gram.
            known = np.convolve((ids != unknown).astype(np.int32), np.ones(NGRAM, dtype=np.int32), "valid") == NGRAM
            windows = windows[known]
            hashes.append(windows)
            owner.append(np.full(len(windows), i, dtype=np.int64))
        if hashes and len(self.grams):
            flat, who = np.concatenate(hashes), np.concatenate(owner)
            at = np.searchsorted(self.grams, flat).clip(max=len(self.grams) - 1)
            for i in np.unique(who[self.grams[at] == flat]):
                hits[int(i)] = True
        return hits

    def describe(self) -> dict:
        return {"repository": C4_REPOSITORY, "revision": C4_REVISION, "files": self.files,
                "documents": self.documents, "urls": len(self.urls), "ngram": NGRAM,
                "distinct_ngrams": int(len(self.grams)),
                "rule": "drop a document whose normalised URL, or any word 13-gram of its lowercased "
                        "alphanumeric words, occurs in any C4 validation document"}


#: Training rows sharing a word 13-gram with a document of the public or hidden perplexity
#: blocks: per source file, the indices of those rows as `load_rows` reads the file.
PPL_OVERLAP = HERE / "perplexity_overlap.json"


def read_documents(paths) -> list[str]:
    """The distinct documents of build_fidelity_assets.py's `documents.jsonl` files."""
    seen, out = set(), []
    for path in sorted(Path(p) for p in paths):
        for line in path.read_text(encoding="utf-8").splitlines():
            doc = json.loads(line)
            if (doc["shard"], doc["id"]) not in seen:
                seen.add((doc["shard"], doc["id"]))
                out.append(doc["text"])
    return out


def ppl_overlap() -> dict:
    return json.loads(PPL_OVERLAP.read_text())


def same_filter(manifest: dict, keys, pinned: dict) -> bool:
    """Were the staged datasets filtered with these benchmark signatures and these overlap rows?"""
    return (manifest.get("filter_signatures_sha256") == benchmark_guard.fingerprint(keys)
            and manifest.get("ppl_overlap_rows") == pinned["rows"])


def as_text_rows(rows: list[dict]) -> list[dict]:
    """Rows as the overlap filter reads them: a `text`, and a `url` where there is one."""
    return [{"text": r.get("text") or " ".join(str(v) for v in r.values()), "url": r.get("url")} for r in rows]


def load_rows(name: str, file: str):
    """One source file as a pyarrow table (the first ROW_LIMITS rows, the COLUMNS kept)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    repo, revision = ALL_SOURCES[name][:2]
    local = hf_hub_download(repo, file, repo_type="dataset", revision=revision)
    if file.endswith(".jsonl"):
        with open(local, encoding="utf-8") as handle:
            return pa.Table.from_pylist([json.loads(line) for line in handle if line.strip()])
    if name in ROW_LIMITS:
        batches, taken = [], 0
        for batch in pq.ParquetFile(local).iter_batches(batch_size=10_000, columns=COLUMNS.get(name)):
            batches.append(batch.slice(0, ROW_LIMITS[name] - taken))
            taken += batches[-1].num_rows
            if taken >= ROW_LIMITS[name]:
                break
        return pa.Table.from_batches(batches)
    return pq.read_table(local, columns=COLUMNS.get(name))


def pin_perplexity_overlap(document_files) -> dict:
    """Write PPL_OVERLAP: every training row sharing a word 13-gram with one of the documents."""
    texts = read_documents(document_files)
    held_overlap = C4Overlap(documents=[(None, t) for t in texts])
    rows_out = {}
    for name, (_, _, files, _, _) in SOURCES.items():
        for file in files:
            rows = load_rows(name, file).to_pylist()
            held = held_overlap.overlapping(as_text_rows(rows))
            rows_out[f"{name}/{file}"] = [i for i, h in enumerate(held) if h]
            print(f"{name}/{file}: {sum(held)} rows overlap", flush=True)
    pinned = {"documents": len(texts), "ngram": NGRAM, "rows": rows_out}
    write_overlap(pinned)
    return pinned


def write_overlap(pinned: dict) -> None:
    """PPL_OVERLAP with one source file per line."""
    lines = [f"  {json.dumps(key)}: {json.dumps(rows)}" for key, rows in pinned["rows"].items()]
    PPL_OVERLAP.write_text(f'{{\n "documents": {pinned["documents"]},\n "ngram": {pinned["ngram"]},\n'
                           f' "rows": {{\n' + ",\n".join(lines) + "\n }\n}\n")


def prepare() -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq
    keys = benchmark_guard.signatures(benchmark_guard.benchmark_questions())
    pinned = ppl_overlap()
    manifest = {"kind": "compression_bench_training_datasets", "schema_version": 1,
                "filter": "rows holding a question of the hidden evaluation inputs (first ten words "
                          "as a word 10-gram) removed; rows sharing a word 13-gram with a document of "
                          "the hidden perplexity blocks removed",
                "filter_signatures_sha256": benchmark_guard.fingerprint(keys),
                "ppl_overlap_rows": pinned["rows"],
                "datasets": {}}
    stale = sorted(p.name for p in OUT.iterdir() if p.is_dir() and p.name not in ALL_SOURCES) if OUT.is_dir() else []
    if stale:
        # The image copies the whole directory: an unlisted dataset would ship.
        raise SystemExit(f"unlisted dataset directories in {OUT}: {stale}; move them out of the build context")
    previous = {}
    if (OUT / "DATASETS.json").is_file():
        staged = json.loads((OUT / "DATASETS.json").read_text())
        if same_filter(staged, keys, pinned):
            previous = staged.get("datasets", {})
    overlap = None
    for name, (repo, revision, files, license_, _) in ALL_SOURCES.items():
        old = previous.get(name)
        if (old and old.get("source") == repo and old.get("revision") == revision
                and [f["source_file"] for f in old["files"]] == files
                and all((OUT / name / f["file"]).is_file() and sha256(OUT / name / f["file"]) == f["sha256"]
                        for f in old["files"])):
            manifest["datasets"][name] = {**old, "use": "check" if name in CHECK_SOURCES else "train"}
            print(f"{name}: staged copy verified, kept", flush=True)
            continue
        target = OUT / name
        target.mkdir(parents=True, exist_ok=True)
        entry = {"source": repo, "revision": revision, "license": license_,
                 "use": "check" if name in CHECK_SOURCES else "train", "files": []}
        for file in files:
            table = load_rows(name, file)
            if name in ROW_LIMITS:
                entry["rows_taken"] = f"the first {table.num_rows} rows"
            rows = table.to_pylist()
            benchmark = [benchmark_guard.matches(json.dumps(row, ensure_ascii=False), keys) for row in rows]
            c4 = [False] * len(rows)
            if name in C4_FILTERED:
                overlap = overlap or C4Overlap(cache=Path.home() / ".cache" / "compression-bench-c4")
                c4 = overlap.overlapping(rows)
                entry["c4_overlap"] = overlap.describe()
            held = [False] * len(rows)
            if name not in CHECK_SOURCES:
                ruled_out = set(pinned["rows"][f"{name}/{file}"])
                held = [i in ruled_out for i in range(len(rows))]
                entry["ppl_overlap"] = {"documents": pinned["documents"], "ngram": pinned["ngram"]}
            kept = table.filter(pa.array([not (b or c or h) for b, c, h in zip(benchmark, c4, held)]))
            path = target / (Path(file).stem + ".parquet")
            pq.write_table(kept, path)
            entry["files"].append({"source_file": file, "file": path.name, "rows": kept.num_rows,
                                   "dropped": len(rows) - kept.num_rows,
                                   "dropped_benchmark": sum(benchmark), "dropped_c4_overlap": sum(c4),
                                   "dropped_ppl_overlap": sum(held),
                                   "sha256": sha256(path)})
            print(f"{name}/{path.name}: {kept.num_rows} rows, {len(rows) - kept.num_rows} dropped "
                  f"({sum(benchmark)} benchmark, {sum(c4)} C4 overlap, {sum(held)} perplexity-text overlap)", flush=True)
        manifest["datasets"][name] = entry
    (OUT / "DATASETS.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify(filter_too: bool = False) -> int:
    manifest = json.loads((OUT / "DATASETS.json").read_text())
    if filter_too:
        keys = benchmark_guard.signatures(benchmark_guard.benchmark_questions())
        if not same_filter(manifest, keys, ppl_overlap()):
            print("STAGED WITH ANOTHER BENCHMARK FILTER")
            return 1
    bad = [f"{name}/{f['file']}" for name, entry in manifest["datasets"].items() for f in entry["files"]
           if not (OUT / name / f["file"]).is_file() or sha256(OUT / name / f["file"]) != f["sha256"]]
    for name in bad:
        print(f"MISSING OR CHANGED {name}")
    return 1 if bad else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--pin-perplexity-overlap", nargs="+", metavar="DOCUMENTS_JSONL",
                        help="write perplexity_overlap.json from the perplexity documents (public and hidden)")
    args = parser.parse_args(argv)
    if args.pin_perplexity_overlap:
        pin_perplexity_overlap(args.pin_perplexity_overlap)
        return 0
    if args.verify:
        return verify()
    staged = json.loads((OUT / "DATASETS.json").read_text()) if (OUT / "DATASETS.json").is_file() else {}
    if set(staged.get("datasets", {})) == set(ALL_SOURCES) and verify(filter_too=True) == 0:
        print("datasets staged and verified")
        return 0
    prepare()
    return verify()


if __name__ == "__main__":
    sys.exit(main())
