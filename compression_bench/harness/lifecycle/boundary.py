"""Versions: what the agent's submission was at each milestone it named.

ServerBench captures each version as a git tree. A compression submission is
tens of GiB of weights, so a version here has two halves:

* tracked, `compression_output/<lineage>/<lineage>_vN/`: `VERSION.json`,
  `MANIFEST.json` (every file's path, size and full SHA-256) and copies of the
  small code and configuration files;
* host-local, `compression_output/.objects/<lineage>/vN/` (gitignored): a
  TRUE snapshot of the whole submission made with `cp --reflink=always`, which
  on a reflink filesystem (XFS, Btrfs) is instant and shares blocks until the
  agent rewrites a file. When reflink is unavailable (EOPNOTSUPP, a different
  filesystem) or free space is below `COMPRESSION_BENCH_SNAPSHOT_MIN_FREE_GB`, the
  version is manifest-only and says so. Checkpoints are measured, not archived:
  every snapshot is kept while the run lives, each distinct one gets the full
  hidden verdict after the run (`final_validate/tools/verdict_checkpoints.py`),
  and `harvest.py` then
  deletes every measured one (`COMPRESSION_BENCH_KEEP_CHECKPOINTS`, default 0, keeps
  more). A version whose content equals the latest version's is not recorded:
  `./version` answers `unchanged`.

Every version keeps its metrics whatever happens to its weights: VERSION.json
holds the manifest identity, the weight bytes and file composition, and the
public Checker result for exactly that file listing (`public_check`, from the
Checker's own history in `<run>/public-checks/checks.jsonl`).

With a snapshot, the manifest is hashed from the snapshot, so it describes
exactly the captured state even while the agent keeps working. A lineage
remembers each digest with the (path, size, mtime) it was read at, so a later
version re-reads only files that changed. The final capture uses the same
rule, so "the final submission is version N" is a like-for-like comparison.

The agent asks with `/app/work/version "why"`; the harness never decides on a
clock. A request is a small JSON file in `work/.versions/`; the acknowledgement
is written next to it.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
from pathlib import Path

from .utils import read_json, read_jsonl, sha256_file, until_space, utc_now, write_json_atomically

COPY_LIMIT = 8 << 20
COPY_SUFFIXES = frozenset({".py", ".json", ".md", ".txt", ".yaml", ".yml",
                           ".jinja", ".sh", ".toml", ".cfg", ".ini"})
MAX_VERSIONS = 500
MAX_REQUEST_BYTES = 64 << 10
OBJECTS = ".objects"


def snapshot_min_free_bytes() -> int:
    return int(float(os.environ.get("COMPRESSION_BENCH_SNAPSHOT_MIN_FREE_GB", "100")) * 2**30)


def keep_checkpoints() -> int:
    """Measured checkpoints a lineage keeps after a run besides the unmeasured ones
    (default none: every checkpoint gets a full verdict, then its weights go)."""
    return max(0, int(os.environ.get("COMPRESSION_BENCH_KEEP_CHECKPOINTS", "0")))


WEIGHT_SUFFIXES = frozenset({".safetensors", ".gguf", ".bin", ".pt", ".pth", ".ckpt",
                             ".npy", ".npz", ".pkl", ".h5", ".onnx", ".msgpack", ".tensors"})


def composition_of(manifest: list[dict]) -> dict:
    """Bytes and file counts by suffix, and the weight bytes among them."""
    by_suffix: dict[str, dict] = {}
    for entry in manifest:
        if entry.get("type") != "file":
            continue
        suffix = Path(entry["path"]).suffix.lower() or "(none)"
        slot = by_suffix.setdefault(suffix, {"files": 0, "bytes": 0})
        slot["files"] += 1
        slot["bytes"] += int(entry.get("size") or 0)
    weights = sum(v["bytes"] for k, v in by_suffix.items() if k in WEIGHT_SUFFIXES)
    return {"weight_bytes": weights, "by_suffix": by_suffix}


#: What a version keeps of the public Checker's response to a check of its files.
PUBLIC_CHECK_FIELDS = ("valid", "error", "total_gib", "public_ppl", "public_bf16_ppl",
                       "public_reward", "decode_latency_seconds", "ppl_increase_percent")


def public_check_for(history: Path, manifest: list[dict], before: str = "~") -> dict | None:
    """The last formal full public check of exactly this file listing."""
    listing = sorted([e["path"], int(e["size"])] for e in manifest if e.get("type") == "file")
    if not listing:
        return None
    for row in reversed(read_jsonl(history)):
        request = row.get("request") if isinstance(row.get("request"), dict) else {}
        if (request.get("action") == "full-check" and request.get("mode", "formal") == "formal"
                and (row.get("utc_start") or "") <= before
                and sorted([str(p), int(n)] for p, n in row.get("submission") or []) == listing):
            response = row.get("response") or {}
            return {"utc": row.get("utc_start"), "seconds": row.get("seconds"),
                    **{k: response.get(k) for k in PUBLIC_CHECK_FIELDS if k in response}}
    return None


def reflink_snapshot(source: Path, target: Path) -> dict:
    """Snapshot `source` into `target` with block-sharing copies, or explain
    why not. Never a full byte copy: that would double tens of GiB."""
    target.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(target.parent).free
    if free < snapshot_min_free_bytes():
        return {"status": "skipped", "why": f"only {free / 2**30:.0f} GiB free on "
                f"the snapshot filesystem (minimum {snapshot_min_free_bytes() / 2**30:.0f})"}
    target.mkdir(exist_ok=False)
    got = subprocess.run(["cp", "-a", "--reflink=always", f"{source}/.", str(target)],
                         capture_output=True, text=True, check=False)
    if got.returncode != 0:
        shutil.rmtree(target, ignore_errors=True)
        return {"status": "none", "why": (got.stderr or got.stdout).strip()[-300:]
                or f"cp exited {got.returncode}"}
    return {"status": "reflink", "path": str(target)}


def listing(root: Path) -> list[dict]:
    """Every entry under `root`, sorted, without reading file contents."""
    entries = []
    if not root.is_dir():
        return entries
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            entries.append({"path": relative, "type": "symlink",
                            "target": os.readlink(path)})
        elif path.is_file():
            stat = path.stat()
            entries.append({"path": relative, "type": "file",
                            "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return entries


def hash_listing(root: Path, entries: list[dict],
                 cache: dict | None = None) -> list[dict]:
    """Add SHA-256 digests to a listing; mark files that changed while being
    read. `cache` maps (path, size, mtime_ns) to a digest already computed."""
    cache = cache if cache is not None else {}
    out = []
    for entry in entries:
        entry = dict(entry)
        if entry["type"] == "file":
            path = root / entry["path"]
            key = (entry["path"], entry["size"], entry["mtime_ns"])
            try:
                if key in cache:
                    entry["sha256"] = cache[key]
                else:
                    entry["sha256"] = sha256_file(path)
                    stat = path.stat()
                    if (stat.st_size, stat.st_mtime_ns) != (entry["size"], entry["mtime_ns"]):
                        entry["changed_during_capture"] = True
                    else:
                        cache[key] = entry["sha256"]
            except OSError as exc:
                entry["sha256"] = None
                entry["unreadable"] = str(exc)[:200]
        out.append(entry)
    return out


def identity_of(manifest: list[dict]) -> str:
    """One digest for a whole manifest, in its order: paths, types, sizes, file
    digests and link targets."""
    sha = hashlib.sha256()
    for entry in manifest:
        sha.update(json.dumps([entry["path"], entry["type"], entry.get("size"),
                               entry.get("sha256"), entry.get("target")]).encode())
    return sha.hexdigest()


class Lineage:
    """One (task, agent) ascent: its versions, in order, and its verdict."""

    def __init__(self, root: Path | str, lineage_id: str):
        self.lineage_id = lineage_id
        self.dir = Path(root) / lineage_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ledger = self.dir / "versions.jsonl"
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self._digests: dict = {}

    def versions(self) -> list[dict]:
        return read_jsonl(self.ledger)

    @property
    def objects(self) -> Path:
        return self.dir.parent / OBJECTS / self.lineage_id

    def capture(self, submission: Path, *, note: str, run_id: str,
                source: str, background: bool = True, snapshot: bool = True) -> dict:
        """Freeze the submission as the next version: snapshot, manifest, code."""
        with self._lock:
            index = len(self.versions()) + 1
            if index > MAX_VERSIONS:
                raise RuntimeError(f"lineage {self.lineage_id} already holds "
                                   f"{MAX_VERSIONS} versions")
            name = f"{self.lineage_id}_v{index}"
            target = self.dir / name
            target.mkdir(exist_ok=False)
            snap = (reflink_snapshot(submission, self.objects / f"v{index}")
                    if snapshot else {"status": "none", "why": "not requested"})
            frozen = Path(snap["path"]) if snap["status"] == "reflink" else submission
            entries = listing(frozen)
            copied = []
            for entry in entries:
                if (entry["type"] == "file" and entry["size"] <= COPY_LIMIT
                        and Path(entry["path"]).suffix in COPY_SUFFIXES):
                    destination = target / "files" / entry["path"]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.copyfile(frozen / entry["path"], destination)
                        destination.chmod(0o444)
                        copied.append(entry["path"])
                    except OSError:
                        continue
            record = {
                "kind": "compression_bench_version", "schema_version": 1,
                "version": f"v{index}", "name": name, "lineage_id": self.lineage_id,
                "run_id": run_id, "source": source, "note": note,
                "captured_utc": utc_now(), "files": sum(e["type"] == "file" for e in entries),
                "symlinks": sum(e["type"] == "symlink" for e in entries),
                "total_bytes": sum(e.get("size", 0) for e in entries),
                "copied_files": copied, "manifest": "MANIFEST.json",
                "manifest_status": "hashing",
                "snapshot": {**snap, "path": (str(Path(snap["path"]).relative_to(self.dir.parent))
                                              if snap.get("path") else None)},
            }
            write_json_atomically(target / "VERSION.json", record)
            ledger_line = json.dumps({k: record[k] for k in (
                "version", "name", "run_id", "source", "note", "captured_utc",
                "files", "total_bytes")} | {"snapshot": snap["status"]}) + "\n"

            def append_ledger():
                with self.ledger.open("a", encoding="utf-8") as handle:
                    handle.write(ledger_line)
            until_space(append_ledger, "recording the version")

        history = Path(submission).parent / "public-checks" / "checks.jsonl"

        def finish():
            manifest = hash_listing(frozen, entries, self._digests)
            write_json_atomically(target / "MANIFEST.json", manifest)
            done = dict(record, manifest_status="complete",
                        manifest_identity=identity_of(manifest),
                        changed_during_capture=sorted(
                            e["path"] for e in manifest
                            if e.get("changed_during_capture")),
                        composition=composition_of(manifest),
                        public_check=public_check_for(history, manifest,
                                                      before=record["captured_utc"]))
            write_json_atomically(target / "VERSION.json", done)

        if background:
            thread = threading.Thread(target=finish, daemon=True,
                                      name=f"capture-{name}")
            thread.start()
            self._threads.append(thread)
        else:
            finish()
        return record

    def unchanged_since(self, submission: Path) -> str | None:
        """The latest version, when `submission` now computes with exactly its files
        (interpreter caches and tokenizer files aside); else None. Digests come from
        the capture cache where nothing changed, so an unchanged submission costs
        no hashing."""
        from . import verdicts
        rows = self.versions()
        if not rows:
            return None
        manifest = read_json(self.dir / rows[-1]["name"] / "MANIFEST.json", None)
        if not manifest:
            return None
        previous = verdicts.content_identity(manifest)
        current = verdicts.content_identity(hash_listing(Path(submission), listing(Path(submission)),
                                                         self._digests))
        return rows[-1]["version"] if previous and previous == current else None

    def join(self, timeout: float | None = None) -> None:
        for thread in self._threads:
            thread.join(timeout)


def pending_requests(run_root: Path | str) -> list[Path]:
    """Version requests the agent wrote and the host has not acknowledged."""
    folder = Path(run_root) / "work" / ".versions"
    if not folder.is_dir():
        return []
    out = []
    for path in sorted(folder.glob("*.json")):
        if path.name.endswith(".ack.json") or path.name.startswith("."):
            continue
        if not path.with_name(path.stem + ".ack.json").exists():
            out.append(path)
    return out


def read_request(path: Path) -> dict:
    """The agent's request, bounded and validated; never trusted further."""
    if path.is_symlink() or path.stat().st_size > MAX_REQUEST_BYTES:
        return {"action": "invalid", "why": "request is a symlink or too large"}
    data = read_json(path, None)
    if not isinstance(data, dict) or data.get("action") != "publish":
        return {"action": "invalid", "why": "request is not a publish request"}
    return {"action": "publish", "note": str(data.get("note", ""))[:4000]}


def acknowledge(request: Path, payload: dict) -> None:
    write_json_atomically(request.with_name(request.stem + ".ack.json"), payload)
