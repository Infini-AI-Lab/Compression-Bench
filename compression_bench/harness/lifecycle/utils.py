"""Small helpers every lifecycle module shares. Stdlib only."""
from __future__ import annotations

import datetime
import errno
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

#: The repository this checkout of the harness belongs to.
REPO = Path(__file__).resolve().parents[3]


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def utc_stamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ")


def utc_after(seconds: float) -> str:
    """The UTC time `seconds` from now, in the journal's format."""
    return (datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(value: str) -> datetime.datetime:
    return datetime.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=datetime.timezone.utc)


def seconds_between(start: str | None, end: str | None) -> float | None:
    """Seconds from one journal timestamp to another (0.1 s), or None if either is unknown."""
    if not start or not end:
        return None
    try:
        return round((parse_utc(end) - parse_utc(start)).total_seconds(), 1)
    except (ValueError, TypeError):
        return None


FULL_DISK = (errno.ENOSPC, errno.EDQUOT)
#: How long a driver write waits before it tries again on a full disk.
FULL_DISK_POLL_S = 30.0
#: How often a still-full disk repeats its message.
FULL_DISK_REMIND_S = 600.0


def until_space(operation, what: str, *, poll: float | None = None, sleep=None):
    """Run `operation()`; when the disk is full (ENOSPC/EDQUOT), PAUSE and retry it until it
    fits, instead of letting the driver die. A run is never failed or killed for want of
    space: it waits for whoever owns the disk to free some and continues by itself.
    `operation` must be safe to repeat (it is re-run from the start)."""
    poll = FULL_DISK_POLL_S if poll is None else poll
    sleep = sleep or time.sleep          # resolved per call, so a caller (or a test) can replace time.sleep
    waited, reminded = 0.0, 0.0
    while True:
        try:
            result = operation()
        except OSError as exc:
            if exc.errno not in FULL_DISK:
                raise
            if waited == 0.0 or reminded >= FULL_DISK_REMIND_S:
                print(f"[compression_bench] PAUSED {utc_now()}: no space left on device while {what}; "
                      f"the run waits and continues by itself once space is freed "
                      f"(paused {int(waited)}s so far)", file=sys.stderr, flush=True)
                reminded = 0.0
            sleep(poll)
            waited += poll
            reminded += poll
            continue
        if waited:
            print(f"[compression_bench] RESUMED {utc_now()}: {what} written after {int(waited)}s on a full disk",
                  file=sys.stderr, flush=True)
        return result


def write_json_atomically(path: Path | str, payload) -> Path:
    """Write JSON through a temporary file and a rename: never a torn file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def write():
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    until_space(write, f"writing {path.name}")
    return path


def read_json(path: Path | str, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def read_jsonl(path: Path | str) -> list:
    """Every parseable line of a JSONL file (a torn or blank line is skipped); [] if absent."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    rows = []
    for line in text.splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def sha256_file(path: Path | str, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def task_hash(task_id: str) -> str:
    """A task's short handle: the first 8 hex digits of its id's SHA-256."""
    return hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:8]


def read_contract(task_dir: Path | str) -> dict:
    """The task's `contract.yaml`, parsed. yaml is imported here, not at
    module scope, so the core stays importable in a bare interpreter."""
    import yaml

    path = Path(task_dir) / "contract.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data.get("task_id"):
        raise ValueError(f"{path} is not a task contract (no task_id)")
    return data


def agent_slug(model: str, effort: str = "") -> str:
    """`claude-opus-5` + `xhigh` -> `claude-opus-5-xhigh`; lowercase, safe."""
    raw = "-".join(part for part in (model or "default", effort) if part)
    safe = "".join(ch if ch.isalnum() else "-" for ch in raw.lower())
    while "--" in safe:
        safe = safe.replace("--", "-")
    return safe.strip("-") or "agent"
