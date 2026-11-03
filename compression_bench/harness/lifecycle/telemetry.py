"""What an evolve run used and when: the agent's memory, resources and timing.

Three records, beside the trajectory (`trajectory.py`), for every run:

* the agent's memory -- the notes its CLI keeps for itself (Claude Code's
  project memory under `.claude/projects/*/memory/`, Codex's `memories/`) and
  any memory file it writes in the workspace (`memory.md`, `MEMORY/`,
  `CLAUDE.md`, `AGENTS.md`). Copied into each version's record when the agent
  takes a version, and into the trajectory at the end.
* resources -- one JSON line per sample while the agent runs
  (`<run>/resources.jsonl`): the run's GPU (memory used, utilisation, every
  compute process on it), each container of the run's project (memory, CPU),
  and the disk (free bytes; the submission's and, less often, the workspace's
  size). Sampling never stops a run: a failed probe is recorded as one.
* timing -- `<run>/TIMING.json`, assembled after the verdict: the run's phases
  from the journal, every agent launch, every public check, every stage of the
  hidden verdict (`verifier/timing.json`) and the time spent per tool.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from pathlib import Path

from . import record
from .utils import read_json, read_jsonl, seconds_between, until_space, utc_now

RESOURCES = "resources.jsonl"
TIMING = "TIMING.json"
SAMPLE_SECONDS = 60.0
WORKSPACE_SIZE_EVERY = 5            # samples between walks of the workspace

AGENT_HOME = Path("work") / ".agent-home"
MEMORY_GLOBS = (".claude/projects/*/memory/**/*", ".claude/CLAUDE.md", ".codex/memories/**/*",
                ".codex/AGENTS.md")
WORKSPACE_MEMORY = ("memory.md", "MEMORY.md", "CLAUDE.md", "AGENTS.md")


# --- the agent's memory ---------------------------------------------------------

def memory_files(root: Path) -> list[tuple[Path, str]]:
    """(file, its name in a memory record) for every memory note the agent keeps."""
    root = Path(root)
    found: list[tuple[Path, str]] = []
    home = root / AGENT_HOME
    for pattern in MEMORY_GLOBS:
        for path in sorted(home.glob(pattern)):
            if path.is_file():
                found.append((path, "agent-home/" + path.relative_to(home).as_posix()))
    work = root / "work"
    for name in WORKSPACE_MEMORY:
        if (work / name).is_file():
            found.append((work / name, "work/" + name))
    for path in sorted((work / "MEMORY").rglob("*")) if (work / "MEMORY").is_dir() else ():
        if path.is_file():
            found.append((path, "work/" + path.relative_to(work).as_posix()))
    return found


def copy_memory(root: Path, dest: Path) -> dict:
    """Copy the agent's memory notes into `dest`; what was copied."""
    files = memory_files(root)
    total = 0
    for source, name in files:
        target = Path(dest) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        total += target.stat().st_size
    return {"files": len(files), "bytes": total, "names": [name for _, name in files]}


# --- resources ------------------------------------------------------------------

def _run(argv: list[str], timeout: float = 30.0) -> str:
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                          check=True).stdout


def gpu_sample(index: int | str) -> dict:
    line = _run(["nvidia-smi", f"--id={index}", "--query-gpu=uuid,memory.used,memory.total,"
                 "utilization.gpu", "--format=csv,noheader,nounits"]).strip().splitlines()[0]
    uuid, used, total, util = [part.strip() for part in line.split(",")]
    apps = []
    for row in _run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                     "--format=csv,noheader,nounits"]).strip().splitlines():
        parts = [part.strip() for part in row.split(",")]
        if len(parts) == 3 and parts[0] == uuid:
            apps.append({"pid": int(parts[1]), "memory_mib": int(parts[2])})
    return {"index": int(index), "memory_used_mib": int(used), "memory_total_mib": int(total),
            "utilization_percent": int(util), "processes": apps}


def _bytes(text: str) -> int | None:
    units = {"b": 1, "kib": 2**10, "mib": 2**20, "gib": 2**30, "tib": 2**40,
             "kb": 10**3, "mb": 10**6, "gb": 10**9, "tb": 10**12}
    text = text.strip().lower()
    for unit in sorted(units, key=len, reverse=True):
        if text.endswith(unit):
            try:
                return int(float(text[: -len(unit)]) * units[unit])
            except ValueError:
                return None
    return None


def container_samples(project: str) -> list[dict]:
    names = _run(["docker", "ps", "--filter", f"label=com.docker.compose.project={project}",
                  "--format", "{{.Names}}"]).split()
    if not names:
        return []
    out = []
    for line in _run(["docker", "stats", "--no-stream", "--format", "{{json .}}", *names],
                     timeout=60).splitlines():
        row = json.loads(line)
        used, _, limit = row.get("MemUsage", "").partition("/")
        out.append({"name": row.get("Name"), "memory_bytes": _bytes(used),
                    "memory_limit_bytes": _bytes(limit), "cpu_percent": row.get("CPUPerc")})
    return out


def tree_bytes(path: Path) -> int:
    total = 0
    for directory, _, names in os.walk(path):
        for name in names:
            try:
                total += os.lstat(os.path.join(directory, name)).st_size
            except OSError:
                continue
    return total


class ResourceSampler:
    """Appends one sample to `<run>/resources.jsonl` every SAMPLE_SECONDS, from a
    background thread between `start()` and `stop()`."""

    def __init__(self, root: Path, *, gpu: int | str, project: str, every: float = SAMPLE_SECONDS):
        self.root, self.gpu, self.project, self.every = Path(root), gpu, project, every
        self.count = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        def loop():
            while not self._stop.is_set():
                try:
                    self.sample()
                except Exception:                         # noqa: BLE001 - sampling never stops a run
                    pass
                self._stop.wait(self.every)
        self._thread = threading.Thread(target=loop, name="compression-bench-resources", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=120)
            self._thread = None

    def sample(self) -> dict:
        row: dict = {"utc": utc_now()}
        for key, probe in (("gpu", lambda: gpu_sample(self.gpu)),
                           ("containers", lambda: container_samples(self.project))):
            try:
                row[key] = probe()
            except Exception as exc:                      # noqa: BLE001 - recorded, never fatal
                row[key + "_error"] = f"{type(exc).__name__}: {exc}"[:300]
        try:
            row["disk_free_bytes"] = shutil.disk_usage(self.root).free
            row["submission_bytes"] = tree_bytes(self.root / "submission")
            if self.count % WORKSPACE_SIZE_EVERY == 0:
                row["work_bytes"] = tree_bytes(self.root / "work")
        except OSError as exc:
            row["disk_error"] = str(exc)[:300]
        self.count += 1
        def append():
            with (self.root / RESOURCES).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
        until_space(append, "sampling resources")
        return row


def resource_summary(root: Path) -> dict:
    rows = read_jsonl(Path(root) / RESOURCES)
    gpu = [r["gpu"] for r in rows if isinstance(r.get("gpu"), dict)]
    mem = {}
    for r in rows:
        for c in r.get("containers") or []:
            if isinstance(c.get("memory_bytes"), int):
                mem[c["name"]] = max(mem.get(c["name"], 0), c["memory_bytes"])
    work = [r["work_bytes"] for r in rows if isinstance(r.get("work_bytes"), int)]
    return {"samples": len(rows), "every_seconds": SAMPLE_SECONDS,
            "first_utc": rows[0]["utc"] if rows else None, "last_utc": rows[-1]["utc"] if rows else None,
            "gpu_memory_peak_mib": max((g["memory_used_mib"] for g in gpu), default=None),
            "gpu_memory_total_mib": gpu[0]["memory_total_mib"] if gpu else None,
            "gpu_utilization_mean_percent": (round(sum(g["utilization_percent"] for g in gpu) / len(gpu), 1)
                                             if gpu else None),
            "gpu_busy_fraction": (round(sum(g["utilization_percent"] > 0 for g in gpu) / len(gpu), 3)
                                  if gpu else None),
            "container_memory_peak_bytes": mem,
            "workspace_peak_bytes": max(work, default=None),
            "disk_free_min_bytes": min((r["disk_free_bytes"] for r in rows
                                        if isinstance(r.get("disk_free_bytes"), int)), default=None),
            "probe_errors": sum(1 for r in rows if any(k.endswith("_error") for k in r))}


# --- timing ---------------------------------------------------------------------

def timing(root: Path, trajectory_summary: dict | None = None) -> dict:
    """Every phase, launch, public check and verdict stage, with its duration."""
    root = Path(root)
    events = list(record.Journal(root))

    def first(name):
        return next((e.get("utc") for e in events if e.get("event") == name), None)

    def last(name):
        return next((e.get("utc") for e in reversed(events) if e.get("event") == name), None)

    created, launched = first("run_created"), first("agent_launched")
    ended = last("agent_phase_ended")
    # The run ends at its terminal state; what a later sweep appends (harvest, cleaning) is not the run's time.
    done = next((e.get("utc") for e in reversed(events)
                 if e.get("event") == "state" and e.get("to") in record.TERMINAL), None)
    # The Verifier's own start, or the driver's when validate_evolve ended before it.
    verifier = (first("verifier_started") or first("final_validation_started"),
                last("verifier_finished"))
    phases = {
        "setup_seconds": seconds_between(created, launched),
        "agent_seconds": seconds_between(launched, ended),
        "stop_to_verdict_seconds": seconds_between(ended, verifier[0]),
        "verdict_seconds": seconds_between(*verifier),
        "total_seconds": seconds_between(created, done or (events[-1].get("utc") if events else None)),
    }
    launches = [{"kind": e.get("kind"), "seconds": e.get("seconds"), "tool_calls": e.get("tool_calls"),
                 "exit_code": e.get("exit_code"), "finished_utc": e.get("utc")}
                for e in events if e.get("event") == "agent_finished"]
    checks = []
    for row in read_jsonl(root / "public-checks" / "checks.jsonl"):
        response = row.get("response") or {}
        checks.append({"utc": row.get("utc_start"), "action": (row.get("request") or {}).get("action"),
                       "seconds": row.get("seconds"), "valid": response.get("valid"),
                       "stage_seconds": response.get("stage_seconds")})
    stages = (read_json(root / "verifier" / "timing.json", {}) or {}).get("stages", [])
    full = [c for c in checks if c["action"] == "full-check" and isinstance(c["seconds"], (int, float))]
    tools = (trajectory_summary or {}).get("tools") or {}
    return {"kind": "compression_bench_timing", "schema_version": 1, "phases": phases, "launches": launches,
            "public_checks": checks,
            "public_check_totals": {"count": len(checks), "full": len(full),
                                    "full_seconds": round(sum(c["seconds"] for c in full), 1)},
            "verdict_stages": stages,
            "tool_seconds": {name: {k: row.get(k) for k in ("calls", "seconds", "max_seconds")}
                             for name, row in sorted(tools.items())}}


def write_timing(root: Path, trajectory_summary: dict | None = None) -> dict:
    data = timing(root, trajectory_summary)
    until_space(lambda: (Path(root) / TIMING).write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8"),
                "writing TIMING.json")
    return data
