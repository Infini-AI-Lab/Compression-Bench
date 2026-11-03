"""Helpers shared by the host-side study scripts: repository paths, JSON records, file digests, run and checkpoint
lookup, GPU claims and the `docker run` prefix of every study container.

Standard library only; the harness's GPU claims (compression_bench/harness/lifecycle/gpus.py) are imported on first use.
"""
from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[3]
HARNESS = REPO / "compression_bench/harness"
RUNS = REPO / "compression_bench/evolve_space/evolve"
TRAJECTORY_ANALYSIS = REPO / "evaluation_progress/trajectory_analysis"
QUANT_ERROR_DIR = REPO / "compression_bench/evolve_space/analysis/quant_error"   # scratch of studies/insights/


# --- records -----------------------------------------------------------------------------------------------------
def available(p) -> bool:
    """A JSON record exists, as written or as its portable `.gz` archive."""
    return Path(p).exists() or Path(str(p) + ".gz").exists()


def read_json(p):
    """A JSON record, from its `.gz` archive when only that is kept (archive_portable)."""
    p = Path(p)
    if p.exists():
        return json.loads(p.read_text())
    with gzip.open(str(p) + ".gz", "rt") as f:
        return json.load(f)


def write_json(p, value, ensure_ascii=True) -> None:
    """Write a JSON record atomically (indented, trailing newline), creating its directory."""
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=ensure_ascii) + "\n")
    tmp.replace(p)


def file_sha256(p) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def manifest(root) -> dict:
    """{relative path: {bytes, sha256}} of every file under root."""
    root = Path(root)
    return {str(p.relative_to(root)): {"bytes": p.stat().st_size, "sha256": file_sha256(p)}
            for p in sorted(root.rglob("*")) if p.is_file()}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def archive_portable(out: Path, **manifest_fields) -> None:
    """Gzip (losslessly, verified) every JSON/JSONL record of at least 256 KiB under out, git-ignore the originals, and
    write raw_archives.json and evidence_manifest.json (manifest_fields, then the remaining portable files' digests)."""
    ignored = ["*.pid", "*.tmp"]
    archives = []
    for path in sorted(out.rglob("*")):
        if not path.is_file() or path.suffix not in (".json", ".jsonl") or path.stat().st_size < 256 * 1024:
            continue
        original_sha = file_sha256(path)
        packed = Path(str(path) + ".gz")
        with path.open("rb") as src, packed.open("wb") as dest:
            with gzip.GzipFile(filename="", mode="wb", fileobj=dest, mtime=0) as gz:
                shutil.copyfileobj(src, gz)
        h = hashlib.sha256()
        with gzip.open(packed, "rb") as src:
            for block in iter(lambda: src.read(4 << 20), b""):
                h.update(block)
        assert h.hexdigest() == original_sha
        rel = str(path.relative_to(out))
        ignored.append("/" + rel)
        archives.append({"original": rel, "original_bytes": path.stat().st_size, "original_sha256": original_sha,
                         "archive": str(packed.relative_to(out)), "archive_bytes": packed.stat().st_size,
                         "archive_sha256": file_sha256(packed)})
    (out / ".gitignore").write_text("\n".join(ignored) + "\n")
    write_json(out / "raw_archives.json", archives)
    write_json(out / "evidence_manifest.json", {
        "created_utc": utc_now(), **manifest_fields,
        "portable_files": {str(p.relative_to(out)): {"bytes": p.stat().st_size, "sha256": file_sha256(p)}
                           for p in sorted(out.rglob("*")) if p.is_file() and p.name != "evidence_manifest.json"
                           and not p.name.endswith((".pid", ".tmp")) and "/" + str(p.relative_to(out)) not in ignored}})


# --- runs, tasks, checkpoints ------------------------------------------------------------------------------------
def run_dir(run_id: str) -> Path:
    """A recorded run's directory, compression_bench/evolve_space/evolve/<task>/<run id>."""
    return next(RUNS.glob(f"*/{run_id}"))


def model_dir(task: str) -> str:
    """The task's pinned BF16 checkpoint on this host (model_for_task.py)."""
    return subprocess.check_output([sys.executable, str(REPO / "compression_bench/scripts/model_for_task.py"),
                                    str(REPO / "compression_tasks" / task)], text=True).strip()


def limit_bytes(task: str) -> int:
    """The capacity limit a task id names (`<model>_<N>g_...`), in bytes."""
    return int(re.search(r"_(\d+)g_", task).group(1)) * 2**30


# --- GPUs and containers -----------------------------------------------------------------------------------------
def gpus():
    """The harness's GPU-claim module (lifecycle/gpus.py)."""
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    from lifecycle import gpus as module
    return module


def claim_gpu(run_id: str, wait_s: float | None = None):
    """(card, claim) for an idle card with no compute process, claimed under run_id. Without wait_s, refuse when no
    card is idle; with it, look again every wait_s seconds."""
    g = gpus()
    while True:
        gpu = g.pick()
        if gpu is not None:
            held = g.Claim(gpu, run_id=run_id)
            if held.acquire():
                if not g.sample()[gpu]["pids"]:
                    return gpu, held
                held.release()
        if wait_s is None:
            raise RuntimeError("no idle card; do not contend")
        time.sleep(wait_s)


def gpu_uuid(gpu: int) -> str:
    return gpus().sample()[gpu]["uuid"]


def gpu_hardware(gpu: int) -> str:
    """nvidia-smi's identity, memory, power limit and clocks of one card, as one CSV line."""
    return subprocess.check_output(["nvidia-smi", "--id", str(gpu), "--query-gpu=index,uuid,name,driver_version,"
                                    "memory.total,power.limit,clocks.max.sm,clocks.max.memory",
                                    "--format=csv,noheader"], text=True).strip()


def docker_run(*, name: str | None = None, gpu: int | None = None, cpus: str = "8", memory: str = "128g") -> list[str]:
    """The `docker run` prefix of a study container: removed on exit, no network, the caller's uid, HOME=/tmp, no
    bytecode. A GPU container (gpu given) also gets 16 GiB of shared memory and offline Hugging Face / vLLM settings."""
    cmd = ["docker", "run", "--rm"]
    if name:
        cmd += ["--name", name]
    if gpu is not None:
        cmd += ["--gpus", f"device={gpu}"]
    cmd += ["--network=none", "--cpus", cpus, "--memory", memory]
    if gpu is not None:
        cmd += ["--shm-size", "16g"]
    cmd += ["--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp", "-e", "PYTHONDONTWRITEBYTECODE=1"]
    if gpu is not None:
        cmd += ["-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1", "-e", "VLLM_NO_USAGE_STATS=1"]
    return cmd


def hf_cache_mounts(model) -> list[str]:
    """Read-only mounts that resolve a Hugging Face cache snapshot's links (into ../../blobs) inside a container."""
    model = Path(model)
    mounts = ["-v", f"{model.parent.parent}/blobs:/blobs:ro"]
    for parent in sorted({str(f.resolve().parent) for f in model.glob("*.safetensors")}):
        mounts += ["-v", f"{parent}:{parent}:ro"]
    return mounts


def run_container(cmd: list[str], log: Path, name: str, timeout: float):
    """Run a named container with its output in log; its return code, or 'timeout' (the container is then stopped)."""
    with Path(log).open("w") as f:
        try:
            return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "stop", "-t", "10", name], stdout=f, stderr=subprocess.STDOUT)
            return "timeout"
