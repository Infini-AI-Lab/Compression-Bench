"""Free space where a run writes: the Docker root (every container's own
filesystem, `/tmp` included) and the run root (workspace, submission, checkpoints).

A full Docker root fails every container on the host at once (`No space left on
device` in the agent's prompt staging or in the Verifier's compile cache), so a run
refuses to start, and a checkpoint verdict refuses to begin, below these floors.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

GIB = 2 ** 30
#: Floors in GiB; COMPRESSION_BENCH_MIN_FREE_DOCKER_GIB / COMPRESSION_BENCH_MIN_FREE_RUN_GIB override.
MIN_FREE_DOCKER_GIB = 60.0
MIN_FREE_RUN_GIB = 100.0
MIN_FREE_VERDICT_DOCKER_GIB = 30.0


def docker_root() -> Path | None:
    try:
        got = subprocess.run(["docker", "info", "--format", "{{.DockerRootDir}}"],
                             capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    root = got.stdout.strip()
    return Path(root) if got.returncode == 0 and root else None


def free_gib(path: Path | str) -> float | None:
    try:
        return shutil.disk_usage(path).free / GIB
    except OSError:
        return None


def floor(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def check(run_root: Path | str, *, docker_floor: float | None = None,
          run_floor: float | None = None) -> dict:
    """Free GiB at the Docker root and the run root, and whether both clear their floors."""
    docker_floor = floor("COMPRESSION_BENCH_MIN_FREE_DOCKER_GIB", MIN_FREE_DOCKER_GIB) \
        if docker_floor is None else docker_floor
    run_floor = floor("COMPRESSION_BENCH_MIN_FREE_RUN_GIB", MIN_FREE_RUN_GIB) if run_floor is None else run_floor
    root = docker_root()
    # The Docker root is root-owned; its mount point's usage is readable from its parent.
    probe = root if root is not None and root.exists() else (root.parent if root else None)
    docker_free = free_gib(probe) if probe is not None else None
    run_free = free_gib(Path(run_root))
    problems = []
    if docker_free is not None and docker_free < docker_floor:
        problems.append(f"Docker root {root}: {docker_free:.1f} GiB free < {docker_floor:g}")
    if run_free is not None and run_free < run_floor:
        problems.append(f"run root {run_root}: {run_free:.1f} GiB free < {run_floor:g}")
    return {"docker_root": str(root) if root else None, "docker_free_gib": docker_free,
            "docker_floor_gib": docker_floor, "run_free_gib": run_free,
            "run_floor_gib": run_floor, "ok": not problems, "problems": problems}
