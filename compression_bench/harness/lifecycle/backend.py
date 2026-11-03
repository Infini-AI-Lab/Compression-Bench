"""The one seam to the container: `docker exec` by exact container name.

Harbor composes and starts the environment (`scripts/agent/harbor_env.py`);
after that the harness talks to the main container directly, as ServerBench
does. Nothing here matches containers by pattern.
"""
from __future__ import annotations

import os
import shlex
import subprocess

#: main runs as the host user; the agent CLI refuses to bypass permissions as
#: root, and files it writes into the run's bind mounts stay the operator's.
AGENT_HOME = "/app/work/.agent-home"
WORKDIR = "/app/work"
CLI_DIR = "/opt/compression_bench/bin"


def host_user() -> str:
    return f"{os.getuid()}:{os.getgid()}"


def exec_argv(container: str, command: str, *, workdir: str = WORKDIR,
              interactive: bool = False) -> list[str]:
    """`docker exec` running `command` under bash as the agent user, with
    HOME the agent's home and the mounted CLI on PATH."""
    argv = ["docker", "exec"]
    if interactive:
        argv.append("-i")
    argv += ["-u", host_user(), "-w", workdir, "-e", f"HOME={AGENT_HOME}",
             container, "bash", "-c", f"export PATH={CLI_DIR}:$PATH; {command}"]
    return argv


def run(container: str, command: str, *, timeout: float = 120.0,
        workdir: str = WORKDIR, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        exec_argv(container, command, workdir=workdir, interactive=stdin is not None),
        input=stdin, capture_output=True, text=True, timeout=timeout, check=False)


def put_file(container: str, text: str, path: str) -> None:
    """Write `text` to `path` in the container over stdin (no shell quoting
    of the content, so nothing in it can end the command early)."""
    got = run(container, f"umask 077; cat > {shlex.quote(path)}", stdin=text,
              timeout=60.0, workdir="/")
    if got.returncode != 0:
        raise RuntimeError(f"could not write {path} in {container}: "
                           f"{(got.stderr or got.stdout)[-400:]}")


def pause(container: str) -> bool:
    """Freeze every process of the container (`docker pause`); its files, GPU memory
    and network stay."""
    got = subprocess.run(["docker", "pause", container], capture_output=True, text=True,
                         timeout=120, check=False)
    return got.returncode == 0


def unpause(container: str) -> bool:
    got = subprocess.run(["docker", "unpause", container], capture_output=True, text=True,
                         timeout=120, check=False)
    return got.returncode == 0


def running(container: str) -> bool:
    got = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", container],
        capture_output=True, text=True, timeout=30, check=False)
    return got.returncode == 0 and got.stdout.strip() == "true"


def project_containers(project: str) -> list[str]:
    """Every container, running or not, of one Compose project, by label."""
    got = subprocess.run(
        ["docker", "ps", "-a", "--filter",
         f"label=com.docker.compose.project={project}",
         "--format", "{{.Names}}"],
        capture_output=True, text=True, timeout=30, check=False)
    return [name for name in got.stdout.split() if name]
