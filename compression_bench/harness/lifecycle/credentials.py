"""The operator's agent credentials, at an inode this run owns.

As in ServerBench, the credentials reach the container as a bind of ONE FILE, and a
Docker file bind resolves to the file's inode when the container starts. Claude
Code refreshes its OAuth token by atomic replace (new inode) and the refresh
revokes the token it replaced, so a bind of the host's own file would leave the
agent holding a revoked token the first time anything else on the host
refreshed.

`stage()` copies the credentials to `<run>/secrets/<name>`, and that copy is
what gets bound read-only. `refresh()` rewrites it IN PLACE (open r+, write,
truncate) whenever the host's file changes, so the container, holding exactly
that inode, sees the new bytes on its next read. The value is never logged.
"""
from __future__ import annotations

import os
from pathlib import Path

from .backend import AGENT_HOME

SECRETS_DIR = "secrets"

#: Where each CLI keeps its login on the host.
SOURCES = {
    "claude": "~/.claude/.credentials.json",
    "codex": "~/.codex/auth.json",
}

#: Where the staged copy is bound inside main ($HOME is the agent's home).
IN_CONTAINER = {
    "claude": f"{AGENT_HOME}/.claude/.credentials.json",
    "codex": f"{AGENT_HOME}/.codex/auth.json",
}


def source_path(agent_cli: str) -> Path:
    return Path(SOURCES[agent_cli]).expanduser()


def stage(credentials: Path | str, run_root: Path | str) -> Path | None:
    """Copy the credentials into the run and return the copy, or None when
    there is nothing to copy (a host with no login, or an API-key route)."""
    credentials = Path(credentials)
    if not credentials.is_file():
        return None
    secrets = Path(run_root) / SECRETS_DIR
    secrets.mkdir(parents=True, exist_ok=True)
    secrets.chmod(0o700)
    staged = secrets / credentials.name
    staged.write_bytes(credentials.read_bytes())
    staged.chmod(0o600)
    return staged


def refresh(staged: Path | str, source: Path | str) -> bool:
    """Rewrite the staged copy in place when the host's token moved on.

    True when the bytes changed. Every failure is False rather than an
    exception: this runs on a timer inside a live run, and a transient unlink
    while the host is mid-rename must not end it.
    """
    staged, source = Path(staged), Path(source)
    try:
        fresh = source.read_bytes()
    except OSError:
        return False
    if not fresh.strip():
        return False
    try:
        with staged.open("r+b") as handle:
            if handle.read() == fresh:
                return False
            handle.seek(0)
            handle.write(fresh)
            handle.truncate()
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        return False
    return True


def placeholder(run_root: Path | str, agent_cli: str) -> Path:
    """Create the mount point inside the run's `work/` before the container
    starts. Docker would otherwise create it as a root-owned path inside the
    agent's home. The placeholder stays empty; the secret is the bind."""
    relative = IN_CONTAINER[agent_cli][len("/app/work/"):]
    target = Path(run_root) / "work" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.touch(mode=0o600)
    return target
