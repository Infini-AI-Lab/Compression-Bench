"""Load the agent-facing specification. This module composes no prose.

As in ServerBench, every word an agent reads lives in the task's own
`instruction.md`, rendered by `compression_tasks/_manage/taskctl.py` from the
frozen contract. The harness reads that file and delivers it; it does not write,
extend, wrap or annotate it.

The specification is the document up to `## Continuation`: it is the first
prompt and `work/AGENT_TASK.md`. The sections from there on are what a later
launch of the same run is told, selected by name, with this run's two numbers
substituted. The run's own facts travel as data: `WORKSPACE.json` records the
deadline and `./clock` reports the time left.
"""
from __future__ import annotations

from pathlib import Path

BRIEF_NAME = "AGENT_TASK.md"

#: The session sections `agent.py` sends when it launches into an existing run.
SESSION_HEADINGS = {
    "continuation": "## Continuation",
    "idle-restart": "## Idle restart",
    "cold-restart": "## Cold restart",
}

#: Where the delivered specification ends.
FIRST_SESSION_HEADING = "## Continuation"

PLACEHOLDERS = ("{budget_hours}", "{deadline_utc}")


def _read(task_dir: Path | str) -> tuple[Path, str]:
    path = Path(task_dir) / "instruction.md"
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"{path} is empty")
    return path, text


def brief(task_dir: Path | str) -> str:
    """The specification, without the session sections."""
    path, text = _read(task_dir)
    if FIRST_SESSION_HEADING not in text:
        raise ValueError(f"{path} has no session sections; run `taskctl render`")
    document = text.partition(FIRST_SESSION_HEADING)[0].rstrip() + "\n"
    leftover = [token for token in PLACEHOLDERS if token in document]
    if leftover:
        raise ValueError(f"{path} leaves {leftover} in the specification itself")
    return document


def session_prompt(task_dir: Path | str, kind: str, *, budget_seconds: float,
                   deadline_utc: str) -> str:
    """One session section's body, in the task's own words."""
    if kind not in SESSION_HEADINGS:
        raise KeyError(f"no session section named {kind!r}")
    path, text = _read(task_dir)
    heading = SESSION_HEADINGS[kind]
    if heading not in text:
        raise ValueError(f"{path} has no `{heading}` section; run `taskctl render`")
    after = text.partition(heading)[2]
    for other in SESSION_HEADINGS.values():
        if other != heading and other in after:
            after = after.partition(other)[0]
    return (after.strip()
            .replace("{budget_hours}", f"{budget_seconds / 3600.0:.2f}")
            .replace("{deadline_utc}", str(deadline_utc)) + "\n")
