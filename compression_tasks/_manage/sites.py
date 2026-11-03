"""Where task data lives, and the one site each settable parameter is written to.

A task's sources are `contract.yaml` and `assets/`; every other
file in its directory is generated (`render.py`). `taskctl set` writes a
parameter only through `SCALARS`: one YAML key under one parent in
`contract.yaml`, edited in place by `blocks.set_scalar` so comments and layout
survive, then re-rendered and read back.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASKS = Path(os.environ.get("COMPRESSION_BENCH_TASK_ROOT") or HERE.parent).resolve()
REPO = HERE.parent.parent

# Generated files are never edited by hand; `taskctl check` refuses drift.
GENERATED = ("task.toml", "instruction.md", "README.md", "MODE.md", "environment/", "tests/")
SOURCES = ("contract.yaml", "assets/")

# Runs and their evidence: renames and re-renders never rewrite these.
EVIDENCE = (REPO / "compression_bench" / "evolve_space", REPO / "compression_output",
            REPO / "evaluation_progress")


def live_tasks() -> list[str]:
    """A live task: a directory not starting with `_`, with a contract."""
    if not TASKS.is_dir():
        return []
    return sorted(p.name for p in TASKS.iterdir()
                  if p.is_dir() and not p.name.startswith("_") and (p / "contract.yaml").is_file())


def task_dir(task: str) -> Path:
    """Resolve a task by id or by its 8-hex hash."""
    import hashlib
    for name in live_tasks():
        if task in (name, hashlib.sha256(name.encode()).hexdigest()[:8]):
            return TASKS / name
    raise SystemExit(f"no live task {task!r} under {TASKS}")


@dataclass(frozen=True)
class Scalar:
    key: str
    indent: int
    under: str | None
    kind: type
    loosens_when: str | None = None   # "up": a larger value makes the task easier;
                                      # "change": any new value may (accepting more formats does)


SCALARS: dict[str, Scalar] = {
    "capacity_gib": Scalar("capacity_gib", 0, None, float, loosens_when="up"),
    "version": Scalar("version", 0, None, str),
    "description": Scalar("description", 0, None, str),
    "model_revision": Scalar("revision", 2, "model", str),
    "agent_hours": Scalar("agent_hours", 2, "budget", int, loosens_when="up"),
    "public_check_timeout_s": Scalar("public_check_timeout_s", 2, "budget", int),
    "verifier_timeout_s": Scalar("verifier_timeout_s", 2, "budget", int),
    "build_timeout_s": Scalar("build_timeout_s", 2, "budget", int),
    "starter": Scalar("starter", 0, None, str),
    "platform_patches": Scalar("patches", 2, "platform", list),
    "ppl_protocol": Scalar("ppl_protocol", 2, "evaluation", str),
    "payload_rule": Scalar("payload_rule", 2, "evaluation", str, loosens_when="change"),
    "link_rule": Scalar("link_rule", 2, "evaluation", str, loosens_when="change"),
}


def contract_value(contract: dict, name: str):
    site = SCALARS[name]
    holder = contract if site.under is None else contract[site.under]
    return holder[site.key]


def formatted(value, kind: type) -> str:
    """The YAML text for a scalar; strings are always quoted (JSON is valid YAML)."""
    import json
    if kind is str:
        return json.dumps(str(value))
    if kind is list:
        return json.dumps([str(v) for v in value])
    if kind is int:
        return str(int(value))
    number = float(value)
    return str(int(number)) if number.is_integer() else repr(number)
