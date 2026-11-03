"""The run record: who this run is, what happened, and what state it is in.

Three things that are one thing, as in ServerBench:

* **Identity** is allocated once, atomically (a run id is claimed by creating
  its directory with ``exist_ok=False``), and never inferred afterwards.
  Dry runs and tests allocate in a separate namespace so they consume no
  production id.
* **The journal** is one append-only JSONL file per run, appended under an
  exclusive lock, flushed and fsynced. If it is not an event there, it did not
  happen.
* **The state machine** is the only thing allowed to move a run, and it moves
  it by journalling. A reader reconstructs the run from the journal alone.
"""
from __future__ import annotations

import dataclasses
import fcntl
import json
import os
import re
import secrets
from collections.abc import Iterator
from pathlib import Path

from .utils import REPO, read_jsonl, task_hash, until_space, utc_now, utc_stamp, write_json_atomically

#: Production runs live here; nothing else may.
PRODUCTION = "evolve"
#: Dry runs and tests. Same code, different tree.
SCRATCH = "evolve-scratch"

_RUN_ID = re.compile(r"^[0-9a-f]{8}-\d{8}T\d{6}Z-[0-9a-f]{6}$")


class IdentityError(RuntimeError):
    """A provenance question was answered with a guess. Refuse instead."""


@dataclasses.dataclass(frozen=True)
class RunIdentity:
    """The immutable half of a run, written once at creation."""

    run_id: str
    task_id: str
    task_hash: str
    lineage_id: str
    namespace: str
    created_at: str
    deadline_utc: str
    budget_hours: float
    root: Path

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["root"] = str(self.root)
        d["kind"] = "run_identity"
        d["schema_version"] = 1
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RunIdentity":
        return cls(run_id=d["run_id"], task_id=d["task_id"],
                   task_hash=d["task_hash"], lineage_id=d["lineage_id"],
                   namespace=d["namespace"], created_at=d["created_at"],
                   deadline_utc=d["deadline_utc"],
                   budget_hours=float(d["budget_hours"]), root=Path(d["root"]))

    @property
    def is_production(self) -> bool:
        return self.namespace == PRODUCTION

    @property
    def project(self) -> str:
        """The Docker Compose project this run's containers belong to.
        Lowercase: Compose lowercases project names, and the containers'
        labels carry the lowercased form."""
        return f"compression-bench-{self.run_id}".lower()


def evolve_space_root(repo: Path = REPO, namespace: str = PRODUCTION) -> Path:
    if namespace not in (PRODUCTION, SCRATCH):
        raise IdentityError(f"unknown namespace {namespace!r}")
    return Path(repo) / "compression_bench" / "evolve_space" / namespace


def allocate(task_id: str, *, lineage_id: str, deadline_utc: str,
             budget_hours: float, namespace: str = PRODUCTION,
             repo: Path = REPO) -> RunIdentity:
    """Claim a fresh run directory and write `identity.json` into it."""
    parent = evolve_space_root(repo, namespace) / task_id.lower()
    parent.mkdir(parents=True, exist_ok=True)
    short = task_hash(task_id)
    for _ in range(16):
        run_id = f"{short}-{utc_stamp()}-{secrets.token_hex(3)}"
        root = parent / run_id
        try:
            root.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        identity = RunIdentity(
            run_id=run_id, task_id=task_id, task_hash=short,
            lineage_id=lineage_id, namespace=namespace, created_at=utc_now(),
            deadline_utc=deadline_utc, budget_hours=float(budget_hours),
            root=root)
        write_json_atomically(root / "identity.json", identity.to_dict())
        return identity
    raise IdentityError("could not allocate a unique run id in 16 draws")


def load_identity(root: Path | str) -> RunIdentity:
    root = Path(root)
    data = json.loads((root / "identity.json").read_text(encoding="utf-8"))
    identity = RunIdentity.from_dict(data)
    if identity.root.resolve() != root.resolve():
        raise IdentityError(f"{root} holds the identity of {identity.root}")
    if not _RUN_ID.match(identity.run_id):
        raise IdentityError(f"malformed run id {identity.run_id!r}")
    return identity


class Journal:
    """Append-only, locked, fsynced JSONL. The run's only state record."""

    NAME = "journal.jsonl"

    def __init__(self, root: Path | str):
        self.path = Path(root) / self.NAME

    def append(self, event: str, **fields) -> dict:
        record = {"event": event, "utc": utc_now(), **fields}
        line = json.dumps(record, sort_keys=False, default=str) + "\n"
        data = line.encode("utf-8")
        retried = False

        def tail_state() -> str:
            """After a failed attempt: 'complete' if this exact line already ends the file,
            'torn' if the file ends mid-line, 'clean' otherwise."""
            try:
                size = self.path.stat().st_size
                with self.path.open("rb") as existing:
                    existing.seek(max(0, size - len(data)))
                    tail = existing.read()
            except OSError:
                return "clean"
            if tail == data:
                return "complete"
            return "clean" if tail.endswith(b"\n") or not tail else "torn"

        def write():
            nonlocal retried
            with self.path.open("a", encoding="utf-8") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                try:
                    state = tail_state()
                    if not retried and state == "complete":
                        state = "clean"          # an identical earlier line is a different event
                    retried = True
                    if state != "complete":
                        if state == "torn":
                            handle.write("\n")      # close the half line a full disk left behind
                        handle.write(line)
                    handle.flush()
                    os.fsync(handle.fileno())
                finally:
                    fcntl.flock(handle, fcntl.LOCK_UN)

        until_space(write, f"journalling {event}")
        return record

    def __iter__(self) -> Iterator[dict]:
        # A torn final line from a crash is not an event: read_jsonl skips it.
        return iter(read_jsonl(self.path))

    def events(self, event: str) -> list[dict]:
        return [record for record in self if record.get("event") == event]

    def last(self, event: str) -> dict | None:
        found = self.events(event)
        return found[-1] if found else None


# --- the run state machine --------------------------------------------------------

CREATED = "CREATED"
WORKSPACE_READY = "WORKSPACE_READY"
AGENT_RUNNING = "AGENT_RUNNING"
VERDICT = "VERDICT"
COMPLETED = "COMPLETED"
FAILED = "FAILED"
CANCELLED = "CANCELLED"

TERMINAL = frozenset({COMPLETED, FAILED, CANCELLED})

# The run's lifecycle and nothing else: why the agent phase ended is
# `agent_phase_ended.reason` alone, and the verdict is the Verifier's status alone
# (SCORED | ZERO | NOT_RUN, `FINAL.json`); no state repeats either.
TRANSITIONS: dict[str, frozenset[str]] = {
    CREATED: frozenset({WORKSPACE_READY, FAILED, CANCELLED}),
    WORKSPACE_READY: frozenset({AGENT_RUNNING, COMPLETED, FAILED, CANCELLED}),   # COMPLETED: a dry run
    AGENT_RUNNING: frozenset({VERDICT, COMPLETED, FAILED, CANCELLED}),           # COMPLETED: --no-verdict
    VERDICT: frozenset({COMPLETED, FAILED, CANCELLED}),
    COMPLETED: frozenset(),
    FAILED: frozenset(),
    CANCELLED: frozenset(),
}

class StateError(RuntimeError):
    """A transition the table does not allow."""


def current_state(journal: Journal) -> str:
    last = journal.last("state")
    return last["to"] if last else CREATED


def transition(journal: Journal, to: str, **why) -> dict:
    """Move the run, by journalling, or refuse."""
    now = current_state(journal)
    if to not in TRANSITIONS.get(now, frozenset()):
        raise StateError(f"run state {now} cannot move to {to}")
    return journal.append("state", **{"from": now, "to": to}, **why)


def lineage_id_for(task_id: str, slug: str) -> str:
    return f"{utc_stamp()}-{task_id.lower()}-{slug}"
