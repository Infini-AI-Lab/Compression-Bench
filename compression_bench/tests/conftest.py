"""Shared fixtures for the compression-bench harness tests: no GPU, no Docker,
no network. The harness package root is put on sys.path the same way
`compression_bench/scripts/env.sh` does."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for path in (REPO / "compression_bench" / "harness", REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


CONTRACT = """\
kind: compression_task_contract
schema: compression_bench.task_contract/1
task_id: {task_id}
task_hash: "{task_hash}"
title: A test task
model:
  id: Org/Model-1B
  revision: 0123456789abcdef0123456789abcdef01234567
  local_name: Model-1B
capacity_gib: 4
budget:
  agent_hours: 8
  verifier_timeout_s: 1800
  public_check_timeout_s: 1800
debug_model: null
"""


INSTRUCTION = """\
# Compress Model-1B to 4 GiB

The specification.

## Continuation

Budget remaining {budget_hours} h against this run's deadline of {deadline_utc}.

## Idle restart

Idle. Budget remaining {budget_hours} h until {deadline_utc}.

## Cold restart

Cold. Budget remaining {budget_hours} h until {deadline_utc}.
"""


@pytest.fixture
def make_task(tmp_path):
    """A minimal task directory (contract + instruction) under tmp_path/tasks."""
    from lifecycle.utils import task_hash

    def make(task_id: str = "compress-test-4g", root: Path | None = None) -> Path:
        root = root or tmp_path / "tasks"
        task = root / task_id
        task.mkdir(parents=True)
        (task / "contract.yaml").write_text(
            CONTRACT.format(task_id=task_id, task_hash=task_hash(task_id)))
        (task / "instruction.md").write_text(INSTRUCTION)
        return task
    return make
