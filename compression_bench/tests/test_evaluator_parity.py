"""Every evaluator copy of a task measures alike.

Each task carries two byte copies of `compression_bench/harness/final_validate/`:
the hidden Verifier's (`tests/`) and the public Checker's
(`environment/public-checker/`), each with the task's `task_policy.json`. This
test runs `parity_probe.py` against both copies of every live task on the same
inputs and requires identical output wherever both copies answer (the Checker
alone carries the public response and the debug profile), so a public check
measures exactly what the verdict measures. It also requires the measurements
themselves to work on the task's own inputs: package inspection, the private
evaluation view, perplexity input loading and arithmetic, and the smoke stage.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "compression_tasks"
PROBE = Path(__file__).with_name("parity_probe.py")
sys.path.insert(0, str(TASKS / "_manage"))
import sites  # noqa: E402

LIVE = sites.live_tasks()

#: Probe keys only the public Checker's copy answers.
CHECKER_ONLY = ("public/", "ppl-measure/debug-9b")
#: The measurements every copy must complete on the task's own inputs.
MEASUREMENTS = ("ppl-load/hidden", "ppl-load/public", "ppl-measure/hidden", "ppl-measure/public",
                "inspect/safetensors", "inspect/with-tokenizer", "view/safetensors", "smoke")
#: Packages and inputs every copy must refuse.
REFUSED = ("inspect/symlink", "inspect/too-large", "inspect/no-config",
           "ppl-load/hidden/wrong-split", "ppl-load/public/wrong-split")


def probe(task: str, side: str, scratch: Path) -> dict:
    """The probe's output for one copy; `scratch` is the same path for every copy
    compared, so file paths in error messages compare equal."""
    root = TASKS / task
    copy_dir = root / "tests" if side == "hidden" else root / "environment" / "public-checker"
    public = root / "environment" / "public-checker" / "evaluator-assets"
    hidden = root / "tests" / "evaluator-assets"
    scratch.mkdir(parents=True, exist_ok=True)
    done = subprocess.run([sys.executable, "-B", str(PROBE), str(copy_dir), str(public), str(hidden),
                           str(scratch)], capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-4000:]
    return json.loads(done.stdout)


def shared(output: dict) -> dict:
    return {k: v for k, v in output.items() if not k.startswith(CHECKER_ONLY)}


@pytest.mark.parametrize("task", LIVE)
def test_the_checker_measures_what_the_verifier_measures(task, tmp_path):
    hidden = probe(task, "hidden", tmp_path / "scratch")
    public = probe(task, "public", tmp_path / "scratch")
    assert set(hidden) == set(shared(public))
    mismatched = {k: (hidden[k], public[k]) for k in hidden if hidden[k] != public[k]}
    assert not mismatched, json.dumps(mismatched, indent=1, default=str)[:6000]
    for key in MEASUREMENTS:
        assert "ok" in hidden[key], (key, hidden[key])
    for key in REFUSED:
        assert "error" in hidden[key], (key, hidden[key])
    assert hidden["ppl-load/hidden"]["ok"]["blocks"] == hidden["ppl-load/public"]["ok"]["blocks"] == 256
    assert hidden["ppl-load/hidden"]["ok"]["first"] != hidden["ppl-load/public"]["ok"]["first"]
