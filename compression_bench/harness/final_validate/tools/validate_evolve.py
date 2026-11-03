#!/usr/bin/env python3
"""The final verdict of one evolve run: the hidden Verifier, once, host-side.

    validate_evolve.py --run <run root>

Called by `lifecycle/run.py` after the agent's deadline and after the agent
environment is down. Refuses unless no container of the run's project is
running and the submission lock can be taken exclusively (then releases it:
the Verifier's own `test.sh` takes it for the whole evaluation). Runs
`scripts/agent/harbor_env.py verify` (the task's `tests/` environment, no
network), then:

* writes `compression_output/<lineage>/FINAL.json` (a pointer: status
  SCORED | ZERO | NOT_RUN, reward, PPL, decode latency, correctness gate, package
  and resident GiB, and which recorded version the final submission matches)
  and copies the verifier's outputs to `FINAL_VALIDATION/<run id>/`. A
  non-result (NOT_RUN) never overwrites a measured verdict;
* packages the official tokenizer into the final submission with the task's
  own `tests/tokenizer_assets.py`.

Prints exactly one of VALID_PASS (reward > 0), VALID_FAIL (the Verifier ran
and the reward is 0) or INVALID_INFRA (no verdict was produced, or a stage failed
on the host's account -- a full disk -- which is not a measurement: NOT_RUN).
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "compression_bench" / "harness"))

from lifecycle import backend, boundary, gpus, record, verdicts  # noqa: E402
from lifecycle.utils import read_contract, read_json, utc_now, write_json_atomically  # noqa: E402

HARBOR_ENV = REPO / "compression_bench" / "scripts" / "agent" / "harbor_env.py"
GIB = 2 ** 30
STAGE_FILES = ("check", "smoke", "ppl", "accuracy", "decode", "score")


def lock_is_free(lock: Path) -> bool:
    with lock.open("rb") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(handle, fcntl.LOCK_UN)
    return True


def _float(value):
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def summarize_verifier(verifier: Path) -> dict:
    """Read the Verifier's outputs into the FINAL.json fields."""
    reward_txt = verifier / "reward.txt"
    stages = {name: read_json(verifier / f"{name}.json", None) for name in STAGE_FILES}
    score = stages["score"] or {}
    check = stages["check"] or {}
    reward = None
    if reward_txt.exists():
        try:
            reward = float(reward_txt.read_text().strip())
        except ValueError:
            reward = None
    memory = score.get("model_memory") or (stages["decode"] or {}).get("model_memory") or {}
    resident = memory.get("max_resident_bytes")
    failed_stage = next((name for name in STAGE_FILES
                         if isinstance(stages[name], dict)
                         and stages[name].get("valid") is False), None)
    error = (score.get("error") or (stages[failed_stage] or {}).get("error")
             if failed_stage else score.get("error"))
    # The host failed (a full disk), not the submission: no measurement was made.
    infra_error = reward is not None and reward <= 0 and verdicts.is_infra_error(str(error))
    if reward is None or infra_error:
        status = "NOT_RUN"
    elif reward > 0:
        status = "SCORED"
    else:
        status = "ZERO"
    return {
        "status": status, "reward": reward,
        "ppl": _float(score.get("ppl")) or _float((stages["ppl"] or {}).get("ppl")),
        "accuracy": _float(score.get("accuracy")) or _float((stages["accuracy"] or {}).get("accuracy")),
        **{key: _float(score.get(key)) for key in ("bf16_ppl", "bf16_accuracy", "chance_accuracy",
                                                    "ppl_fidelity",
                                                    "accuracy_retention", "accuracy_gate_floor")},
        "accuracy_gate_pass": score.get("accuracy_gate_pass"),
        "decode_latency_seconds": _float(score.get("decode_latency_seconds")),
        "quality_pass": score.get("quality_pass"),
        "ppl_increase_percent": _float(score.get("ppl_increase_percent")),
        "package_gib": _float(check.get("total_gib")),
        "resident_gib": (resident / GIB) if isinstance(resident, int) else None,
        "scoring_version": score.get("scoring_version"),
        "failed_stage": failed_stage,
        "error": error,
        "infra_error": infra_error,
        "stages_present": [name for name in STAGE_FILES if stages[name] is not None],
    }


def final_manifest_identity(lineage: boundary.Lineage, run_id: str,
                            submission: Path) -> tuple[str | None, str | None]:
    """(identity of the final submission, the agent version it matches)."""
    identity = None
    for row in reversed(lineage.versions()):
        if row.get("run_id") == run_id and row.get("source") == "final":
            data = read_json(lineage.dir / row["name"] / "VERSION.json", {}) or {}
            identity = data.get("manifest_identity")
            break
    if identity is None:
        manifest = boundary.hash_listing(submission, boundary.listing(submission))
        identity = boundary.identity_of(manifest)
    matched = None
    for row in lineage.versions():
        if row.get("source") != "agent":
            continue
        data = read_json(lineage.dir / row["name"] / "VERSION.json", {}) or {}
        if data.get("manifest_identity") == identity:
            matched = row["version"]
    return identity, matched


def run_verifier(*, task_dir: Path, contract: dict, root: Path, project: str, gpu: int,
                 model_dir: str, timeout: int | None, journal=None) -> str:
    """Run the task's Verifier on the submission under `root` (`harbor_env.py verify`);
    its combined output."""
    command = [os.environ.get("COMPRESSION_BENCH_HARBOR_PY") or sys.executable, str(HARBOR_ENV),
               "verify", "--task-dir", str(task_dir), "--run-root", str(root),
               "--project", project, "--gpu", str(gpu), "--model-dir", model_dir,
               "--build-timeout", str(contract["budget"]["build_timeout_s"])]
    if timeout:
        command += ["--timeout", str(timeout)]
    if journal is not None:
        journal.append("verifier_started", argv=command[2:])
    hard = int(timeout or contract["budget"]["verifier_timeout_s"]) + 2 * 3600
    try:
        got = subprocess.run(command, capture_output=True, text=True, timeout=hard, check=False)
        return got.stdout + got.stderr
    except subprocess.TimeoutExpired as exc:
        return f"harbor_env verify exceeded {hard}s: {exc}"


def package_tokenizer(task_dir: Path, submission: Path) -> dict:
    """Materialize the official tokenizer in the final submission."""
    helper = task_dir / "tests" / "tokenizer_assets.py"
    tokenizer = task_dir / "tests" / "evaluator-assets" / "tokenizer"
    if not (submission / "model").is_dir():
        return {"status": "skipped", "why": "no submission/model"}
    # -E -s, not -I: the helper imports its sibling task_policy.py, and -I would
    # drop the script's own directory from sys.path. -B: no __pycache__ in tests/.
    got = subprocess.run([sys.executable, "-B", "-E", "-s", str(helper), str(submission),
                          "--tokenizer-dir", str(tokenizer)],
                         capture_output=True, text=True, timeout=600, check=False)
    return {"status": "packaged" if got.returncode == 0 else "failed",
            "returncode": got.returncode, "output": (got.stdout + got.stderr)[-1000:]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--timeout", type=int, help="Verifier timeout override (s)")
    args = parser.parse_args(argv)

    identity = record.load_identity(args.run)
    journal = record.Journal(identity.root)
    created = journal.last("run_created") or {}
    task_dir = Path(created["task_dir"])
    contract = read_contract(task_dir)
    lineage = boundary.Lineage(Path(created["lineage_dir"]).parent, identity.lineage_id)
    submission = identity.root / "submission"
    verifier = identity.root / "verifier"

    running = [name for name in backend.project_containers(identity.project)
               if backend.running(name)]
    if running:
        print(f"refusing: the agent environment is still running ({', '.join(running)})",
              file=sys.stderr)
        print("INVALID_INFRA")
        return 3
    if not lock_is_free(identity.root / ".submission.lock"):
        print("refusing: the submission lock is held", file=sys.stderr)
        print("INVALID_INFRA")
        return 3
    # A previous attempt's outputs are moved aside, never mixed with this one.
    if verifier.exists() and any(verifier.iterdir()):
        aside = identity.root / f"verifier.previous.{len(list(identity.root.glob('verifier.previous.*')))}"
        verifier.rename(aside)
    verifier.mkdir(parents=True, exist_ok=True)

    # The verdict needs an idle card: a foreign process on the run's own card would fail the engine's
    # start-up memory check and score the submission zero for the host's fault.
    got_card = gpus.acquire_idle(int(created["gpu"]), run_id=identity.run_id,
                                 give_up_s=float(os.environ.get("COMPRESSION_BENCH_VERDICT_GPU_WAIT_H", "2")) * 3600,
                                 log=lambda m: print(f"[compression_bench] {m}", file=sys.stderr, flush=True))
    if got_card is None:
        journal.append("verdict_gpu_unavailable", gpu=created["gpu"])
        print("refusing: no idle GPU for the verdict", file=sys.stderr)
        print("INVALID_INFRA")
        return 3
    card, card_claim = got_card
    if card != int(created["gpu"]):
        journal.append("verdict_gpu_moved", from_gpu=created["gpu"], to_gpu=card)
    try:
        harbor_output = run_verifier(task_dir=task_dir, contract=contract, root=identity.root,
                                     project=identity.project, gpu=card, model_dir=created["model_dir"],
                                     timeout=args.timeout, journal=journal)
    finally:
        if card_claim is not None:
            card_claim.release()
    (identity.root / "harbor_env.verify.log").write_text(harbor_output, encoding="utf-8")

    summary = summarize_verifier(verifier)
    final_identity, matched = final_manifest_identity(lineage, identity.run_id, submission)
    evidence = lineage.dir / "FINAL_VALIDATION" / identity.run_id
    if evidence.exists():
        shutil.rmtree(evidence)
    shutil.copytree(verifier, evidence)
    final = {
        "kind": "final_verdict_pointer", "schema_version": 1,
        "task_id": identity.task_id, "run_id": identity.run_id,
        "lineage_id": identity.lineage_id, "verified_utc": utc_now(),
        **summary,
        "final_manifest_identity": final_identity, "matches_version": matched,
        "evidence": str(evidence.relative_to(REPO)) if evidence.is_relative_to(REPO)
        else str(evidence),
        "run_root": str(identity.root),
    }
    pointer = lineage.dir / "FINAL.json"
    previous = read_json(pointer, None)
    with (lineage.dir / "FINAL.attempts.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(final) + "\n")
    if (summary["status"] != "NOT_RUN" or not previous
            or previous.get("status") == "NOT_RUN"):
        write_json_atomically(pointer, final)
    if summary["status"] != "NOT_RUN":
        packaging = package_tokenizer(task_dir, submission)
        journal.append("submission_packaged", **packaging)
    status = {"SCORED": "VALID_PASS", "ZERO": "VALID_FAIL"}.get(summary["status"],
                                                                 "INVALID_INFRA")
    journal.append("verifier_finished", status=summary["status"], reward=summary["reward"],
                   failed_stage=summary["failed_stage"], matches_version=matched)
    print(json.dumps({k: final[k] for k in ("status", "reward", "ppl",
                                            "decode_latency_seconds", "package_gib",
                                            "resident_gib", "matches_version")}))
    print(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
