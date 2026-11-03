#!/usr/bin/env python3
"""The full hidden verdict on every checkpoint of a run; then its weights go.

    verdict_checkpoints.py --run <run root> [--gpu N] [--versions v3,v5] [--keep-weights]

Checkpoints are not an archive. A checkpoint is a version's reflink snapshot
(`compression_output/.objects/<lineage>/<version>/`); each distinct one (by
content, interpreter caches ignored) is evaluated by the task's own Verifier
exactly as a final submission is -- package check, smoke, hidden PPL, the
correctness gate, decode -- in its own directory (`<run>/checkpoint-verdicts/
<version>/`: the snapshot reflink-copied as `submission/`, a lock, `verifier/`).
Only once its verdict is a measurement (SCORED or ZERO, not a failure of the
host such as a full disk) are the snapshot's weights deleted; an unmeasured
checkpoint stays for a later attempt.

* A checkpoint identical to the final submission takes the final verdict when
  that one is a measurement.
* When the run has no measured final verdict, the verdict of the checkpoint
  identical to its submission (the final snapshot, or the agent's last version)
  becomes the run's final verdict: `FINAL.json`, `FINAL_VALIDATION/<run id>/`.

Evidence: every Verifier output under `<lineage>/CHECKPOINT_VERDICTS/<version>/`,
one row per checkpoint in `<lineage>/checkpoint_verdicts.jsonl`, `verdict` in each
VERSION.json, and the run's journal. Prints one JSON line per checkpoint.
Called by `lifecycle/run.py` after the final verdict, on the run's GPU; standalone
for an ended run (inspect the card first).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "compression_bench" / "harness"))

from lifecycle import boundary, disk, gpus, record, verdicts  # noqa: E402
from lifecycle.utils import read_contract, read_json, utc_now, write_json_atomically  # noqa: E402

_spec = importlib.util.spec_from_file_location("compression_bench_validate_evolve",
                                               Path(__file__).resolve().parent / "validate_evolve.py")
validate_evolve = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validate_evolve)


def version_identity(lineage: boundary.Lineage, row: dict) -> str | None:
    manifest = read_json(lineage.dir / row["name"] / "MANIFEST.json", []) or []
    return verdicts.content_identity(manifest)


def submission_identity(submission: Path) -> str | None:
    if not submission.is_dir():
        return None
    # The official tokenizer a verdict packages in is ignored by content_identity.
    return verdicts.content_identity(boundary.hash_listing(submission, boundary.listing(submission)))


def has_weights(folder: Path) -> bool:
    return folder.is_dir() and any(p.is_file() and p.suffix.lower() in boundary.WEIGHT_SUFFIXES
                                   for p in folder.rglob("*"))


def delete_snapshot(lineage: boundary.Lineage, row: dict, why: str) -> int:
    folder = lineage.objects / row["version"]
    freed = 0
    if folder.is_dir():
        freed = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file() and not p.is_symlink())
        shutil.rmtree(folder, ignore_errors=True)
    path = lineage.dir / row["name"] / "VERSION.json"
    data = read_json(path, {}) or {}
    if data:
        data["snapshot"] = {**(data.get("snapshot") or {}), "status": "deleted",
                            "deleted_utc": utc_now(), "why": why}
        write_json_atomically(path, data)
    return freed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--gpu", type=int, help="default: the run's card")
    parser.add_argument("--versions", help="comma-separated versions (default: every checkpoint)")
    parser.add_argument("--timeout", type=int, help="Verifier timeout override (s)")
    parser.add_argument("--keep-weights", action="store_true", help="verdict, but delete nothing")
    args = parser.parse_args(argv)

    identity = record.load_identity(args.run)
    journal = record.Journal(identity.root)
    created = journal.last("run_created") or {}
    task_dir = Path(created["task_dir"])
    contract = read_contract(task_dir)
    lineage = boundary.Lineage(Path(created["lineage_dir"]).parent, identity.lineage_id)
    gpu = args.gpu if args.gpu is not None else int(created["gpu"])
    wanted = set(args.versions.split(",")) if args.versions else None

    rows = [r for r in lineage.versions() if r.get("run_id") == identity.run_id
            and (wanted is None or r["version"] in wanted)]
    final_row = next((r for r in reversed(rows) if r.get("source") == "final"), None)
    final = read_json(lineage.dir / "FINAL.json", {}) or {}
    final_is_this_run = final.get("run_id") == identity.run_id
    final_measured = final_is_this_run and verdicts.measured(final)
    final_content = (version_identity(lineage, final_row) if final_row
                     else submission_identity(identity.root / "submission"))

    by_content: dict[str, dict] = {}
    if final_measured and final_content:
        by_content[final_content] = {**{k: final.get(k) for k in (
            "status", "reward", "ppl", "accuracy", "accuracy_retention", "accuracy_gate_pass",
            "package_gib", "resident_gib", "decode_latency_seconds", "failed_stage", "error")},
            "evidence": final.get("evidence"), "verdict_of": "final"}
    ledger = lineage.dir / "checkpoint_verdicts.jsonl"
    results = []
    for row in rows:
        snapshot = lineage.objects / row["version"]
        if not has_weights(snapshot):
            continue
        content = version_identity(lineage, row)
        verdict = by_content.get(content) if content else None
        if verdict is None:
            while True:
                space = disk.check(identity.root, docker_floor=disk.floor(
                    "COMPRESSION_BENCH_MIN_FREE_VERDICT_DOCKER_GIB", disk.MIN_FREE_VERDICT_DOCKER_GIB), run_floor=0)
                if space["ok"] or os.environ.get("COMPRESSION_BENCH_NO_DISK_WAIT"):
                    break
                # A full disk pauses the verdict (it is not NOT_RUN): the run waits for space.
                print(f"[compression_bench] PAUSED {time.strftime('%FT%TZ', time.gmtime())}: verdict of "
                      f"{row['version']} waits for disk space: {'; '.join(space['problems'])}",
                      file=sys.stderr, flush=True)
                time.sleep(120)
            if not space["ok"]:
                verdict = {"status": "NOT_RUN", "infra_error": True,
                           "error": "; ".join(space["problems"]), "verdict_of": row["version"]}
            else:
                root = identity.root / "checkpoint-verdicts" / row["version"]
                if root.exists():
                    shutil.rmtree(root)
                for sub in ("verifier", "compose"):
                    (root / sub).mkdir(parents=True, exist_ok=True)
                (root / ".submission.lock").touch()
                subprocess.run(["cp", "-a", "--reflink=auto", f"{snapshot}/.", str(root / "submission")],
                               check=True)
                journal.append("checkpoint_verdict_started", version=row["version"], gpu=gpu,
                               content_identity=content, disk=space)
                # An idle card, never one a foreign process holds memory on (the engine's start-up
                # memory check would fail and the checkpoint would score zero for the host's fault).
                got_card = gpus.acquire_idle(
                    gpu, run_id=identity.run_id,
                    give_up_s=float(os.environ.get("COMPRESSION_BENCH_VERDICT_GPU_WAIT_H", "2")) * 3600,
                    log=lambda m: print(f"[compression_bench] {m}", file=sys.stderr, flush=True))
                if got_card is None:
                    shutil.rmtree(root / "submission", ignore_errors=True)
                    verdict = {"status": "NOT_RUN", "infra_error": True, "verdict_of": row["version"],
                               "error": f"no idle GPU for the verdict (GPU {gpu} stayed busy)"}
                    journal.append("verdict_gpu_unavailable", version=row["version"], gpu=gpu)
                else:
                    card, card_claim = got_card
                    if card != gpu:
                        journal.append("verdict_gpu_moved", version=row["version"], from_gpu=gpu, to_gpu=card)
                    try:
                        output = validate_evolve.run_verifier(
                            task_dir=task_dir, contract=contract, root=root,
                            project=f"{identity.project}-ck-{row['version']}", gpu=card,
                            model_dir=created["model_dir"], timeout=args.timeout)
                    finally:
                        if card_claim is not None:
                            card_claim.release()
                    (root / "harbor_env.verify.log").write_text(output, encoding="utf-8")
                    summary = validate_evolve.summarize_verifier(root / "verifier")
                    evidence = lineage.dir / "CHECKPOINT_VERDICTS" / row["version"]
                    if evidence.exists():
                        shutil.rmtree(evidence)
                    shutil.copytree(root / "verifier", evidence)
                    shutil.rmtree(root / "submission", ignore_errors=True)
                    verdict = {**summary, "verdict_of": row["version"],
                               "evidence": str(evidence.relative_to(REPO)) if evidence.is_relative_to(REPO)
                               else str(evidence)}
            if verdicts.measured(verdict) and content:
                by_content[content] = verdict
        entry = {"kind": "compression_bench_checkpoint_verdict", "utc": utc_now(), "run_id": identity.run_id,
                 "version": row["version"], "source": row.get("source"), "content_identity": content,
                 "same_as_final": bool(content and content == final_content), **verdict}
        path = lineage.dir / row["name"] / "VERSION.json"
        data = read_json(path, {}) or {}
        if data:
            data["verdict"] = {k: v for k, v in entry.items() if k != "kind"}
            write_json_atomically(path, data)
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        # The run's final verdict, when it had none that measured anything.
        if (not final_measured and verdicts.measured(verdict) and content == final_content
                and verdict.get("verdict_of") == row["version"]):
            pointer = {"kind": "final_verdict_pointer", "schema_version": 1,
                       "task_id": identity.task_id, "run_id": identity.run_id,
                       "lineage_id": identity.lineage_id, "verified_utc": utc_now(),
                       **{k: v for k, v in verdict.items() if k not in ("verdict_of",)},
                       "final_manifest_identity": content, "matches_version": row["version"],
                       "verified_from": f"checkpoint {row['version']} (identical to the submission)",
                       "run_root": str(identity.root)}
            target = lineage.dir / "FINAL_VALIDATION" / identity.run_id
            if target.exists():
                shutil.rmtree(target)
            # The same files: hard links, not a second copy on the host's disk.
            shutil.copytree(REPO / verdict["evidence"] if not Path(verdict["evidence"]).is_absolute()
                            else Path(verdict["evidence"]), target, copy_function=os.link)
            pointer["evidence"] = str(target.relative_to(REPO)) if target.is_relative_to(REPO) else str(target)
            with (lineage.dir / "FINAL.attempts.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(pointer) + "\n")
            write_json_atomically(lineage.dir / "FINAL.json", pointer)
            final_measured = True
            journal.append("final_verdict_from_checkpoint", version=row["version"],
                           status=verdict.get("status"), reward=verdict.get("reward"))
        freed = 0
        if verdicts.measured(verdict) and not args.keep_weights:
            freed = delete_snapshot(lineage, row, f"verdict {verdict.get('status')} "
                                    f"({verdict.get('verdict_of')}) recorded")
        journal.append("checkpoint_verdict", version=row["version"], status=verdict.get("status"),
                       reward=verdict.get("reward"), verdict_of=verdict.get("verdict_of"),
                       infra_error=verdict.get("infra_error"), freed_bytes=freed)
        line = {k: entry.get(k) for k in ("version", "status", "reward", "ppl", "accuracy",
                                          "accuracy_gate_pass", "package_gib", "resident_gib",
                                          "verdict_of", "same_as_final", "error")}
        print(json.dumps({**line, "freed_bytes": freed}), flush=True)
        results.append(entry)
    print(json.dumps({"checkpoints": len(results),
                      "measured": sum(verdicts.measured(r) for r in results),
                      "kept_unmeasured": [r["version"] for r in results if not verdicts.measured(r)]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
