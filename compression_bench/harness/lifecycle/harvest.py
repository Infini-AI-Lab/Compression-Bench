"""After a run: keep every metric, then free the weights.

A run leaves full-size weights in three places: the agent's workspace (its
candidates and intermediates, often several model copies), the submission, and
the lineage's reflink checkpoints (`compression_output/.objects/`). Weights
are not an archive here; metrics are. So the order is fixed:

1. `collect` reads everything measurable about the run into one document:
   identity and task, the agent's usage, every version (manifest identity,
   composition, the public Checker result for exactly its files), every public
   check the agent ran (the Checker's own history), the hidden verdict stage by
   stage, the run's timeline and its disk footprint.
2. `harvest` writes it as `<run>/METRICS.json` and
   `<lineage>/METRICS/<run id>.json` (the lineage is the durable record) and
   reads both back.
3. Only then `clean` deletes weights: every weight file the agent left in its
   workspace (debug-submission included), the compile and temp caches, and every
   checkpoint a verdict has measured (`final_validate/tools/verdict_checkpoints.py`
   runs the full hidden verdict on each one first). Checkpoints are not kept for
   rollback (`COMPRESSION_BENCH_KEEP_CHECKPOINTS`, default 0); one no verdict measured yet
   (the host failed, or the step did not run) stays for a later attempt, and so
   do the submission's weights while the final verdict has not measured them.
   Code, configs, logs, manifests (every weight's sha256) and every verdict's
   evidence stay. What was deleted is listed in METRICS.json.

`collect` also carries every verdict in detail (`verdicts.detail`): the final one
and each checkpoint's -- per-block hidden perplexities, the gate's per-benchmark
counts and predictions, each decode request, the resident census by module, the
stage timings -- under `verdicts`.

Nothing here measures a submission; it reads what the run already recorded.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from . import boundary, record, trajectory, verdicts
from .report import usage_totals
from .utils import REPO, read_contract, read_json, read_jsonl, utc_now, write_json_atomically

SCHEMA = "compression_bench.run_metrics/1"

WEIGHT_SUFFIXES = boundary.WEIGHT_SUFFIXES
#: A file with a weight suffix and at least this many bytes is a weight.
WEIGHT_MIN_BYTES = 1 << 20
#: Any file at least this large is cleaned whatever its name (dumps, tarballs),
#: except text: logs and transcripts are the trajectory, never cleaned.
LARGE_FILE_BYTES = 256 << 20
TEXT_SUFFIXES = {".log", ".txt", ".md", ".jsonl", ".csv", ".out", ".err"}
#: Regenerable caches under the run root, removed whole.
CACHE_DIRS = ("work/.tmp", "work/.agent-home/.cache")


class NotFinished(RuntimeError):
    """The run is still live (or never wrote its summary); nothing is touched."""


# --- reading ------------------------------------------------------------------------

def _first(events: list[dict], name: str) -> dict:
    return next((e for e in events if e.get("event") == name), {})


def _last(events: list[dict], name: str) -> dict:
    found = [e for e in events if e.get("event") == name]
    return found[-1] if found else {}


def scalars(value, prefix: str = "", depth: int = 0, out: dict | None = None) -> dict:
    """Flatten a JSON document to its scalar leaves (dotted keys); short numeric
    lists are kept, long arrays and blobs are not (they stay in the evidence)."""
    out = {} if out is None else out
    if isinstance(value, dict) and depth < 4:
        for key, item in value.items():
            scalars(item, f"{prefix}.{key}" if prefix else str(key), depth + 1, out)
    elif isinstance(value, list):
        if len(value) <= 16 and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                    for x in value):
            out[prefix] = value
    elif isinstance(value, str):
        if len(value) <= 400:
            out[prefix] = value
    elif value is None or isinstance(value, (int, float, bool)):
        out[prefix] = value
    return out


def lineage_dir(events: list[dict]) -> Path | None:
    """The run's lineage directory, as `run_created` recorded it (a relative path is the repository's)."""
    recorded = _first(events, "run_created").get("lineage_dir")
    if not recorded:
        return None
    path = Path(recorded) if Path(recorded).is_absolute() else REPO / recorded
    return path if path.is_dir() else None


def verdict_for(run_id: str, lineage: Path | None) -> dict:
    """This run's verdict pointer: FINAL.json if it is this run's, else its attempt."""
    if lineage is None:
        return {}
    final = read_json(lineage / "FINAL.json", {}) or {}
    if final.get("run_id") == run_id:
        return final
    found = {}
    for row in read_jsonl(lineage / "FINAL.attempts.jsonl"):
        if row.get("run_id") == run_id:
            found = row
    return found


def hidden_stages(root: Path, lineage: Path | None, run_id: str) -> dict:
    """Each hidden stage's scalars (per-token arrays stay in the evidence), from the
    lineage's evidence (or the run's verifier/)."""
    folders = ([lineage / "FINAL_VALIDATION" / run_id] if lineage else []) + [root / "verifier"]
    stages = {}
    for stage in verdicts.STAGES:
        for folder in folders:
            data = read_json(folder / f"{stage}.json", None)
            if isinstance(data, dict):
                stages[stage] = scalars(data)
                break
    return stages


def verdict_details(root: Path, lineage: Path | None, run_id: str) -> dict:
    """The final verdict and every checkpoint verdict of this run, in detail."""
    final = verdict_for(run_id, lineage)
    folder = lineage / "FINAL_VALIDATION" / run_id if lineage else None
    if folder is None or not folder.is_dir():
        folder = root / "verifier"
    out = {"final": {"status": final.get("status") or "NOT_RUN",
                     "measured": verdicts.measured(final),
                     "verified_from": final.get("verified_from"),
                     "detail": verdicts.detail(folder) if folder.is_dir() else None},
           "checkpoints": []}
    if lineage is None:
        return out
    latest: dict[str, dict] = {}
    for row in read_jsonl(lineage / "checkpoint_verdicts.jsonl"):
        if row.get("run_id") == run_id:
            latest[row["version"]] = row
    for version, row in sorted(latest.items(), key=lambda kv: int(kv[0].lstrip("v") or 0)):
        evidence = row.get("evidence")
        own = row.get("verdict_of") == version and evidence
        path = (Path(evidence) if Path(str(evidence)).is_absolute() else REPO / str(evidence)) if own else None
        out["checkpoints"].append({
            **{k: row.get(k) for k in ("version", "source", "content_identity", "same_as_final",
                                       "status", "reward", "ppl", "accuracy", "accuracy_retention",
                                       "accuracy_gate_pass", "package_gib", "resident_gib",
                                       "decode_latency_seconds", "failed_stage", "error",
                                       "infra_error", "verdict_of", "evidence", "utc")},
            "measured": verdicts.measured(row),
            # A checkpoint identical to an earlier one points at that one's detail.
            "detail": verdicts.detail(path) if path is not None and path.is_dir() else None})
    return out


def lineage_versions(lineage: Path | None) -> list[dict]:
    """Every version of the lineage, in order, from its VERSION.json."""
    if lineage is None:
        return []
    out = []
    for row in read_jsonl(lineage / "versions.jsonl"):
        detail = read_json(lineage / row.get("name", "") / "VERSION.json", {}) or {}
        out.append({**row, **detail})
    return out


def version_metrics(version: dict, lineage: Path, history: Path) -> dict:
    """What we keep about one checkpoint once its weights are gone."""
    manifest = read_json(lineage / version["name"] / "MANIFEST.json", []) or []
    composition = version.get("composition") or boundary.composition_of(manifest)
    check = version.get("public_check")
    if check is None and manifest:
        check = boundary.public_check_for(history, manifest,
                                          before=version.get("captured_utc") or "~")
    snapshot = version.get("snapshot") or {}
    return {"version": version.get("version"), "source": version.get("source"),
            "captured_utc": version.get("captured_utc"), "note": version.get("note"),
            "files": version.get("files"), "total_bytes": version.get("total_bytes"),
            "weight_bytes": composition.get("weight_bytes"),
            "composition": composition.get("by_suffix"),
            "manifest_identity": version.get("manifest_identity"),
            "public_check": check,
            "snapshot": snapshot.get("status") if isinstance(snapshot, dict) else snapshot}


PUBLIC_KEYS = boundary.PUBLIC_CHECK_FIELDS + ("limit_gib", "full_checks_used", "debug_only")


def public_checks(root: Path) -> dict:
    """Every public check the agent ran, from the Checker's own history."""
    out = []
    for row in read_jsonl(root / "public-checks" / "checks.jsonl"):
        response = row.get("response") or {}
        request = row.get("request") if isinstance(row.get("request"), dict) else {}
        listing = row.get("submission") or []
        out.append({"utc": row.get("utc_start"), "seconds": row.get("seconds"),
                    "action": request.get("action"), "mode": request.get("mode", "formal"),
                    "submission_files": len(listing),
                    "submission_bytes": sum(int(n) for _, n in listing),
                    "resident_bytes": (response.get("model_memory") or {}).get("max_resident_bytes"),
                    **{k: response[k] for k in PUBLIC_KEYS if k in response}})
    return {"source": "checker-history", "complete": True, "checks": out}


def footprint(root: Path, objects: Path | None, top: int = 15) -> dict:
    """Apparent bytes per top-level entry of the run root (work/ one level
    deeper), the lineage's checkpoints, and the largest files."""
    by_entry: dict[str, int] = {}
    largest: list[tuple[int, str]] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            path = Path(dirpath) / name
            try:
                if path.is_symlink():
                    continue
                size = path.stat().st_size
            except OSError:
                continue
            parts = path.relative_to(root).parts
            entry = "/".join(parts[:2]) if parts[0] == "work" and len(parts) > 1 else parts[0]
            by_entry[entry] = by_entry.get(entry, 0) + size
            largest.append((size, "/".join(parts)))
    largest.sort(reverse=True)
    return {"run_bytes": sum(by_entry.values()),
            "by_entry": dict(sorted(by_entry.items(), key=lambda kv: -kv[1])),
            "checkpoint_bytes": _tree_bytes(objects) if objects else 0,
            "largest_files": [{"path": p, "bytes": s} for s, p in largest[:top]]}


def _tree_bytes(folder: Path) -> int:
    total = 0
    if folder.is_dir():
        for dirpath, _, filenames in os.walk(folder):
            for name in filenames:
                try:
                    total += (Path(dirpath) / name).stat().st_size
                except OSError:
                    continue
    return total


def collect(root: Path | str) -> dict:
    """Everything measurable about a finished run, from what it recorded."""
    root = Path(root)
    events = list(record.Journal(root))
    created = _first(events, "run_created")
    identity = read_json(root / "identity.json", {}) or {}
    summary = read_json(root / "SUMMARY.json", {}) or {}
    run_id = identity.get("run_id") or created.get("run_id")
    lineage = lineage_dir(events)
    task_dir = Path(created["task_dir"]) if created.get("task_dir") else None
    contract = read_contract(task_dir) if task_dir and (task_dir / "contract.yaml").exists() else {}
    evaluation = contract.get("evaluation") or {}

    agent = dict(summary.get("agent") or {})
    launches = [e for e in events if e.get("event") == "agent_launched"]
    if (root / "agent.stream.jsonl").exists() or "usage" not in agent:
        agent["usage"] = usage_totals(root / "agent.stream.jsonl")     # priced by pricing.py
    agent.setdefault("launches", len(launches))
    for key in ("agent_cli", "agent_model", "agent_effort"):
        agent.setdefault(key, created.get(key))

    history = root / "public-checks" / "checks.jsonl"
    versions = [version_metrics(v, lineage, history) for v in lineage_versions(lineage)
                if v.get("run_id") == run_id] if lineage else []
    checks = public_checks(root)
    by_identity = {}
    for version in versions:          # a version identical to an earlier checked one
        contents = version.get("manifest_identity")
        if version["public_check"] is None and contents in by_identity:
            version["public_check"] = {**by_identity[contents], "matched_by": "same-manifest"}
        elif version["public_check"] is not None and contents:
            by_identity[contents] = version["public_check"]
    final = verdict_for(run_id, lineage)
    validation = _last(events, "final_validation")
    ended = _last(events, "agent_phase_ended")
    objects = lineage.parent / boundary.OBJECTS / lineage.name if lineage else None
    return {
        "kind": "compression_bench_run_metrics", "schema": SCHEMA, "collected_utc": utc_now(),
        "run": {"run_id": run_id, "namespace": identity.get("namespace") or created.get("namespace"),
                "root": str(root), "lineage_id": created.get("lineage_id"),
                "lineage_dir": str(lineage) if lineage else None,
                "state": record.current_state(record.Journal(root)), "gpu": created.get("gpu"),
                "budget_hours": identity.get("budget_hours") or created.get("budget_hours"),
                "model_dir": created.get("model_dir"),
                "debug_model_dir": created.get("debug_model_dir")},
        "task": {"task_id": contract.get("task_id") or created.get("task_id"),
                 "task_hash": created.get("task_hash"), "version": contract.get("version"),
                 "model": (contract.get("model") or {}).get("id"),
                 "model_revision": (contract.get("model") or {}).get("revision"),
                 "capacity_gib": contract.get("capacity_gib"),
                 "platform": (contract.get("platform") or {}).get("image"),
                 "quality_rule": evaluation.get("quality_rule"),
                 "quality": evaluation.get("quality"),
                 "ppl_protocol": evaluation.get("ppl_protocol"),
                 "scoring_protocol": evaluation.get("scoring_protocol"),
                 "reference": (contract.get("reference") or {}).get("label")},
        "agent": agent,
        "timeline": {
            "created_utc": created.get("utc"),
            "agent_started_utc": launches[0].get("utc") if launches else None,
            "agent_ended_utc": ended.get("utc"), "agent_end_reason": ended.get("reason"),
            "deadline_utc": identity.get("deadline_utc") or created.get("deadline_utc"),
            "final_captured_utc": _last(events, "final_captured").get("utc"),
            "verifier_started_utc": _last(events, "final_validation_started").get("utc"),
            "verifier_finished_utc": validation.get("utc"),
            "summary_utc": summary.get("written_utc")},
        "verdict": {"status": final.get("status") or "NOT_RUN",
                    **{k: final.get(k) for k in (
                        "reward", "ppl", "accuracy", "ppl_fidelity",
                        "bf16_ppl", "bf16_accuracy", "chance_accuracy", "accuracy_retention",
                        "accuracy_gate_floor", "accuracy_gate_pass", "decode_latency_seconds",
                        "ppl_increase_percent", "package_gib", "resident_gib",
                        "scoring_version", "failed_stage", "error", "matches_version",
                        "final_manifest_identity", "evidence")}},
        "hidden_stages": hidden_stages(root, lineage, run_id),
        "verdicts": verdict_details(root, lineage, run_id),
        "versions": versions,
        "public_checks": checks,
        "disk_before_cleanup": footprint(root, objects),
    }


# --- cleaning -----------------------------------------------------------------------

def is_weight(path: Path, size: int) -> bool:
    suffix = path.suffix.lower()
    return (suffix in WEIGHT_SUFFIXES and size >= WEIGHT_MIN_BYTES) \
        or (size >= LARGE_FILE_BYTES and suffix not in TEXT_SUFFIXES)


def finished(root: Path) -> None:
    """Refuse unless the run reached a terminal state and wrote its summary."""
    state = record.current_state(record.Journal(root))
    if state not in record.TERMINAL:
        raise NotFinished(f"{root.name} is {state}, not finished")
    if not (root / "SUMMARY.json").exists():
        raise NotFinished(f"{root.name} has no SUMMARY.json yet (the driver is still closing)")


def checkpoints_to_keep(versions: list[dict], objects: Path, keep: int) -> list[str]:
    """The newest `keep` versions with a checkpoint on disk and distinct contents."""
    kept, identities = [], set()
    if keep <= 0:
        return kept
    for version in reversed(versions):
        if not (objects / version["version"]).is_dir():
            continue
        identity = version.get("manifest_identity")
        if identity and identity in identities:
            continue
        kept.append(version["version"])
        identities.add(identity)
        if len(kept) == keep:
            break
    return kept


def clean(root: Path | str, metrics: dict, *, keep: int | None = None,
          dry_run: bool = False) -> dict:
    """Delete the run's weights once verdicts measured them (module docstring)."""
    root = Path(root)
    finished(root)
    keep = boundary.keep_checkpoints() if keep is None else max(0, keep)
    deleted: list[dict] = []
    freed = 0

    def remove_file(path: Path) -> None:
        nonlocal freed
        try:
            size = path.stat().st_size
        except OSError:
            return
        deleted.append({"path": str(path.relative_to(root)) if str(path).startswith(str(root))
                        else str(path), "bytes": size})
        freed += size
        if not dry_run:
            path.unlink(missing_ok=True)

    # 1. caches, removed whole
    for relative in CACHE_DIRS:
        folder = root / relative
        if folder.is_dir():
            size = _tree_bytes(folder)
            freed += size
            deleted.append({"path": relative, "bytes": size, "kind": "cache directory"})
            if not dry_run:
                shutil.rmtree(folder, ignore_errors=True)

    # 2. checkpoints: each one goes once a verdict measured it (or measured the final
    #    submission it is the snapshot of); an unmeasured one stays; `keep` more stay.
    lineage = Path(metrics["run"]["lineage_dir"]) if metrics["run"].get("lineage_dir") else None
    kept: list[str] = []
    unmeasured: list[str] = []
    dropped: list[dict] = []
    run_id = metrics["run"]["run_id"]
    final_measured = verdicts.measured(verdict_for(run_id, lineage)) if lineage else False
    if lineage is not None:
        objects = lineage.parent / boundary.OBJECTS / lineage.name
        versions = lineage_versions(lineage)
        final_runs: dict[str, bool] = {}

        def is_measured(version: dict) -> bool:
            if verdicts.measured(version.get("verdict") or {}):
                return True
            if version.get("source") == "final":
                owner = version.get("run_id")
                if owner not in final_runs:
                    final_runs[owner] = verdicts.measured(verdict_for(owner, lineage))
                return final_runs[owner]
            return False
        measured_versions = [v for v in versions if is_measured(v)]
        kept = checkpoints_to_keep(measured_versions, objects, keep)
        for version in versions:
            folder = objects / version["version"]
            if not folder.is_dir():
                continue
            if not is_measured(version):
                unmeasured.append(version["version"])
                continue
            if version["version"] in kept:
                continue
            size = _tree_bytes(folder)
            freed += size
            dropped.append({"version": version["version"], "bytes": size})
            if not dry_run:
                shutil.rmtree(folder, ignore_errors=True)
                path = lineage / version["name"] / "VERSION.json"
                data = read_json(path, {}) or {}
                if data:
                    data["snapshot"] = {**(data.get("snapshot") or {}), "status": "cleaned",
                                        "cleaned_utc": utc_now()}
                    write_json_atomically(path, data)

    # 3. weights in the workspace; the submission's once the final verdict measured them
    scopes = [root / "work"] + ([root / "submission"] if final_measured else [])
    caches = [root / c for c in CACHE_DIRS]
    for scope in scopes:
        for dirpath, _, filenames in os.walk(scope):
            if any(Path(dirpath).is_relative_to(c) for c in caches):
                continue
            for name in filenames:
                path = Path(dirpath) / name
                try:
                    if path.is_symlink():
                        continue
                    size = path.stat().st_size
                except OSError:
                    continue
                if is_weight(path, size):
                    remove_file(path)

    return {"cleaned_utc": utc_now(), "dry_run": dry_run, "keep_checkpoints": keep,
            "checkpoints_kept": kept, "checkpoints_unmeasured_kept": unmeasured,
            "checkpoints_deleted": dropped,
            "submission_weights_removed": final_measured,
            "freed_bytes": freed,
            "files_deleted": sum(1 for d in deleted if "kind" not in d),
            "deleted": deleted}


# --- the whole step -----------------------------------------------------------------

def harvest(root: Path | str, *, clean_weights: bool = True, keep: int | None = None,
            dry_run: bool = False) -> dict:
    """Collect, write and read back METRICS.json; then clean. Returns the metrics."""
    root = Path(root)
    finished(root)
    metrics = collect(root)
    previous = read_json(root / "METRICS.json", {}) or {}
    if (previous.get("task") or {}).get("task_id"):
        # The task as the run was measured; a later contract (a new protocol, a re-render) must not rewrite it.
        metrics["task"] = previous["task"]
    if previous.get("cleanup"):
        # A second pass (a sweep): the first footprint and every cleanup stay on record.
        metrics["disk_before_cleanup"] = previous.get("disk_before_cleanup")
        metrics["cleanup_history"] = previous.get("cleanup_history") or [previous["cleanup"]]
    targets = [root / "METRICS.json"]
    if metrics["run"].get("lineage_dir"):
        targets.append(Path(metrics["run"]["lineage_dir"]) / "METRICS" /
                       f"{metrics['run']['run_id']}.json")
    if dry_run:
        metrics["cleanup"] = clean(root, metrics, keep=keep, dry_run=True) if clean_weights else None
        return metrics
    try:
        metrics["trajectory"] = trajectory.brief(
            trajectory.export(root, metrics["run"].get("lineage_dir")))
    except Exception as exc:   # the metrics are still written; cleaning spares the sources
        metrics["trajectory"] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
    # Timing, resources and the agent's memory, at the top level (telemetry.py).
    for key, source in (("timing", "timing"), ("resources", "resources"), ("agent_memory", "memory")):
        metrics[key] = (metrics.get("trajectory") or {}).get(source)
    for target in targets:
        write_json_atomically(target, metrics)
        back = read_json(target, {}) or {}
        if back.get("run", {}).get("run_id") != metrics["run"]["run_id"] \
                or back.get("verdict") != metrics["verdict"] \
                or len(back.get("versions") or []) != len(metrics["versions"]):
            raise RuntimeError(f"{target} did not read back; nothing cleaned")
    journal = record.Journal(root)
    journal.append("metrics_harvested", files=[str(t) for t in targets],
                   verdict=metrics["verdict"].get("status"),
                   trajectory_steps=(metrics.get("trajectory") or {}).get("steps"),
                   public_checks=len(metrics["public_checks"]["checks"]),
                   versions=len(metrics["versions"]))
    if not clean_weights:
        metrics["cleanup"] = None
        return metrics
    cleanup = clean(root, metrics, keep=keep)
    metrics["cleanup"] = cleanup
    metrics["cleanup_history"] = (metrics.get("cleanup_history") or []) + [
        {k: v for k, v in cleanup.items() if k != "deleted"}]
    for target in targets:
        write_json_atomically(target, metrics)
    journal.append("weights_cleaned", freed_bytes=cleanup["freed_bytes"],
                   files_deleted=cleanup["files_deleted"],
                   checkpoints_kept=cleanup["checkpoints_kept"],
                   checkpoints_deleted=[c["version"] for c in cleanup["checkpoints_deleted"]])
    return metrics
