"""The evolve driver: one run of one task, from an empty workspace to a verdict.

    compression_bench/scripts/evolve --task compression_tasks/<task> --gpu N --hours H \\
        --model-dir <pinned BF16 checkpoint> [--debug-model-dir D] \\
        [--agent-cli claude --agent-model M --agent-effort E] [--dry-run]

Stages, in order (ServerBench's, with the compression task's specifics):

  create          allocate the run id and its directory, open the journal,
                  open the lineage under compression_output/
  claim GPU       refuse a card with a compute process, memory in use, or
                  another run's claim; hold a flock on it (a container
                  reservation is only reported)
  model identity  the checkpoint is the one the task pins
  workspace       .submission.lock, work/, submission/, verifier/; the brief,
                  WORKSPACE.json and the version/clock tools; staged credentials
  container       Harbor composes the task's environment/ (main + public
                  Checker + egress allowlist) with the per-run overlay
  boot proof      in-container checks; any failure is HARNESS_SETUP_FAILED
  develop         the agent CLI, continued until the deadline; version
                  requests are captured while it runs
  finish          the final capture; the environment is stopped, which
                  releases the submission lock
  final verdict   the hidden Verifier, once, host-side, on what /app/submission
                  holds (final_validate/tools/validate_evolve.py)
  checkpoints     the same verdict for every distinct ./version snapshot
                  (final_validate/tools/verdict_checkpoints.py)
  release         the GPU claim; anything still up is stopped by its project
  record          SUMMARY.json and evaluation_progress/<task>.csv
  harvest         every metric into METRICS.json (run and lineage), then every
                  weight a verdict measured is deleted (harvest.py)

`--hours` bounds the agent, not the run: the verdict runs after the deadline.
An exit code is not a verdict.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import agent, boundary, containers, credentials, gpus, harvest, record, telemetry
from . import disk, report, workspace
from .utils import (REPO, agent_slug, read_contract, read_json, sha256_file,
                    until_space, utc_after, utc_now, write_json_atomically)

HARBOR_ENV = REPO / "compression_bench" / "scripts" / "agent" / "harbor_env.py"
IMAGE_FRESHNESS = REPO / "compression_bench" / "scripts" / "agent" / "image_freshness.py"
VALIDATE = REPO / "compression_bench" / "harness" / "final_validate" / "tools" / "validate_evolve.py"
CHECKPOINT_VERDICTS = VALIDATE.with_name("verdict_checkpoints.py")
RECORD_RUN = REPO / "evaluation_progress" / "record_run.py"
CREDENTIAL_REFRESH_S = 60.0
#: The longest timeout subprocess can wait on (poll() takes int milliseconds), with a margin.
POLL_TIMEOUT_MAX_S = (2 ** 31 - 1) // 1000 - 3600


class Cancelled(BaseException):
    """SIGTERM/SIGHUP/SIGINT: stop the agent, preserve the run, skip the verdict."""


#: The signals that cancel a run (SIGINT too: a Ctrl-C cancels, it is not a harness
#: failure).
CANCEL_SIGNALS = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)


class Unbreakable:
    """A stream whose writes never raise: a driver started from a terminal that has
    since gone (EIO) or a closed pipe (EPIPE) must not lose its run over a print."""

    def __init__(self, stream):
        self._stream = stream

    def write(self, text):
        try:
            return self._stream.write(text)
        except (OSError, ValueError):
            return len(text)

    def flush(self):
        try:
            self._stream.flush()
        except (OSError, ValueError):
            pass

    def __getattr__(self, name):
        return getattr(self._stream, name)


class SetupFailed(RuntimeError):
    def __init__(self, kind: str, why: str):
        super().__init__(why)
        self.kind = kind


def resolve_task_dir(task: str, repo: Path = REPO) -> Path:
    """A task directory, id (any case) or 8-hex task hash -> its directory."""
    path = Path(task)
    if (path / "contract.yaml").is_file():
        return path.resolve()
    root = repo / "compression_tasks"
    for candidate in sorted(root.iterdir()) if root.is_dir() else ():
        if not (candidate / "contract.yaml").is_file():
            continue
        if candidate.name.lower() == task.lower():
            return candidate.resolve()
        try:
            contract = read_contract(candidate)
        except Exception:                                   # noqa: BLE001
            continue
        if str(contract.get("task_hash", "")) == task:
            return candidate.resolve()
    raise SystemExit(f"no task {task!r} under {root}")


def model_identity(model_dir: Path, contract: dict) -> dict:
    """The checkpoint is the one the task pins, as far as this host can tell."""
    config = Path(model_dir) / "config.json"
    if not config.is_file():
        raise SetupFailed("MODEL_MISMATCH", f"{config} does not exist")
    observed = sha256_file(config)
    weights = [p for p in Path(model_dir).iterdir()
               if p.suffix in (".safetensors", ".gguf", ".bin") and p.exists()]
    if not weights:
        raise SetupFailed("MODEL_MISMATCH",
                          f"{model_dir} holds config.json but no weight files "
                          f"(*.safetensors/*.gguf/*.bin): an incomplete checkpoint")
    pinned = (contract.get("model") or {})
    parts = Path(model_dir).resolve().parts
    revision_seen = None
    if "snapshots" in parts:
        revision_seen = parts[parts.index("snapshots") + 1] if parts.index("snapshots") + 1 < len(parts) else None
        if pinned.get("revision") and revision_seen and revision_seen != pinned["revision"]:
            raise SetupFailed("MODEL_MISMATCH",
                              f"snapshot revision {revision_seen} != pinned {pinned['revision']}")
    return {"model_dir": str(Path(model_dir).resolve()), "config_sha256": observed,
            "weight_files": len(weights), "snapshot_revision": revision_seen,
            "model_id": pinned.get("id"), "model_revision": pinned.get("revision")}


class Lifecycle:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.task_dir = resolve_task_dir(args.task)
        self.contract = read_contract(self.task_dir)
        self.task_id = self.contract["task_id"]
        budget = self.contract.get("budget") or {}
        self.hours = float(args.hours if args.hours is not None
                           else budget.get("agent_hours", 8))
        self.identity: record.RunIdentity | None = None
        self.journal: record.Journal | None = None
        self.lineage: boundary.Lineage | None = None
        self.claim: gpus.Claim | None = None
        self.container = ""
        self.environment_up = False
        self.staged_credentials: Path | None = None
        self.staged_agent_env: Path | None = None
        self._last_refresh = 0.0
        self.end_reason = ""
        self.sampler: telemetry.ResourceSampler | None = None

    # --- helpers ------------------------------------------------------------------

    def _harbor(self, *argv: str, timeout: float) -> dict:
        command = [os.environ.get("COMPRESSION_BENCH_HARBOR_PY") or sys.executable,
                   str(HARBOR_ENV), *argv]
        self.journal.append("harbor_env_invoked", argv=command[2:])
        got = subprocess.run(command, capture_output=True, text=True,
                             timeout=timeout, check=False)
        log = self.identity.root / f"harbor_env.{argv[0]}.log"
        entry = f"--- {utc_now()} {' '.join(argv)}\n{got.stderr}\n{got.stdout}\n"

        def log_it():
            with log.open("a", encoding="utf-8") as handle:
                handle.write(entry)
        until_space(log_it, f"writing {log.name}")
        lines = [line for line in got.stdout.splitlines() if line.startswith("{")]
        result = json.loads(lines[-1]) if lines else {
            "status": "error", "error": (got.stderr or got.stdout)[-2000:]}
        result.setdefault("returncode", got.returncode)
        return result

    def _project(self) -> str:
        return self.identity.project

    # --- stages ---------------------------------------------------------------------

    def create(self) -> None:
        a = self.args
        namespace = record.SCRATCH if a.dry_run or a.scratch else record.PRODUCTION
        deadline = utc_after(self.hours * 3600)
        lineage_root = Path(a.lineage_root)
        lineage_id = a.lineage_id or record.lineage_id_for(
            self.task_id, agent_slug(f"{a.agent_cli}-{a.agent_model}", a.agent_effort))
        live = lineage_root / lineage_id / ".live-run.json"
        if live.exists():
            holder = read_json(live, {}) or {}
            pid = holder.get("pid")
            if pid and Path(f"/proc/{pid}").exists():
                raise SystemExit(f"lineage {lineage_id} has a live writer: {holder}")
        self.identity = record.allocate(self.task_id, lineage_id=lineage_id,
                                        deadline_utc=deadline, budget_hours=self.hours,
                                        namespace=namespace)
        self.journal = record.Journal(self.identity.root)
        self.lineage = boundary.Lineage(lineage_root, lineage_id)
        write_json_atomically(live, {"run_id": self.identity.run_id, "pid": os.getpid(),
                                     "since_utc": utc_now()})
        lineage_file = self.lineage.dir / "LINEAGE.json"
        if not lineage_file.exists():
            write_json_atomically(lineage_file, {
                "kind": "compression_bench_lineage", "lineage_id": lineage_id,
                "task_id": self.task_id, "agent_cli": a.agent_cli,
                "agent_model": a.agent_model, "agent_effort": a.agent_effort,
                "created_utc": utc_now()})
        self.journal.append(
            "run_created", run_id=self.identity.run_id, task_id=self.task_id,
            task_dir=str(self.task_dir), task_hash=self.identity.task_hash,
            lineage_id=lineage_id, lineage_dir=str(self.lineage.dir),
            namespace=namespace, deadline_utc=deadline, budget_hours=self.hours,
            gpu=a.gpu, model_dir=str(Path(a.model_dir).resolve()),
            debug_model_dir=a.debug_model_dir, agent_cli=a.agent_cli,
            agent_model=a.agent_model, agent_effort=a.agent_effort,
            dry_run=bool(a.dry_run), project=self.identity.project,
            contract_sha256=sha256_file(self.task_dir / "contract.yaml"))
        print(f"run {self.identity.run_id} at {self.identity.root}", flush=True)

    def claim_gpu(self) -> None:
        why = gpus.why_busy(self.args.gpu)
        if why:
            raise SetupFailed("GPU_UNAVAILABLE", why)
        self.claim = gpus.Claim(self.args.gpu, run_id=self.identity.run_id)
        if not self.claim.acquire():
            raise SetupFailed("GPU_UNAVAILABLE",
                              f"GPU {self.args.gpu} is claimed by another run")
        uuids = {i: c["uuid"] for i, c in gpus.sample().items()}
        reserved = gpus.reservations(uuids).get(self.args.gpu, [])
        self.journal.append("gpu_claimed", gpu=self.args.gpu,
                            claim=str(self.claim.info_path),
                            reserved_by_idle_containers=reserved)
        if reserved:
            print(f"warning: GPU {self.args.gpu} is also reserved by container(s) "
                  f"{', '.join(reserved)} (no compute process or memory in use)",
                  file=sys.stderr, flush=True)

    def check_images(self) -> None:
        """The image the task's environment is built FROM is what this
        checkout's sources would build (compression_bench/environments/images.json)."""
        dockerfile = self.task_dir / "environment" / "Dockerfile"
        base = next((line.split()[1] for line in dockerfile.read_text().splitlines()
                     if line.startswith("FROM ")), "")
        images = read_json(REPO / "compression_bench" / "environments" / "images.json", {}) or {}
        key = next((k for k, v in images.items()
                    if isinstance(v, dict) and v.get("tag") == base), None)
        if key is None or images[key].get("pulled"):
            self.journal.append("image_freshness", base=base, status="unmanaged",
                                note="not built from this checkout; not checked")
            return
        got = subprocess.run([sys.executable, str(IMAGE_FRESHNESS), "check", key],
                             capture_output=True, text=True, timeout=1800, check=False)
        self.journal.append("image_freshness", base=base, key=key,
                            status="fresh" if got.returncode == 0 else "stale",
                            output=(got.stdout + got.stderr)[-1000:])
        if got.returncode != 0:
            raise SetupFailed("HARNESS_SETUP_FAILED",
                              f"image {base} is missing or stale: "
                              f"{(got.stdout + got.stderr).strip()[-400:]}")

    def check_disk(self) -> None:
        """Refuse to start where a full disk would fail the run (lifecycle/disk.py)."""
        space = disk.check(self.identity.root)
        self.journal.append("disk_checked", **space)
        if not space["ok"]:
            raise SetupFailed("DISK_LOW", "; ".join(space["problems"]))

    def check_model(self) -> None:
        found = model_identity(Path(self.args.model_dir), self.contract)
        self.journal.append("model_identity", **found)

    def build_workspace(self) -> None:
        a = self.args
        root = self.identity.root
        paths = workspace.layout(root, debug_model=bool(a.debug_model_dir))
        delivered = workspace.deliver(root, task_dir=self.task_dir, task_id=self.task_id,
                                      deadline_utc=self.identity.deadline_utc,
                                      budget_hours=self.hours, run_id=self.identity.run_id,
                                      lineage_id=self.identity.lineage_id)
        if a.seed_workspace:
            source = Path(a.seed_workspace).resolve()
            earlier = read_json(source / "identity.json", {}) or {}
            if earlier.get("lineage_id") != self.identity.lineage_id:
                raise SetupFailed("HARNESS_SETUP_FAILED",
                                  f"{source} is lineage {earlier.get('lineage_id')!r}, not "
                                  f"{self.identity.lineage_id!r}: a run seeds only from its own lineage")
            state = record.current_state(record.Journal(source))
            if state not in record.TERMINAL:
                raise SetupFailed("HARNESS_SETUP_FAILED", f"{source} has not ended (state {state})")
            self.journal.append("workspace_seeded", **workspace.seed(source, root))
        if a.agent_env:
            # An API-key provider route: its env file is the run's only credential;
            # the CLI's login (an Anthropic/OpenAI token) never enters this run.
            if not containers.provider_base_url(a.agent_env):
                raise SetupFailed("HARNESS_SETUP_FAILED",
                                  f"{a.agent_env} sets no {containers.BASE_URL_KEY}")
            self.staged_agent_env = credentials.stage(Path(a.agent_env), root)
        else:
            self.staged_credentials = credentials.stage(credentials.source_path(a.agent_cli), root)
            if self.staged_credentials is not None:
                credentials.placeholder(root, a.agent_cli)
        self.journal.append("workspace_built", paths={k: str(v) for k, v in paths.items()},
                            delivered=delivered,
                            credentials_staged=self.staged_credentials is not None,
                            agent_env_staged=self.staged_agent_env is not None,
                            provider_url=containers.provider_base_url(a.agent_env) if a.agent_env else None)
        record.transition(self.journal, record.WORKSPACE_READY)

    def start_container(self) -> None:
        a = self.args
        argv = ["start", "--task-dir", str(self.task_dir),
                "--run-root", str(self.identity.root), "--project", self._project(),
                "--gpu", str(a.gpu), "--model-dir", str(Path(a.model_dir).resolve()),
                "--agent-cli", a.agent_cli,
                "--build-timeout", str((self.contract.get("budget") or {}).get(
                    "build_timeout_s", 3600))]
        if self.staged_credentials is not None:
            argv += ["--credentials", str(self.staged_credentials)]
        if self.staged_agent_env is not None:
            argv += ["--agent-env", str(self.staged_agent_env)]
        if a.debug_model_dir:
            argv += ["--debug-model-dir", str(Path(a.debug_model_dir).resolve())]
        self.environment_up = True        # from here on, release() stops it
        result = self._harbor(*argv, timeout=4 * 3600)
        if result.get("status") != "started":
            raise SetupFailed("HARNESS_SETUP_FAILED",
                              f"Harbor could not start the environment: {result.get('error')}")
        self.container = result["main_container"]
        self.journal.append("container_started", container=self.container,
                            project=self._project(), services=result.get("services"),
                            allowed_hosts=result.get("allowed_hosts"),
                            cli=result.get("cli"), start_seconds=result.get("start_seconds"))

    def boot(self) -> None:
        checks = workspace.boot_contract(
            self.container, agent_cli=self.args.agent_cli,
            provider_url=(containers.provider_base_url(self.args.agent_env) if self.args.agent_env
                          else containers.PROVIDER_PROBE_URL[self.args.agent_cli]))
        ok, summary = workspace.summary_of(checks)
        self.journal.append("boot_checked", ok=ok, summary=summary, checks=checks)
        if not ok:
            raise SetupFailed("HARNESS_SETUP_FAILED", f"boot proof failed: {summary}")

    def on_tick(self) -> None:
        """Called every few seconds while the agent runs."""
        for request in boundary.pending_requests(self.identity.root):
            self.take_version(request)
        now = time.monotonic()
        if (self.staged_credentials is not None
                and now - self._last_refresh >= CREDENTIAL_REFRESH_S):
            self._last_refresh = now
            if credentials.refresh(self.staged_credentials,
                                   credentials.source_path(self.args.agent_cli)):
                self.journal.append("credentials_refreshed")

    def take_version(self, request: Path) -> None:
        asked = boundary.read_request(request)
        if asked["action"] != "publish":
            boundary.acknowledge(request, {"status": "refused", "why": asked.get("why")})
            self.journal.append("version_refused", request=request.name, why=asked.get("why"))
            return
        try:
            same = self.lineage.unchanged_since(self.identity.root / "submission")
        except Exception as exc:                            # noqa: BLE001 - then record it
            same = None
            self.journal.append("version_compare_failed", why=str(exc)[:300])
        if same:
            # Nothing changed since the last version: recording it again measures nothing.
            boundary.acknowledge(request, {
                "status": "unchanged", "version": same, "lineage": self.identity.lineage_id,
                "why": f"/app/submission computes with exactly the files of {same} "
                       "(interpreter caches and tokenizer files aside); no version recorded"})
            self.journal.append("version_unchanged", request=request.name, same_as=same,
                                note=asked["note"][:500])
            return
        try:
            captured = self.lineage.capture(self.identity.root / "submission",
                                            note=asked["note"], run_id=self.identity.run_id,
                                            source="agent")
        except Exception as exc:                            # noqa: BLE001
            boundary.acknowledge(request, {"status": "error", "why": str(exc)[:300]})
            self.journal.append("version_failed", request=request.name, why=str(exc)[:300])
            return
        ack = {"status": "captured", "version": captured["version"],
               "lineage": self.identity.lineage_id, "files": captured["files"],
               "total_bytes": captured["total_bytes"],
               "manifest": "SHA-256 digests are being computed by the harness",
               "copied_files": len(captured["copied_files"]),
               "snapshot": captured["snapshot"]["status"]}
        boundary.acknowledge(request, ack)
        try:
            # The agent's memory as it stood when it took this version.
            memory = telemetry.copy_memory(self.identity.root,
                                           self.lineage.dir / captured["name"] / "MEMORY")
        except Exception as exc:                            # noqa: BLE001 - a lost note, not a lost run
            memory = {"error": str(exc)[:300]}
        self.journal.append("version_captured", version=captured["version"],
                            name=captured["name"], note=captured["note"],
                            files=captured["files"], total_bytes=captured["total_bytes"],
                            snapshot=captured["snapshot"], memory=memory)

    def develop(self) -> None:
        record.transition(self.journal, record.AGENT_RUNNING)
        a = self.args
        self.end_reason = agent.develop(
            identity=self.identity, journal=self.journal, container=self.container,
            task_dir=self.task_dir, agent_cli=a.agent_cli, model=a.agent_model,
            effort=a.agent_effort, on_tick=self.on_tick,
            env_file=containers.AGENT_ENV_FILE if self.staged_agent_env is not None else "",
            agent_env=self.staged_agent_env, extend_deadline=self.extend_deadline)
        self.journal.append("agent_phase_ended", reason=self.end_reason,
                            remaining_seconds=round(agent.remaining_seconds(self.identity), 1))

    def extend_deadline(self, deadline_utc: str) -> record.RunIdentity:
        """A credit hold's time given back: the new deadline wherever the run's clock is read
        (`identity.json`, the agent's `WORKSPACE.json`); the journal keeps the old one."""
        self.identity = dataclasses.replace(self.identity, deadline_utc=deadline_utc)
        write_json_atomically(self.identity.root / "identity.json", self.identity.to_dict())
        workspace_file = self.identity.root / "work" / "WORKSPACE.json"
        data = read_json(workspace_file, None)
        if isinstance(data, dict):
            write_json_atomically(workspace_file, {**data, "deadline_utc": deadline_utc})
        self.journal.append("deadline_extended", deadline_utc=deadline_utc, why="credit hold")
        return self.identity

    def finish(self) -> None:
        """The final capture, then stop the environment (frees the lock)."""
        for request in boundary.pending_requests(self.identity.root):
            self.take_version(request)
        try:
            final = self.lineage.capture(self.identity.root / "submission",
                                         note="the run ended; the submission the "
                                              "Verifier evaluates",
                                         run_id=self.identity.run_id, source="final",
                                         background=False)
            self.journal.append("final_captured", version=final["version"],
                                name=final["name"], files=final["files"],
                                total_bytes=final["total_bytes"])
        except Exception as exc:                            # noqa: BLE001
            self.journal.append("final_capture_failed", why=str(exc)[:500])
        self.lineage.join(timeout=3600)
        self.stop_environment()

    def stop_environment(self) -> None:
        if not self.environment_up:
            return
        result = self._harbor("stop", "--run-root", str(self.identity.root), "--project",
                              self._project(), "--remove-images", timeout=1800)
        self.journal.append("container_stopped", status=result.get("status"),
                            remaining=result.get("remaining"))
        if result.get("status") == "stopped":
            self.environment_up = False

    def final_verdict(self) -> str:
        command = [sys.executable, str(VALIDATE), "--run", str(self.identity.root)]
        self.journal.append("final_validation_started", argv=command)
        budget = self.contract.get("budget") or {}
        timeout = 3 * 3600 + int(budget.get("verifier_timeout_s", 1800))
        try:
            got = subprocess.run(command, capture_output=True, text=True,
                                 timeout=timeout, check=False)
            output = got.stdout + got.stderr
        except subprocess.TimeoutExpired as exc:
            output = f"validate_evolve timed out after {timeout}s: {exc}"
        until_space(lambda: (self.identity.root / "final_validation.log").write_text(output, encoding="utf-8"),
                    "writing final_validation.log")
        # The verdict is the Verifier's status, in its own words (SCORED | ZERO),
        # or NOT_RUN when it produced none for this run (the log says why).
        final = read_json(self.lineage.dir / "FINAL.json", {}) or {}
        verdict = final.get("status") if final.get("run_id") == self.identity.run_id else None
        verdict = verdict or "NOT_RUN"
        self.journal.append("final_validation", verdict=verdict, reward=final.get("reward")
                            if verdict != "NOT_RUN" else None)
        return verdict

    def checkpoint_verdicts(self) -> None:
        """The full hidden verdict on every checkpoint, on this run's card; a measured
        checkpoint's weights go (verdict_checkpoints.py), an unmeasured one stays."""
        command = [sys.executable, str(CHECKPOINT_VERDICTS), "--run", str(self.identity.root),
                   "--gpu", str(self.args.gpu)]
        self.journal.append("checkpoint_verdicts_started", argv=command)
        try:
            got = subprocess.run(command, capture_output=True, text=True, check=False,
                                 timeout=self.checkpoint_verdicts_timeout())
            output = got.stdout + got.stderr
        except subprocess.TimeoutExpired as exc:
            output = f"verdict_checkpoints timed out: {exc}"
        except (OSError, ValueError, OverflowError) as exc:
            # The final verdict stands; an unmeasured checkpoint stays for a re-run.
            output = f"verdict_checkpoints did not run: {type(exc).__name__}: {exc}"
        until_space(lambda: (self.identity.root / "checkpoint_verdicts.log").write_text(output, encoding="utf-8"),
                    "writing checkpoint_verdicts.log")
        last = (output.strip().splitlines() or ["{}"])[-1]
        try:
            summary = json.loads(last)
        except json.JSONDecodeError:
            summary = {"error": last[:500]}
        self.journal.append("checkpoint_verdicts_finished", **summary)

    def checkpoint_verdicts_timeout(self) -> int:
        """One verdict's bound (validate_evolve's) per checkpoint this run recorded.
        subprocess waits through poll(), whose millisecond timeout is a C int: a bound
        past ~24.8 days raises OverflowError, so it is capped."""
        budget = self.contract.get("budget") or {}
        each = 3 * 3600 + int(budget.get("verifier_timeout_s", 1800))
        count = len(self.journal.events("version_captured")) + len(self.journal.events("final_captured"))
        return min(max(1, count) * each, POLL_TIMEOUT_MAX_S)

    def record_progress(self) -> None:
        if not self.identity.is_production or not RECORD_RUN.exists():
            return
        got = subprocess.run([sys.executable, str(RECORD_RUN), "--run",
                              str(self.identity.root)], capture_output=True,
                             text=True, timeout=300, check=False)
        self.journal.append("progress_recorded", returncode=got.returncode,
                            output=(got.stdout + got.stderr)[-500:])

    def harvest_and_clean(self) -> None:
        """Every metric into METRICS.json, then the weights go (harvest.py)."""
        try:
            metrics = harvest.harvest(self.identity.root,
                                      clean_weights=not self.args.keep_weights,
                                      keep=self.args.keep_checkpoints)
            cleanup = metrics.get("cleanup") or {}
            print(f"metrics: {self.identity.root / 'METRICS.json'}; freed "
                  f"{(cleanup.get('freed_bytes') or 0) / 2**30:.1f} GiB, kept checkpoints "
                  f"{cleanup.get('checkpoints_kept')}", flush=True)
        except Exception as exc:                            # noqa: BLE001
            self.journal.append("harvest_failed", why=f"{type(exc).__name__}: {exc}"[:500])

    def release(self) -> None:
        if self.sampler is not None:
            self.sampler.stop()
            self.journal.append("resources_sampled", samples=self.sampler.count,
                                every_seconds=self.sampler.every)
            self.sampler = None
        if self.staged_agent_env is not None:
            # The provider key is needed only while the agent runs; the verdict needs none.
            self.staged_agent_env.unlink(missing_ok=True)
            self.journal.append("agent_env_removed")
            self.staged_agent_env = None
        try:
            self.stop_environment()
        except Exception as exc:                            # noqa: BLE001
            if self.journal is not None:
                self.journal.append("release_error", what="environment", why=str(exc)[:300])
        if self.claim is not None:
            self.claim.release()
            self.journal.append("gpu_released", gpu=self.claim.index)
            self.claim = None
        if self.lineage is not None and self.identity is not None:
            live = self.lineage.dir / ".live-run.json"
            holder = read_json(live, {}) or {}
            if holder.get("run_id") == self.identity.run_id:
                live.unlink(missing_ok=True)

    # --- the whole run --------------------------------------------------------------

    def run(self) -> int:
        def cancel(signum, _frame):
            raise Cancelled(signal.Signals(signum).name)
        previous = {sig: signal.getsignal(sig) for sig in CANCEL_SIGNALS}
        streams = sys.stdout, sys.stderr
        for sig in CANCEL_SIGNALS:
            signal.signal(sig, cancel)
        sys.stdout, sys.stderr = Unbreakable(sys.stdout), Unbreakable(sys.stderr)
        self.create()
        verdict = None
        try:
            try:
                self.claim_gpu()
                self.check_disk()
                self.check_model()
                self.check_images()
                self.build_workspace()
                self.start_container()
                self.sampler = telemetry.ResourceSampler(self.identity.root, gpu=self.args.gpu,
                                                         project=self._project())
                self.sampler.start()
                self.boot()
            except SetupFailed as exc:
                self.journal.append("setup_failed", kind=exc.kind, why=str(exc))
                record.transition(self.journal, record.FAILED, kind=exc.kind, why=str(exc))
                print(f"FAILED {exc.kind}: {exc}", file=sys.stderr, flush=True)
                return 2
            if self.args.dry_run:
                self.journal.append("dry_run_complete")
                record.transition(self.journal, record.COMPLETED, dry_run=True)
                print("DRY_RUN_OK", flush=True)
                return 0
            self.develop()
            self.finish()
            if not self.args.no_verdict:
                record.transition(self.journal, record.VERDICT)
                verdict = self.final_verdict()
                self.checkpoint_verdicts()
            record.transition(self.journal, record.COMPLETED, verdict=verdict)
            print(verdict or "NO_VERDICT", flush=True)
            return 0
        except Cancelled as exc:
            self.journal.append("cancelled", signal=str(exc))
            if record.current_state(self.journal) not in record.TERMINAL:
                record.transition(self.journal, record.CANCELLED, signal=str(exc))
            return 130
        except BaseException as exc:                        # noqa: BLE001
            self.journal.append("harness_error", error=f"{type(exc).__name__}: {exc}"[:2000])
            if record.current_state(self.journal) not in record.TERMINAL:
                record.transition(self.journal, record.FAILED, kind="HARNESS",
                                  why=f"{type(exc).__name__}: {exc}"[:500])
            raise
        finally:
            for sig in CANCEL_SIGNALS:              # a second signal must not cut the record short
                signal.signal(sig, signal.SIG_IGN)
            try:
                try:
                    self.release()
                finally:
                    report.write_summary(self.identity, self.journal, verdict=verdict,
                                         lineage_dir=str(self.lineage.dir) if self.lineage else None,
                                         final=read_json(self.lineage.dir / "FINAL.json", None)
                                         if self.lineage else None)
                    if not self.args.dry_run and self.journal is not None:
                        self.record_progress()      # every finished run has a row (MLE-Bench)
                        self.harvest_and_clean()
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
                sys.stdout, sys.stderr = streams


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="compression_bench/scripts/evolve",
                                     description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True,
                        help="task directory, id or 8-hex task hash")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--hours", type=float,
                        help="agent budget; default the contract's budget.agent_hours")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--debug-model-dir")
    parser.add_argument("--agent-cli", choices=("claude", "codex"), default="claude")
    parser.add_argument("--agent-model", default="")
    parser.add_argument("--agent-effort", default="")
    parser.add_argument("--agent-env",
                        help="an API-key provider route: a KEY=VALUE file (ANTHROPIC_BASE_URL, "
                             "ANTHROPIC_API_KEY, ANTHROPIC_MODEL, ...) staged into the run's "
                             "secrets/, bound read-only and sourced by the launch; replaces the "
                             "CLI's login and narrows egress to the base URL's host")
    parser.add_argument("--seed-workspace",
                        help="an ended run of this lineage whose work/ files (code, notes, logs, the "
                             "CLI's memory notes; never weights, transcripts or delivered files) are "
                             "copied into this fresh run's workspace")
    parser.add_argument("--lineage-root", default=str(REPO / "compression_output"))
    parser.add_argument("--lineage-id", help="continue an existing lineage")
    parser.add_argument("--dry-run", action="store_true",
                        help="build and boot, then stop; scratch namespace, no agent")
    parser.add_argument("--scratch", action="store_true",
                        help="allocate in the scratch namespace (tests, smoke checks)")
    parser.add_argument("--no-verdict", action="store_true",
                        help="skip the hidden verdict (development of the harness only)")
    parser.add_argument("--keep-weights", action="store_true",
                        help="write METRICS.json but delete no weights after the run")
    parser.add_argument("--keep-checkpoints", type=int,
                        help="measured weight checkpoints the lineage keeps after the run "
                             "(default COMPRESSION_BENCH_KEEP_CHECKPOINTS, 0; unmeasured ones always stay)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.hours is not None and args.hours <= 0:
        raise SystemExit("--hours must be positive")
    return Lifecycle(args).run()
