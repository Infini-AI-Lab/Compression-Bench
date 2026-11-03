#!/usr/bin/env python3
"""The evolve scheduler: keeps an assigned sweep running until every item has finished.

    python3 compression_bench/scripts/scheduler.py --queue compression_bench/evolve_space/scheduler/queue.json
        [--max-live 4] [--tick 60] [--once] [--dry-run]

One daemon per host. Other checkouts of the harness that share the host's GPUs and disk are named in
COMPRESSION_BENCH_SHARED_CHECKOUTS (paths separated by `:`); their drivers count against the cap and their
ended runs are cleaned like this checkout's.

  cap         at most --max-live evolve drivers run at once, counting every checkout's drivers
              (other sessions' included); nothing launches above it.
  sweep       the queue file lists items {id, task, agent, status, attempts}: `queued`, `running` (with its
              `run_dir`), `done`, `gave_up`. The scheduler launches queued items as slots, a free GPU
              and disk allow, follows each run to its end, and relaunches one that a SYSTEM fault
              ended (a GPU taken meanwhile, a failed setup, a killed container, a dead driver, a full
              disk) -- seeded from its own earlier workspace, in its own lineage -- up to --attempts
              times. A run that ended on its own merits (a verdict, scored or ZERO) is done. It exits
              when no item is left to do, writing `report.json` beside the queue file.
  disk        every tick: free space on the checkout's filesystem and Docker's. Below --low-gib it
              cleans (finished runs' weights, cold checkpoints of live runs, Docker leftovers); below
              --launch-gib it launches nothing; below --critical-gib it cleans checkpoints at once.
  checkpoints `checkpoint_janitor` every --clean-every seconds: the newest checkpoint of a live run stays,
              cold ones are deleted after their data is recorded.
  containers  compose projects of ended runs, of runs whose driver is gone (the run is closed FAILED,
              recorded, and queued again), and exited leftovers. Only `compression-bench-*` projects of a run
              directory of a checkout are touched; baseline containers (`bf16-base`) never.
  agents      an item's `agent` names an entry of AGENTS: the CLI, model and effort it launches with, and
              for a provider route the KEY=VALUE env file (AGENT_ENV of evolve-task.sh) it needs, looked
              up in COMPRESSION_BENCH_CONFIG_DIR (default ~/.config/compression_bench): a filled-in copy
              of compression_bench/scripts/agent/routes/<name>.env.example, named <name>.env. An item
              whose route file is missing waits.
  GPUs        --exclude-gpus (default: COMPRESSION_BENCH_EXCLUDED_GPUS) are never used.

Never signals a process it did not start. Pid files are the only way a process is stopped; the scheduler
itself stops only containers (by compose project name), never drivers.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "compression_bench" / "harness"))
sys.path.insert(0, str(HERE.parent))

from lifecycle import gpus, record, verdicts  # noqa: E402
from lifecycle.disk import free_gib  # noqa: E402
from lifecycle.utils import read_contract, read_json, utc_now, write_json_atomically  # noqa: E402
import bf16_copy_guard  # noqa: E402
import case_study_keep  # noqa: E402
import checkpoint_janitor  # noqa: E402
import model_for_task  # noqa: E402

REPOS = [REPO, *(Path(p) for p in os.environ.get("COMPRESSION_BENCH_SHARED_CHECKOUTS", "").split(":") if p)]
CONFIG = Path(os.environ.get("COMPRESSION_BENCH_CONFIG_DIR") or Path.home() / ".config" / "compression_bench")
AGENTS = {
    "sonnet": {"cli": "claude", "model": "claude-sonnet-5-5", "effort": "max", "env": None},
    "opus": {"cli": "claude", "model": "claude-opus-5-5", "effort": "max", "env": None},
    "glm": {"cli": "claude", "model": "accounts/fireworks/models/glm-5p3", "effort": "max",
            "env": CONFIG / "glm-5p3-fireworks.env"},
    "glm-flash": {"cli": "claude", "model": "accounts/fireworks/models/glm-5p3-flash", "effort": "max",
            "env": CONFIG / "glm-5p3-flash-fireworks.env"},
    "kimi": {"cli": "claude", "model": "moonshotai/kimi-k3.1-pro", "effort": "max",
             "env": CONFIG / "kimi-k3p1-pro-openrouter.env"},
    "minimax": {"cli": "claude", "model": "accounts/fireworks/models/minimax-m3p1-flash", "effort": "max",
                "env": CONFIG / "minimax-m3p1-flash-fireworks.env"},
    "deepseek": {"cli": "claude", "model": "accounts/fireworks/models/deepseek-v4p1-flash-1017", "effort": "max",
                 "env": CONFIG / "deepseek-v4p1-flash-fireworks.env"},
}
#: Why a run ended: these are the system's fault, not the agent's result.
SYSTEM_KINDS = frozenset({"GPU_UNAVAILABLE", "HARNESS_SETUP_FAILED", "HARNESS"})
DRIVER = re.compile(r"^\s*(\d+)\s+\S*python\S*\s+(\S*/compression_bench/scripts/evolve)\s+--task\s+(\S+)\s+--gpu\s+(\d+)")
PROJECT = re.compile(r"^(compression-bench-([0-9a-f]{8}-\d{8}t\d{6}z-[0-9a-f]{6}))-")


# --- observation ---------------------------------------------------------------------------

def parse_drivers(ps_output: str) -> list[dict]:
    """Evolve drivers in `ps -o pid=,args=` output: {pid, repo, task, gpu}."""
    out = []
    for line in ps_output.splitlines():
        found = DRIVER.match(line)
        if found:
            script = Path(found.group(2))
            out.append({"pid": int(found.group(1)), "repo": str(script.parents[2]),
                        "task": Path(found.group(3)).name, "gpu": int(found.group(4))})
    return out


def drivers() -> list[dict]:
    got = subprocess.run(["ps", "-u", str(os.getuid()), "-o", "pid=,args="], capture_output=True, text=True)
    return parse_drivers(got.stdout)


def run_state(run_dir: Path) -> dict:
    """{state, kind, why, has_summary} from the journal (a torn line is skipped)."""
    state, kind, why, phase = None, None, None, None
    try:
        for line in (Path(run_dir) / "journal.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "state":
                state, kind, why = event.get("to"), event.get("kind"), event.get("why")
            elif event.get("event") == "agent_phase_ended":
                phase = event.get("reason")
    except OSError:
        pass
    return {"state": state, "kind": kind, "why": why, "phase": phase,
            "has_summary": (Path(run_dir) / "SUMMARY.json").exists()}


def had_agent(run_dir: Path) -> bool:
    try:
        return '"agent_launched"' in (Path(run_dir) / "journal.jsonl").read_text(encoding="utf-8")
    except OSError:
        return False


def alive_run_ids(claims: Path | None = None) -> set[str]:
    """Runs whose driver holds a card claim and is alive."""
    live = set()
    for path in Path(claims or gpus.CLAIMS).glob("*.json"):
        info = read_json(path, {}) or {}
        pid = info.get("pid")
        if pid:
            try:
                os.kill(pid, 0)
                live.add(info.get("run_id"))
            except OSError:
                pass
    return live


def classify(run: dict) -> str:
    """What an ended run means for its item: `done`, `retry` (the system's fault) or `running`."""
    state = run["state"]
    if state is None or state not in record.TERMINAL:
        return "running"
    if state == record.COMPLETED:
        # The agent phase ended because the agent could never reach its model (a credential or billing
        # fault): the verdict measured an empty submission, so this is no result.
        return "retry" if run.get("phase") == "terminal" else "done"
    if state == record.CANCELLED:
        return "retry"
    return "retry" if run["kind"] in SYSTEM_KINDS or run["kind"] is None else "done"


def disk_tier(free: float, *, low: float, launch: float, critical: float) -> str:
    if free < critical:
        return "critical"
    if free < low:
        return "low"
    return "ok" if free >= launch else "tight"


def may_launch(*, live: int, max_live: int, tier: str, queued: int, since_last_launch: float, spacing: float) -> bool:
    return queued > 0 and live < max_live and tier == "ok" and since_last_launch >= spacing


# --- docker (our side) ----------------------------------------------------------------------

def docker(*argv: str, timeout: float = 120) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *argv], capture_output=True, text=True, timeout=timeout)


def our_projects() -> dict[str, str]:
    """compose project -> run id (lower case) of every `compression-bench-*` container, baselines excluded."""
    got = docker("ps", "-a", "--format", "{{.Names}}")
    projects = {}
    for name in got.stdout.split():
        if "bf16-base" in name:
            continue
        found = PROJECT.match(name)
        if found:
            projects[found.group(1)] = found.group(2)
    return projects


def run_dir_for(run_id: str, repos=None) -> Path | None:
    for repo in (repos or REPOS):
        for path in (Path(repo) / "compression_bench" / "evolve_space" / "evolve").glob("*/*"):
            if path.name.lower() == run_id:
                return path
    return None


def compose_down(project: str) -> None:
    docker("compose", "-p", project, "down", "--remove-orphans", "--rmi", "local", timeout=300)


# --- the scheduler -------------------------------------------------------------------------

def verdict_in_progress(run_dir: Path) -> bool:
    """A checkpoint verdict is reading its copy of a snapshot (verdict_checkpoints.py removes the copy when done)."""
    return any((run_dir / "checkpoint-verdicts").glob("*/submission"))


def measured_versions(lineage_dir: Path) -> set[str]:
    """Versions a verdict measured (lifecycle.verdicts.measured), the rule of the run's own cleanup (harvest.clean):
    only their snapshots may lose their weights. One never measured, or whose verdict failed on the host's account
    (a busy GPU, a full disk: NOT_RUN), stays for a re-run."""
    out = set()
    try:
        rows = [json.loads(line) for line in (lineage_dir / "versions.jsonl").read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return out
    final = read_json(lineage_dir / "FINAL.json", {}) or {}
    for row in rows:
        verdict = (read_json(lineage_dir / row.get("name", "") / "VERSION.json", {}) or {}).get("verdict") or {}
        if verdicts.measured(verdict) or (row.get("source") == "final" and final.get("run_id") == row.get("run_id")
                                          and verdicts.measured(final)):
            out.add(row.get("version"))
    return out


def final_measured(run_dir: Path) -> bool:
    """The run's final verdict measured its submission (METRICS.json); otherwise the submission stays for a re-run."""
    final = ((read_json(run_dir / "METRICS.json", {}) or {}).get("verdicts") or {}).get("final") or {}
    return verdicts.measured(final)


class Scheduler:
    def __init__(self, args):
        self.args = args
        os.environ.setdefault("COMPRESSION_BENCH_EXCLUDED_GPUS", args.exclude_gpus)
        self.keep_on_failure = {r.strip() for r in args.keep_on_failure.split(",") if r.strip()}
        self.queue_path = Path(args.queue)
        self.dir = self.queue_path.parent
        self.log_path = self.dir / "scheduler.log"
        self.queue = read_json(self.queue_path, None)
        if not isinstance(self.queue, dict) or "items" not in self.queue:
            raise SystemExit(f"{self.queue_path} is not a queue file")
        self.last_launch = 0.0
        self.last_clean = 0.0
        self.downloads: dict[str, subprocess.Popen] = {}
        self.last_alert = {}
        self.probe = None
        self.last_size_check = 0.0
        self.last_copy_check = 0.0
        self.copy_indexes: dict[str, dict] = {}     # BF16 checkpoint -> bf16_copy_guard.reference_index
        self.stuck_ticks = 0
        self.cooldown: dict[str, float] = {}
        self.children: list[subprocess.Popen] = []

    def guarded(self, name: str, step, default=None):
        """Run one housekeeping step; a failure is logged and never stops the others (or the sweep)."""
        try:
            return step()
        except Exception as exc:                                # noqa: BLE001
            self.log(f"ERROR {name}: {type(exc).__name__}: {str(exc)[:240]}")
            return default

    # logging / persistence
    def log(self, message: str) -> None:
        line = f"{utc_now()} {message}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def save(self) -> None:
        write_json_atomically(self.queue_path, self.queue)

    # items
    def items(self, status: str):
        return [i for i in self.queue["items"] if i["status"] == status]

    def task_dir(self, item) -> Path:
        return REPO / "compression_tasks" / item["task"]

    def model_missing(self, item) -> bool:
        got = subprocess.run([sys.executable, str(REPO / "compression_bench/scripts/model_for_task.py"), str(self.task_dir(item))],
                             capture_output=True, text=True)
        return got.returncode != 0

    def follow(self) -> None:
        """Move running items to done / queued (retry) from their run's state."""
        alive = alive_run_ids()
        for item in self.items("running"):
            run_dir = Path(item["run_dir"])
            run = run_state(run_dir)
            verdict = classify(run)
            if verdict == "running" and run_dir.name not in alive and self.driver_gone(run_dir):
                if run_dir.name in self.keep_on_failure:
                    # Kept as it was: its containers, workspace and weights stay, the janitor keeps off, nothing is
                    # relaunched; the item stays running until someone resumes or ends it by hand.
                    if not (run_dir / checkpoint_janitor.KEPT_FILE).exists():
                        write_json_atomically(run_dir / checkpoint_janitor.KEPT_FILE, {
                            "since_utc": utc_now(), "state": run["state"],
                            "why": "the driver is gone; --keep-on-failure keeps the run as it was"})
                        self.alert(f"kept-{run_dir.name}", f"KEPT {item['id']}: the driver of {run_dir.name} is gone in "
                                   f"state {run['state']}; its containers, workspace and weights are kept and nothing is "
                                   f"relaunched (--keep-on-failure): resume or end it by hand")
                    continue
                self.log(f"ORPHAN {item['id']}: driver of {run_dir.name} is gone in state {run['state']}; closing it")
                self.close_orphan(run_dir)
                run = run_state(run_dir)
                verdict = classify(run)
            if verdict == "running":
                continue
            summary = read_json(run_dir / "SUMMARY.json", {}) or {}
            final = summary.get("final") or {}
            item["history"] = (item.get("history") or []) + [{
                "run": run_dir.name, "state": run["state"], "kind": run["kind"], "status": final.get("status"),
                "reward": final.get("reward"), "why": (run["why"] or "")[:200]}]
            if verdict == "done":
                item["status"] = "done"
                self.log(f"DONE {item['id']}: {run_dir.name} {run['state']} {final.get('status')} reward {final.get('reward')}")
            elif item["attempts"] >= self.args.attempts:
                item["status"] = "gave_up"
                self.log(f"GAVE UP {item['id']} after {item['attempts']} attempts: {run['kind']} {(run['why'] or '')[:120]}")
            else:
                item["status"] = "queued"
                item["seed"] = str(run_dir) if had_agent(run_dir) else item.get("seed")
                self.cooldown[item["id"]] = time.time() + self.args.retry_wait
                self.log(f"RETRY {item['id']}: {run_dir.name} ended {run['state']}/{run['kind']} "
                         f"({(run['why'] or '')[:100]}); queued again, seeded from {Path(item['seed']).name if item.get('seed') else 'nothing'}")
            self.save()

    def driver_gone(self, run_dir: Path) -> bool:
        """The run is old enough to have claimed a card (the caller has found no live claim naming it)."""
        try:
            age = time.time() - (run_dir / "identity.json").stat().st_mtime
        except OSError:
            return False
        return bool(read_json(run_dir / "identity.json", {})) and age > 600

    def close_orphan(self, run_dir: Path) -> None:
        """A run whose driver died: stop its containers, end it FAILED, and record it like a driver would."""
        from lifecycle import harvest, report
        for project, run_id in our_projects().items():
            if run_id == run_dir.name.lower():
                compose_down(project)
        journal = record.Journal(run_dir)
        record.transition(journal, record.FAILED, kind="HARNESS", why="the driver died; closed by the scheduler")
        if self.args.dry_run:
            return
        identity = record.load_identity(run_dir)
        created = [e for e in journal if e["event"] == "run_created"][0]
        lineage_dir = created.get("lineage_dir")
        report.write_summary(identity, journal, verdict=None, lineage_dir=lineage_dir,
                             final=read_json(Path(lineage_dir) / "FINAL.json", None) if lineage_dir else None)
        record_run = run_dir.parents[4] / "evaluation_progress" / "record_run.py"
        if record_run.exists():
            got = subprocess.run([sys.executable, str(record_run), "--run", str(run_dir)], capture_output=True, text=True, timeout=300)
            journal.append("progress_recorded", returncode=got.returncode, output=(got.stdout + got.stderr)[-300:])
        harvest.harvest(run_dir, clean_weights=True, keep=0)

    # launching
    def live_count(self) -> int:
        return len(drivers())

    def next_item(self):
        now = time.time()
        for item in self.items("queued"):
            if self.cooldown.get(item["id"], 0) <= now and self.agent_ready(item):
                return item
        return None

    def agent_ready(self, item) -> bool:
        agent = AGENTS[item["agent"]]
        if agent["env"] is not None and not agent["env"].is_file():
            self.alert(f"route-{item['agent']}", f"WAIT {item['agent']}: missing provider route {agent['env']}")
            return False
        return True

    def launch(self, item) -> bool:
        agent = AGENTS[item["agent"]]
        evolve_dir = REPO / "compression_bench" / "evolve_space" / "evolve"
        before = {p.name for p in evolve_dir.glob(f"{item['task'].lower()}/*")}
        env = dict(os.environ, HOURS=str(self.args.hours), AGENT_CLI=agent["cli"], AGENT_MODEL=agent["model"],
                   AGENT_EFFORT=agent["effort"], COMPRESSION_BENCH_EXCLUDED_GPUS=self.args.exclude_gpus)
        argv = ["bash", str(REPO / "compression_bench/scripts/compress_evolve/evolve-task.sh"), item["task"]]
        if agent["env"]:
            env["AGENT_ENV"] = str(agent["env"])
        if item.get("seed"):
            env["SEED_WORKSPACE"] = item["seed"]
            lineage = (read_json(Path(item["seed"]) / "identity.json", {}) or {}).get("lineage_id")
            if lineage:
                argv += ["--lineage-id", lineage]
        out = self.dir / f"launch-{item['id']}-{int(time.time())}.log"
        item["attempts"] += 1
        if self.args.dry_run:
            self.log(f"DRY-RUN would launch {item['id']} (attempt {item['attempts']}): {' '.join(argv)}")
            item["attempts"] -= 1
            return False
        with out.open("w") as handle:
            process = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                                       start_new_session=True, cwd=str(REPO))
        self.children.append(process)
        run_dir = None
        deadline = time.time() + 180
        while time.time() < deadline and run_dir is None and process.poll() is None:
            time.sleep(5)
            new = [p for p in evolve_dir.glob(f"{item['task'].lower()}/*") if p.name not in before]
            run_dir = new[0] if new else None
        if run_dir is None:
            self.log(f"LAUNCH FAILED {item['id']}: no run was created ({out.name}: "
                     f"{(out.read_text()[-160:] if out.exists() else '').strip()})")
            item["attempts"] -= 1                      # no run existed, so it is not an attempt
            item["status"] = "queued"
            self.cooldown[item["id"]] = time.time() + self.args.retry_wait
            self.save()
            return False
        time.sleep(80)                       # setup takes ~30-60 s; a driver that dies in it is no launch
        run = run_state(run_dir)
        if process.poll() is not None or run["state"] in (record.FAILED, record.CANCELLED):
            self.log(f"LAUNCH FAILED {item['id']}: {run_dir.name} ended in setup: {run['kind']} {(run['why'] or '')[:140]}")
            item["status"] = "running"        # follow() records the end and queues it again
            item["run_dir"] = str(run_dir)
        else:
            item["status"] = "running"
            item["run_dir"] = str(run_dir)
            self.log(f"LAUNCH {item['id']} attempt {item['attempts']}: {run_dir.name} (pid {process.pid})")
        self.last_launch = time.time()
        self.save()
        return True

    def ensure_model(self, item) -> bool:
        """True when the task's checkpoint is on disk; otherwise start its download when the disk can hold it."""
        # A snapshot gains config.json and its first shard before the download is complete.
        # Never launch from it, or start a second download for the other capacity of this model.
        self.downloads = {task: proc for task, proc in self.downloads.items() if proc.poll() is None}
        model = read_contract(self.task_dir(item))["model"]
        for task in self.downloads:
            other = read_contract(REPO / "compression_tasks" / task)["model"]
            if (model["id"], model["revision"]) == (other["id"], other["revision"]):
                return False
        if not self.model_missing(item):
            return True
        task = item["task"]
        if self.downloads:
            return False
        if free_gib(REPO) < self.args.launch_gib + self.args.download_gib:
            self.log(f"WAIT {item['id']}: its checkpoint is missing and the disk cannot hold the download yet")
            return False
        self.log(f"DOWNLOAD checkpoint of {task}")
        cache = model_for_task.hf_cache() / ("models--" + model["id"].replace("/", "--"))
        # Only newly created, sweep-owned caches can be evicted after this model's jobs finish.
        if not cache.exists():
            owned_path = self.dir / "managed-models.json"
            owned = read_json(owned_path, {}) or {}
            owned[str(cache)] = {"id": model["id"], "revision": model["revision"]}
            write_json_atomically(owned_path, owned)
        with (self.dir / "download.log").open("a") as handle:
            self.downloads[task] = subprocess.Popen([sys.executable, str(REPO / "compression_bench/scripts/deploy/fetch_model.py"),
                                                     str(self.task_dir(item))], stdout=handle, stderr=subprocess.STDOUT,
                                                    start_new_session=True)
        return False

    def release_finished_models(self) -> None:
        """Make room for the next large model; never evict a pre-existing checkpoint."""
        owned_path = self.dir / "managed-models.json"
        owned = read_json(owned_path, {}) or {}
        needed = set()
        for item in self.queue["items"]:
            if item["status"] == "running" or (item["status"] == "queued" and self.agent_ready(item)):
                m = read_contract(self.task_dir(item))["model"]
                needed.add((m["id"], m["revision"]))
        for driver in drivers():
            m = read_contract(Path(driver["repo"]) / "compression_tasks" / driver["task"])["model"]
            needed.add((m["id"], m["revision"]))
        for task, proc in self.downloads.items():
            if proc.poll() is None:
                m = read_contract(REPO / "compression_tasks" / task)["model"]
                needed.add((m["id"], m["revision"]))
        for path, model in list(owned.items()):
            cache = Path(path)
            if (model["id"], model["revision"]) in needed:
                continue
            expected = model_for_task.hf_cache() / ("models--" + model["id"].replace("/", "--"))
            if cache != expected or cache.is_symlink():
                continue
            if any(p.name != model["revision"] for p in (cache / "snapshots").glob("*")):
                continue
            if self.args.dry_run:
                self.log(f"DRY-RUN would release sweep-downloaded checkpoint {model['id']}")
                continue
            if cache.exists():
                shutil.rmtree(cache)
            del owned[path]
            write_json_atomically(owned_path, owned)
            self.log(f"RELEASE finished sweep's downloaded checkpoint {model['id']}; next model can use its disk")

    # disk, checkpoints, containers
    def alert(self, key: str, message: str, every: float = 600) -> None:
        """A line the monitor greps for ("ALERT"), repeated at most every `every` seconds per key."""
        if time.time() - self.last_alert.get(key, 0) >= every:
            self.last_alert[key] = time.time()
            self.log(f"ALERT {message}")

    def housekeeping(self) -> str:
        self.guarded("case studies", self.keep_case_studies)     # before any cleaning, so no snapshot is missed
        free = free_gib(REPO)
        tier = disk_tier(free, low=self.args.low_gib, launch=self.args.launch_gib, critical=self.args.critical_gib)
        now = time.time()
        if tier == "critical":
            self.alert("critical", f"disk CRITICAL: {free:.0f} GiB free on the run disk; cleaning checkpoints idle 5+ min, launches held")
        if tier in ("critical", "low") or now - self.last_clean >= self.args.clean_every:
            cold = 5 if tier == "critical" else self.args.cold_min
            freed = (self.guarded("clean_checkpoints", lambda: self.clean_checkpoints(cold), 0.0)
                     + self.guarded("clean_finished", self.clean_finished, 0.0))
            self.last_clean = now
            if freed:
                self.log(f"CLEAN freed {freed:.1f} GiB (disk {tier}, {free:.0f} GiB free)")
                self.stuck_ticks = 0
            elif tier in ("low", "critical"):
                self.stuck_ticks += 1
                if self.stuck_ticks >= 3:
                    self.alert("stuck", f"cleaning freed nothing for {self.stuck_ticks} rounds with {free:.0f} GiB free: "
                                        f"something other than cold checkpoints is filling the disk", every=900)
        self.guarded("clean_containers", self.clean_containers)
        if tier in ("critical", "low"):
            self.guarded("docker prune", lambda: (docker("builder", "prune", "-f", timeout=300),
                                                  docker("image", "prune", "-f", timeout=300)))
        self.guarded("run sizes", self.check_run_sizes)
        self.guarded("bf16 copies", self.guard_bf16_copies)
        return tier

    def keep_case_studies(self) -> None:
        """Every snapshot of the runs in case_studies.json is hard-linked out of the janitor's reach (case_study_keep.py)."""
        for run_dir in case_study_keep.runs():
            made = (case_study_keep.keep(run_dir, dry_run=self.args.dry_run)
                    + case_study_keep.keep_skipped(run_dir, dry_run=self.args.dry_run))
            if made:
                self.log(f"CASE STUDY kept {len(made)} file(s) of {run_dir.name}: "
                         f"{sorted({p.relative_to(case_study_keep.KEPT).parts[1] for p in made})}")

    def check_run_sizes(self) -> None:
        """A live run that grows past --run-alert-gib is named in an alert (every 15 min at most)."""
        if time.time() - self.last_size_check < 900:
            return
        self.last_size_check = time.time()
        for root in checkpoint_janitor.live_runs(REPOS):
            got = subprocess.run(["du", "-sm", str(root / "work")], capture_output=True, text=True, timeout=240)
            if got.returncode == 0 and got.stdout.split():
                gib = int(got.stdout.split()[0]) / 1024
                if gib > self.args.run_alert_gib:
                    self.alert(f"size-{root.name}", f"run {root.name} holds {gib:.0f} GiB in work/ "
                                                    f"(alert at {self.args.run_alert_gib:.0f}); cold checkpoints are cleaned, "
                                                    f"the current one is not", every=1800)

    def guard_bf16_copies(self) -> None:
        """A live run's second copy of its BF16 checkpoint is removed (the read-only original stays readable).
        Every 3 minutes at most; only a byte-identical copy of a shard is ever deleted (bf16_copy_guard.py)."""
        if time.time() - self.last_copy_check < 180:
            return
        self.last_copy_check = time.time()
        indexes = self.copy_indexes
        for root in checkpoint_janitor.live_runs(REPOS):
            created = record.Journal(root).last("run_created") or {}
            model_dir = created.get("model_dir")
            if not model_dir:
                continue
            if model_dir not in indexes:
                indexes[model_dir] = bf16_copy_guard.reference_index(Path(model_dir))
            for path, size in bf16_copy_guard.find_copies(root, indexes[model_dir]):
                path.unlink(missing_ok=True)
                self.log(f"BF16 COPY removed from {root.name}: {path.relative_to(root)} ({size / 2**30:.1f} GiB)")

    def clean_checkpoints(self, cold_min: float) -> float:
        # A tight disk does not wait for any measurement: the checkpoint is recorded with what exists and deleted.
        free = free_gib(REPO)
        tight = free < self.args.launch_gib
        scale = 0.15 if free < self.args.critical_gib else 0.4 if free < self.args.low_gib else 1.0   # budgets shrink with the disk
        args = argparse.Namespace(run=None, dry_run=self.args.dry_run, cold_min=cold_min, min_mb=200, min_group_gb=2,
                                  force=tight, probe_wait_s=self.args.probe_wait_min * 60, budget_scale=scale)
        freed = 0.0
        for name, row in checkpoint_janitor.sweep(args).items():
            freed += row["freed_gib"]
            for error in row.get("errors", []):
                self.log(f"ERROR janitor {name}: {error}")
        return freed

    def clean_finished(self) -> float:
        """Weights of ended runs (and snapshots of lineages without a live run) of every checkout, by the run's own
        cleanup rule (harvest.clean): a snapshot loses its weights only once a verdict measured that version, and a
        submission only once the final verdict measured it, so a verdict that failed on the host's account (a busy GPU,
        a full disk) can be re-run. An ended run whose checkpoint verdict is reading a snapshot, and every case-study
        run (case_studies.json), keep their weights and their lineage's snapshots."""
        freed = 0
        held = set()
        for root in checkpoint_janitor.live_runs(REPOS):
            held.add((read_json(root / "identity.json", {}) or {}).get("lineage_id"))
        cases = {run.name for run in case_study_keep.runs()}
        for repo in REPOS:
            evolve = Path(repo) / "compression_bench" / "evolve_space" / "evolve"
            for summary in evolve.glob("*/*/SUMMARY.json"):
                run = summary.parent
                if time.time() - summary.stat().st_mtime < 900:
                    continue                                   # the driver is still closing
                if verdict_in_progress(run) or run.name in cases:
                    held.add((read_json(run / "identity.json", {}) or {}).get("lineage_id"))
                    continue
                for path in checkpoint_janitor.weight_files(run, 50 * 2**20):
                    if path.relative_to(run).parts[0] == "submission" and not final_measured(run):
                        continue
                    freed += path.stat().st_size
                    if not self.args.dry_run:
                        path.unlink()
            objects = Path(repo) / "compression_output" / ".objects"
            for lineage in (objects.glob("*") if objects.is_dir() else []):
                if lineage.name in held:
                    continue
                measured = measured_versions(Path(repo) / "compression_output" / lineage.name)
                for path in checkpoint_janitor.weight_files(lineage, 50 * 2**20):
                    if path.relative_to(lineage).parts[0] not in measured:
                        continue
                    freed += path.stat().st_size
                    if not self.args.dry_run:
                        path.unlink()
        return freed / 2**30

    def clean_containers(self) -> None:
        for project, run_id in our_projects().items():
            run_dir = run_dir_for(run_id)
            if run_dir is None:
                continue
            run = run_state(run_dir)
            if run["has_summary"] or run["state"] in record.TERMINAL:
                self.log(f"CONTAINERS of the ended run {run_id}: removed")
                if not self.args.dry_run:
                    compose_down(project)
        exited = docker("ps", "-a", "--filter", "status=exited", "--format", "{{.Names}}").stdout.split()
        ours = [n for n in exited if PROJECT.match(n) and "bf16-base" not in n]
        if ours and not self.args.dry_run:
            self.log(f"CONTAINERS removed exited leftovers: {len(ours)}")
            docker("rm", "-f", *ours, timeout=120)

    # the quick outside measurement of a checkpoint about to be deleted (checkpoint_probe.py)
    def probe_step(self, tier: str) -> None:
        if self.probe is not None:
            proc, started, label = self.probe
            if proc.poll() is None:
                if time.time() - started > self.args.probe_cap_min * 60:
                    proc.terminate()                         # a probe we started; its container is removed by the probe
                    self.log(f"PROBE {label} exceeded {self.args.probe_cap_min:g} min; stopped")
                return
            self.log(f"PROBE {label} finished (exit {proc.returncode}) in {int(time.time() - started)} s")
            self.probe = None
        if tier != "ok":
            return                                           # a tight disk cleans, it does not measure
        candidates = []
        for root in checkpoint_janitor.live_runs(REPOS):
            candidates += checkpoint_janitor.probe_candidates(root)
        candidates.sort(key=lambda c: c["eligible_since"])
        for c in candidates:
            task = REPO / "compression_tasks" / (c["task_id"] or "")
            if not task.is_dir():
                continue
            gpu = gpus.pick(samples=2, interval=1)
            if gpu is None:
                return
            lineage = checkpoint_janitor.prototypes_of(c["run"]) / c["lineage_id"]
            out = lineage / "CHECKPOINT_NOTES" / f"{c['version']}.probe.json"
            label = f"{c['lineage_id'][:34]}/{c['version']} on GPU {gpu}"
            if self.args.dry_run:
                self.log(f"DRY-RUN would probe {label}")
                return
            log = (self.dir / "probe.log").open("a")
            proc = subprocess.Popen([sys.executable, str(HERE.parent / "checkpoint_probe.py"), "--task", str(task),
                                     "--snapshot", str(c["snapshot"]), "--out", str(out), "--gpu", str(gpu),
                                     "--minutes", str(self.args.probe_minutes)], stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            self.children.append(proc)
            self.probe = (proc, time.time(), label)
            self.log(f"PROBE {label}: measuring PPL, reward and accuracy on a subset (at most {self.args.probe_minutes:g} min)")
            return

    # the loop
    def tick(self) -> None:
        self.children = [c for c in self.children if c.poll() is None]      # reap finished launches
        self.guarded("follow", self.follow)
        self.guarded("release finished models", self.release_finished_models)
        tier = self.housekeeping()
        self.guarded("probe", lambda: self.probe_step(tier))
        live = self.live_count()
        item = self.next_item()
        queued = len(self.items("queued"))
        if item is not None and may_launch(live=live, max_live=self.args.max_live, tier=tier, queued=queued,
                                           since_last_launch=time.time() - self.last_launch, spacing=self.args.spacing):
            # Missing checkpoints or provider routes must not strand ready tasks behind them.
            for candidate in self.items("queued"):
                if self.cooldown.get(candidate["id"], 0) > time.time() or not self.agent_ready(candidate):
                    continue
                if self.ensure_model(candidate):
                    self.launch(candidate)
                    break
        elif queued and live < self.args.max_live and tier != "ok":
            self.log(f"HOLD launches: disk {tier} ({free_gib(REPO):.0f} GiB free)")

    def finished(self) -> bool:
        return not self.items("queued") and not self.items("running")

    def report(self) -> None:
        rows = []
        for item in self.queue["items"]:
            rows.append({k: item.get(k) for k in ("id", "task", "agent", "status", "attempts", "history")})
        write_json_atomically(self.dir / "report.json", {"written_utc": utc_now(), "items": rows})
        self.log("SWEEP FINISHED: " + ", ".join(f"{r['id']}={r['status']}" for r in rows))

    def run(self) -> int:
        (self.dir / "scheduler.pid").write_text(str(os.getpid()) + "\n")
        self.log(f"scheduler started: max {self.args.max_live} live, {len(self.queue['items'])} items, "
                 f"checkouts {', '.join(str(r) for r in REPOS)}")
        while True:
            try:
                self.tick()
            except Exception as exc:                            # noqa: BLE001 - a bad tick never stops the sweep
                self.log(f"TICK ERROR {type(exc).__name__}: {str(exc)[:300]}")
            if self.finished():
                self.report()
                return 0
            if self.args.once:
                return 0
            time.sleep(self.args.tick)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--max-live", type=int, default=4)
    parser.add_argument("--tick", type=float, default=60)
    parser.add_argument("--spacing", type=float, default=300, help="seconds between launches")
    parser.add_argument("--hours", type=float, default=8)
    parser.add_argument("--attempts", type=int, default=4, help="launches per item before it is given up")
    parser.add_argument("--retry-wait", type=float, default=600)
    parser.add_argument("--exclude-gpus", default=os.environ.get("COMPRESSION_BENCH_EXCLUDED_GPUS", ""),
                        help="GPU indices never used (comma-separated; default: COMPRESSION_BENCH_EXCLUDED_GPUS)")
    parser.add_argument("--keep-on-failure", default="",
                        help="run ids (comma-separated) whose dead driver leaves the run as it was: containers, workspace "
                             "and weights kept, no relaunch, the janitor off (KEPT_AFTER_FAILURE.json)")
    parser.add_argument("--low-gib", type=float, default=150, help="clean finished runs' weights below this")
    parser.add_argument("--launch-gib", type=float, default=200, help="no launch below this")
    parser.add_argument("--critical-gib", type=float, default=60)
    parser.add_argument("--run-alert-gib", type=float, default=250, help="name a live run whose work/ is larger")
    parser.add_argument("--download-gib", type=float, default=240, help="room a missing checkpoint's download needs")
    parser.add_argument("--clean-every", type=float, default=120)
    parser.add_argument("--probe-minutes", type=float, default=9, help="in-container time limit of one probe")
    parser.add_argument("--probe-cap-min", type=float, default=14, help="wall-clock limit of one probe, container start included")
    parser.add_argument("--probe-wait-min", type=float, default=45,
                        help="minutes a superseded checkpoint waits for its probe before it is deleted anyway")
    parser.add_argument("--cold-min", type=float, default=20)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return Scheduler(parser.parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(main())
