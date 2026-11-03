"""The run's workspace: its directory layout, what is delivered, and the boot proof.

Layout (see `compression_bench/docs/INTERFACES.md`):

    <run>/.submission.lock        bound read-only at /run/compression-bench-submission.lock
    <run>/work/                   /app/work, the agent's scratch; $HOME is .agent-home,
                                  $TMPDIR is .tmp (not the shared Docker root disk)
    <run>/submission/             /app/submission, the single copy of the weights
    <run>/work/debug-submission/  /app/debug-submission, only with a debug model
    <run>/verifier/               the hidden verdict's /logs/verifier
    <run>/public-checks/          checks.jsonl, every public check (Checker-only mount)
    <run>/secrets/                the staged credentials (0700)

Delivered into `work/`: `AGENT_TASK.md` (the task's instruction.md, verbatim),
`WORKSPACE.json`, and two tools, `version` and `clock`; into the CLI's home,
the `compression-papers` skill (`.agent-home/.claude/skills/compression-papers/`,
a copy of `compression_bench/environments/agent-base/papers/skill/`, the same source the
agent image bakes into `/opt/compression-papers/skill/`). No harness code and no
evaluation input is delivered; the evaluator the agent may use is the task's
public Checker, already inside the container.

A run started with `--seed-workspace <earlier run>` (`seed()`) is a fresh start in
every other respect -- new container, delivery, start prompt, CLI session and
budget -- with the earlier run's own `work/` files copied in first: the same
lineage only (one agent continuing its own work, never another's), and never
what the harness delivers, caches, the CLI's session transcripts or credentials,
or weights; of the CLI's home only its memory notes.
"""
from __future__ import annotations

import json
import shlex
import shutil
from pathlib import Path

from . import backend, instructions
from .boundary import WEIGHT_SUFFIXES
from .utils import write_json_atomically

VERSIONS_DIR = ".versions"

#: The paper-search skill: its source, and where Claude Code finds a user skill.
PAPERS_SKILL = Path(__file__).resolve().parents[2] / "environments" / "agent-base" / "papers" / "skill"
SKILLS_DIR = Path(".claude") / "skills"
PAPERS_SKILL_NAME = "compression-papers"

#: Writes a version request and waits for the host's acknowledgement.
VERSION_TOOL = r'''#!/usr/bin/env python3
"""Record a milestone of /app/submission in this run's lineage.

    /app/work/version "what changed and why"

The harness records the current state of /app/submission as a numbered
version: a manifest of every file with its size and SHA-256, plus a copy of
the code and configuration (weights are hashed, not copied). It does not
evaluate anything, submit anything early, or end the run; the hidden Verifier
evaluates /app/submission as it is at the deadline. When /app/submission
computes with exactly the files of the latest version (interpreter caches and
tokenizer files aside), nothing is recorded and the answer is "unchanged".
"""
import json, os, secrets, sys, time
from datetime import datetime, timezone

note = " ".join(sys.argv[1:]).strip()
if not note or note in ("-h", "--help"):
    print(__doc__.strip())
    raise SystemExit(0 if note else 2)
root = "/app/work/.versions"
os.makedirs(root, exist_ok=True)
name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3)
temporary = os.path.join(root, "." + name + ".tmp")
with open(temporary, "w") as handle:
    json.dump({"action": "publish", "note": note[:4000],
               "requested_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}, handle)
os.replace(temporary, os.path.join(root, name + ".json"))
ack = os.path.join(root, name + ".ack.json")
for _ in range(120):
    if os.path.exists(ack):
        with open(ack) as handle:
            print(handle.read().strip())
        raise SystemExit(0)
    time.sleep(0.5)
print(json.dumps({"status": "requested", "request": name,
                  "note": "the harness has not acknowledged yet; it will capture the request"}))
'''

#: Prints the time left on this run's clock.
CLOCK_TOOL = r'''#!/usr/bin/env python3
"""Print the time left before this run's deadline."""
import json
from datetime import datetime, timezone

with open("/app/work/WORKSPACE.json") as handle:
    workspace = json.load(handle)
deadline = datetime.strptime(workspace["deadline_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
left = max(0, int((deadline - datetime.now(timezone.utc)).total_seconds()))
print(json.dumps({"now_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "deadline_utc": workspace["deadline_utc"],
                  "seconds_left": left,
                  "time_left": f"{left // 3600}h{left % 3600 // 60:02d}m{left % 60:02d}s"}))
'''


def layout(run_root: Path | str, *, debug_model: bool = False) -> dict[str, Path]:
    """Create the run's directories and the lock file; return their paths."""
    root = Path(run_root)
    paths = {
        "lock": root / ".submission.lock",
        "work": root / "work",
        "agent_home": root / "work" / ".agent-home",
        "tmp": root / "work" / ".tmp",
        "versions": root / "work" / VERSIONS_DIR,
        "submission": root / "submission",
        "verifier": root / "verifier",
        "public_checks": root / "public-checks",
    }
    for key in ("work", "agent_home", "tmp", "versions", "submission", "verifier",
                "public_checks"):
        paths[key].mkdir(parents=True, exist_ok=True)
    paths["lock"].touch(exist_ok=True)
    if debug_model:
        paths["debug_submission"] = root / "work" / "debug-submission"
        paths["debug_submission"].mkdir(parents=True, exist_ok=True)
    return paths


#: Never taken from an earlier run's work/ by `seed()`.
SEED_SKIP = frozenset({"AGENT_TASK.md", "WORKSPACE.json", "clock", "version", VERSIONS_DIR, ".tmp",
                       "__pycache__", ".cache", ".credentials.json", "auth.json"})
SEED_MAX_BYTES = 2**30


def _seed_keeps(relative: Path, size: int) -> bool:
    parts = relative.parts
    if any(part in SEED_SKIP for part in parts) or relative.suffix.lower() in WEIGHT_SUFFIXES:
        return False
    if parts and parts[0] == ".agent-home":
        # Of the CLI's home, only its memory notes: a fresh start runs a new session.
        return ("memory" in parts[:5] and parts[1:3] == (".claude", "projects")) or \
               ("memories" in parts and parts[1] == ".codex")
    return size <= SEED_MAX_BYTES


def seed(source_run: Path | str, run_root: Path | str) -> dict:
    """Copy an earlier run's own work/ files into this run's freshly delivered work/."""
    source, target = Path(source_run) / "work", Path(run_root) / "work"
    copied = skipped = total = 0
    for path in sorted(source.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(source)
        size = path.stat().st_size
        destination = target / relative
        if not _seed_keeps(relative, size) or destination.exists():
            skipped += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
        destination.chmod(path.stat().st_mode & 0o777)
        copied += 1
        total += size
    return {"from": str(Path(source_run)), "files": copied, "bytes": total, "skipped": skipped}


def deliver(run_root: Path | str, *, task_dir: Path | str, task_id: str,
            deadline_utc: str, budget_hours: float, run_id: str,
            lineage_id: str) -> dict:
    """Write the brief, WORKSPACE.json, the tools and the paper skill into `work/`."""
    work = Path(run_root) / "work"
    brief = instructions.brief(task_dir)
    (work / instructions.BRIEF_NAME).write_text(brief, encoding="utf-8")
    workspace = {"kind": "compression_bench_workspace", "schema_version": 1,
                 "task_id": task_id, "run_id": run_id, "lineage_id": lineage_id,
                 "deadline_utc": deadline_utc, "budget_hours": budget_hours,
                 "submission": "/app/submission", "work": "/app/work"}
    write_json_atomically(work / "WORKSPACE.json", workspace)
    for name, body in (("version", VERSION_TOOL), ("clock", CLOCK_TOOL)):
        tool = work / name
        tool.write_text(body, encoding="utf-8")
        tool.chmod(0o755)
    return {"brief_chars": len(brief), "tools": ["version", "clock"], "skills": [deliver_skill(work / ".agent-home")]}


def deliver_skill(agent_home: Path | str, source: Path = PAPERS_SKILL) -> str:
    """Copy the paper-search skill into the CLI's home; return its name."""
    target = Path(agent_home) / SKILLS_DIR / PAPERS_SKILL_NAME
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return PAPERS_SKILL_NAME


# --- the boot proof -----------------------------------------------------------------

EGRESS_PROBE = r'''
import json, sys, urllib.error, urllib.request
out = {}
for url in sys.argv[1:]:
    try:
        urllib.request.urlopen(url, timeout=15)
        out[url] = "reachable"
    except urllib.error.HTTPError as exc:
        out[url] = "reachable (HTTP %d)" % exc.code
    except Exception as exc:
        out[url] = "blocked (%s)" % type(exc).__name__
print(json.dumps(out))
'''

BLOCKED_PROBES = ("https://pypi.org/simple/", "https://github.com/",
                  "https://huggingface.co/")


def boot_contract(container: str, *, agent_cli: str, provider_url: str,
                  run=backend.run) -> list[dict]:
    """In-container checks that the environment is the one the task states.

    Every check is recorded; the caller fails the run on any failure, because
    an agent started in a broken environment measures the environment.
    """
    checks: list[dict] = []

    def check(name: str, command: str, verdict, timeout: float = 120.0):
        try:
            got = run(container, command, timeout=timeout)
            ok, detail = verdict(got)
        except Exception as exc:                         # noqa: BLE001
            ok, detail = False, f"{type(exc).__name__}: {exc}"[:400]
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    check("model readable", "test -r /input/model/config.json && echo ok",
          lambda g: (g.returncode == 0, g.stdout.strip() or g.stderr[-200:]))
    check("one GPU visible", "nvidia-smi -L",
          lambda g: (g.returncode == 0 and len(g.stdout.strip().splitlines()) == 1,
                     g.stdout.strip()[-300:] or g.stderr[-300:]))
    check("agent CLI runs", f"{agent_cli} --version",
          lambda g: (g.returncode == 0, (g.stdout or g.stderr).strip()[-200:]))
    check("agent home writable",
          "touch /app/work/.agent-home/.compression-bench-probe && rm /app/work/.agent-home/.compression-bench-probe && echo ok",
          lambda g: (g.returncode == 0, g.stdout.strip() or g.stderr[-200:]))
    check("submission writable",
          "touch /app/submission/.compression-bench-probe && rm /app/submission/.compression-bench-probe && echo ok",
          lambda g: (g.returncode == 0, g.stdout.strip() or g.stderr[-200:]))
    skill_dir = f"{backend.AGENT_HOME}/{SKILLS_DIR.as_posix()}/{PAPERS_SKILL_NAME}"
    check("paper collection searchable",
          f"test -r {skill_dir}/SKILL.md && python3 /opt/compression-papers/skill/scripts/search.py --categories | tail -1",
          lambda g: (g.returncode == 0 and g.stdout.strip().endswith("papers"),
                     g.stdout.strip()[-200:] or g.stderr[-200:]))

    def checker_answers(got):
        try:
            reply = json.loads(got.stdout)
        except json.JSONDecodeError:
            return False, (got.stdout + got.stderr)[-400:]
        error = str(reply.get("error", ""))
        # An empty submission must be refused BY THE CHECKER: that proves the
        # socket, the service and the evaluator all answer.
        return (reply.get("valid") is False and "model" in error,
                error[:300])
    check("public Checker answers",
          "python3 /app/check_submission.py /app/submission --package-only",
          checker_answers, timeout=300.0)

    urls = (provider_url, *BLOCKED_PROBES)

    def egress(got):
        try:
            seen = json.loads(got.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            return False, (got.stdout + got.stderr)[-400:]
        ok = (seen.get(provider_url, "").startswith("reachable")
              and all(seen.get(url, "").startswith("blocked")
                      for url in BLOCKED_PROBES))
        return ok, json.dumps(seen)
    check("egress allowlist",
          f"python3 -c {shlex.quote(EGRESS_PROBE)} "
          + " ".join(shlex.quote(u) for u in urls),
          egress, timeout=120.0)
    return checks


def summary_of(checks: list[dict]) -> tuple[bool, str]:
    failed = [c for c in checks if not c["ok"]]
    if not failed:
        return True, f"{len(checks)} boot checks passed"
    return False, "; ".join(f"{c['check']}: {c['detail']}" for c in failed)
