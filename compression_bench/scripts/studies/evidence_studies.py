#!/usr/bin/env python3
"""Repeatability trials: rerun recorded (agent, task) runs from an empty workspace and compare the outcomes.

Source compression_bench/scripts/env.sh, then:

    evidence_studies.py prepare --config TRIALS.json   # freeze the plan (plan.json); refuses to overwrite
    evidence_studies.py run                            # launch every queued trial once, on idle cards
    evidence_studies.py collect                        # original/repeat results (repeat_results.json/.csv)

TRIALS.json defines the study:

    {"harness_commit": "<commit the isolated checkout is pinned to>",
     "cli_version": "<agent CLI version>", "cli_sha256": "<SHA-256 of WORK/bin/claude>",
     "models": {"<agent>": "<provider model id>"},
     "env_files": {"<agent>": "<provider env file>"},        # default ~/.config/compression_bench/<model basename>.env
     "provider_base_url": "<ANTHROPIC_BASE_URL the env files must name>",
     "hours": 8, "effort": "max",
     "trials": [{"agent": "<agent>", "task": "<task id without _standard_1xH200NVL>", "original_run": "<run id>"}],
     "model_dirs": {"<task id>": "<BF16 checkpoint>"},       # optional; default the original run's, else model_for_task.py
     "protect": ["<repo-relative file>", ...]}               # optional; hash-guarded while the trials run

Each trial runs in an isolated checkout of the harness (WORK/checkout, at harness_commit) with the agent CLI binary
WORK/bin/claude, in the checkout's scratch namespace. Work files under WORK (compression_bench/evolve_space/studies/
evidence), records under OUT (evaluation_progress/supplemental_studies/evidence). Never retries an allocated trial;
failed trials remain outcomes.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from common import REPO, file_sha256, model_dir as task_model_dir, read_json, run_dir, utc_now, write_json

WORK = REPO / "compression_bench/evolve_space/studies/evidence"
OUT = REPO / "evaluation_progress/supplemental_studies/evidence"
CHECKOUT = WORK / "checkout"
SCRATCH_RUNS = CHECKOUT / "compression_bench/evolve_space/evolve-scratch"
TASK_SUFFIX = "_standard_1xH200NVL"


def plan():
    return read_json(OUT / "plan.json")


def originals():
    """(agent, task, original run id) of every trial in the frozen plan."""
    return [(j["agent"], j["task"].removesuffix(TASK_SUFFIX), j["original_run"]) for j in plan()["jobs"]]


def env_file(cfg, agent):
    named = cfg.get("env_files", {}).get(agent)
    if named:
        return Path(named).expanduser()
    return Path.home() / ".config/compression_bench" / f"{cfg['models'][agent].rsplit('/', 1)[1]}.env"


def protected(cfg_or_plan=None):
    """SHA-256 of every hash-guarded file: the configured ones and each original run's identity and metrics."""
    cfg = cfg_or_plan or plan()["config"]
    paths = [REPO / p for p in cfg.get("protect", [])]
    for t in cfg["trials"]:
        paths.extend(run_dir(t["original_run"]) / n for n in ("identity.json", "METRICS.json"))
    return {str(p.relative_to(REPO)): file_sha256(p) for p in paths if p.is_file()}


def provider(cfg, agent):
    # Exact allowlist: never copy API keys, auth tokens, or arbitrary custom headers into evidence.
    allowed = {"ANTHROPIC_BASE_URL", "ANTHROPIC_MODEL", "ANTHROPIC_SMALL_FAST_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL",
               "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL",
               "CLAUDE_CODE_SUBAGENT_MODEL", "CLAUDE_CODE_MAX_CONTEXT_TOKENS",
               "MAX_THINKING_TOKENS", "CLAUDE_CODE_EFFORT_LEVEL", "API_TIMEOUT_MS",
               "DISABLE_PROMPT_CACHING", "DISABLE_AUTOUPDATER"}
    values = {}
    for line in env_file(cfg, agent).read_text().splitlines():
        k, sep, v = line.strip().removeprefix("export ").partition("=")
        if sep and k in allowed:
            values[k] = v.strip().strip("'\"")
    if cfg.get("provider_base_url"):
        assert values["ANTHROPIC_BASE_URL"] == cfg["provider_base_url"]
    assert values["ANTHROPIC_MODEL"] == cfg["models"][agent]
    return {"settings": values, "credential_values_recorded": False,
            "sampling": "provider defaults through the unchanged agent CLI; not overridden by the study"}


def model_dir(cfg, task_id, recorded):
    if task_id in cfg.get("model_dirs", {}):
        return cfg["model_dirs"][task_id]
    if recorded and Path(recorded).is_dir():
        return recorded
    return task_model_dir(task_id)


def prepare(config):
    if (OUT / "plan.json").exists():
        raise SystemExit("plan already exists; do not overwrite the pre-execution record")
    cfg = read_json(config)
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CHECKOUT, text=True).strip() == cfg["harness_commit"]
    assert file_sha256(WORK / "bin/claude") == cfg["cli_sha256"]
    jobs = []
    for t in cfg["trials"]:
        agent, task, run = t["agent"], t["task"], t["original_run"]
        p = run_dir(run)
        task_id = task + TASK_SUFFIX
        root = CHECKOUT / "compression_tasks" / task_id
        current = REPO / "compression_tasks" / task_id
        # Every tracked task input (including private evaluation assets) is unchanged.
        files = subprocess.check_output(["git", "ls-files", str(current.relative_to(REPO))], cwd=REPO, text=True).splitlines()
        assert all(file_sha256(REPO / f) == file_sha256(CHECKOUT / f) for f in files), task
        prompt = (root / "instruction.md").read_text().partition("## Continuation")[0].rstrip() + "\n"
        assert prompt == (p / "prompts/0.start.txt").read_text()
        created = next(json.loads(l) for l in (p / "journal.jsonl").read_text().splitlines()
                       if json.loads(l)["event"] == "run_created")
        jobs.append({"id": f"{agent}-{task}", "agent": agent, "task": task_id, "original_run": run,
                     "model_dir": model_dir(cfg, task_id, created.get("model_dir")),
                     "contract_sha256": file_sha256(root / "contract.yaml"), "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                     "provider": provider(cfg, agent), "hours": cfg.get("hours", 8), "seed_workspace": None,
                     "status": "queued", "lineage_id": f"repeat-{agent}-{task}"})
    out = {"created_utc": utc_now(), "study": "repeatability trials of recorded runs",
           "config": cfg, "harness_commit": cfg["harness_commit"], "cli_version": cfg.get("cli_version"),
           "cli_sha256": cfg["cli_sha256"], "exact_trials": len(jobs), "reasoning_effort": cfg.get("effort", "max"),
           "uncontended_gpu": "lifecycle flock plus three idle samples",
           "namespace": "isolated checkout/evolve-scratch; no progress-CSV writes",
           "jobs": jobs, "protected_sha256": protected(cfg)}
    write_json(OUT / "plan.json", out)
    write_json(OUT / "repeat_state.json", {"jobs": jobs, "created_utc": utc_now()})
    collect()


def result(p, agent, task, kind):
    """One run's final verification, agent usage and cost, from its METRICS.json."""
    m = read_json(p / "METRICS.json")
    final = m["verdicts"]["final"]
    detail = final.get("detail") or {}
    acc = detail.get("accuracy") or {}
    by = acc.get("benchmarks") or {}
    latency = m["verdict"].get("decode_latency_seconds")
    usage = m["agent"].get("usage") or {}
    return {"kind": kind, "agent": agent, "task": task, "run_id": p.name,
            "status": final.get("status"), "passed": final.get("status") == "SCORED",
            "verification_ppl": (detail.get("ppl") or {}).get("ppl"),
            "code_correct": (by.get("code") or {}).get("correct"),
            "knowledge_correct": (by.get("knowledge") or {}).get("correct"),
            "math_correct": (by.get("math") or {}).get("correct"),
            "mean_accuracy": acc.get("accuracy"), "decode_seconds": latency,
            "tps": 127 / latency if latency else None,
            "api_cost_usd": usage.get("cost_usd"), "api_cost_basis": usage.get("cost_basis"),
            "cli_reported_cost_usd": usage.get("total_cost_usd"),
            "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
            "source": str(p / "METRICS.json")}


def collect():
    """repeat_results.json/.csv: every original run's outcome and each finished repeat's."""
    rows = [result(run_dir(run), a, t, "original") for a, t, run in originals()]
    state = read_json(OUT / "repeat_state.json")
    for job in state["jobs"]:
        matches = list(SCRATCH_RUNS.glob(f"*/{job['run_id']}")) if job.get("run_id") else []
        if matches and (matches[0] / "METRICS.json").exists():
            rows.append(result(matches[0], job["agent"], job["task"].removesuffix(TASK_SUFFIX), "repeat"))
    comparisons = []
    for old in [r for r in rows if r["kind"] == "original"]:
        new = next((r for r in rows if r["kind"] == "repeat" and (r["agent"], r["task"]) == (old["agent"], old["task"])), None)
        if new:
            comparisons.append({"task": old["task"], "agent": old["agent"], "ppl_delta": new["verification_ppl"] - old["verification_ppl"] if new["verification_ppl"] else None,
                                "accuracy_delta": new["mean_accuracy"] - old["mean_accuracy"] if new["mean_accuracy"] is not None else None,
                                "pass_agreement": new["passed"] == old["passed"]})
    write_json(OUT / "repeat_results.json", {"collected_utc": utc_now(), "rows": rows, "comparisons": comparisons,
          "cost_caveat": "cost_usd is the harness's recorded price estimate and cli_reported_cost_usd the CLI's own; neither is an invoice",
          "repeat_jobs": [{k: j.get(k) for k in ("id", "status", "run_id", "gpu", "returncode")} for j in state["jobs"]]})
    with (OUT / "repeat_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def run():
    sys.path.insert(0, str(CHECKOUT / "compression_bench/harness"))
    from lifecycle import gpus
    cfg = plan()["config"]
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / "controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read_json(OUT / "repeat_state.json")
        # No implicit resumption/relaunch on controller restart: that could exceed the planned trials.
        if any(j["status"] != "queued" for j in state["jobs"]):
            raise SystemExit("controller already launched trials; inspect state rather than rerunning")
        active = {}
        while any(j["status"] in ("queued", "running") for j in state["jobs"]):
            for jid, (proc, job, log) in list(active.items()):
                if proc.poll() is not None:
                    log.close()
                    job.update(status="finished" if proc.returncode == 0 else "execution_failed", returncode=proc.returncode, finished_utc=utc_now())
                    del active[jid]
            busy, cards = gpus.busy_cards(samples=3)
            held = gpus.held_by_claims(cards)
            free = [i for i in cards if i not in busy | held | gpus.excluded() | gpus.auto_excluded()]
            for job, gpu in zip([j for j in state["jobs"] if j["status"] == "queued"], free):
                env = dict(os.environ)
                env["PATH"] = str(WORK / "bin") + os.pathsep + env["PATH"]
                env["PYTHONPATH"] = str(CHECKOUT / "compression_bench/harness")
                argv = [env["COMPRESSION_BENCH_PY"], str(CHECKOUT / "compression_bench/scripts/evolve"), "--task", str(CHECKOUT / "compression_tasks" / job["task"]),
                        "--gpu", str(gpu), "--model-dir", job["model_dir"], "--hours", str(job["hours"]), "--agent-cli", "claude",
                        "--agent-model", cfg["models"][job["agent"]], "--agent-effort", cfg.get("effort", "max"),
                        "--agent-env", str(env_file(cfg, job["agent"])),
                        "--scratch", "--keep-weights", "--lineage-root", str(WORK / "lineages"), "--lineage-id", job["lineage_id"]]
                log = (WORK / f"{job['id']}.log").open("w")
                proc = subprocess.Popen(argv, cwd=CHECKOUT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                job.update(status="running", pid=proc.pid, gpu=gpu, started_utc=utc_now(), argv=argv)
                active[job["id"]] = (proc, job, log)
            for job in state["jobs"]:
                if job["status"] == "running" and not job.get("run_id"):
                    for ident in SCRATCH_RUNS.glob("*/*/identity.json"):
                        d = read_json(ident)
                        if d["lineage_id"] == job["lineage_id"]:
                            job["run_id"] = d["run_id"]
            state["updated_utc"] = utc_now()
            write_json(OUT / "repeat_state.json", state)
            collect()
            assert protected() == plan()["protected_sha256"], "a hash-guarded file changed"
            if active or any(j["status"] == "queued" for j in state["jobs"]):
                time.sleep(30)
        write_json(OUT / "repeat_complete.json", {"completed_utc": utc_now(), "trials": len(state["jobs"]), "protected_files_unchanged": True})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("prepare", "run", "collect"))
    p.add_argument("--config", help="trial definition (prepare only)")
    a = p.parse_args()
    if a.command == "prepare":
        if not a.config:
            p.error("prepare needs --config")
        prepare(a.config)
    else:
        {"run": run, "collect": collect}[a.command]()
