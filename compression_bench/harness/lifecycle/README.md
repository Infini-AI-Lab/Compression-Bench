# `compression_bench/harness/lifecycle/` — the evolve driver

`compression_bench/scripts/evolve` hands its arguments to `run.py`, which takes one run
of one task from an empty workspace to a recorded verdict. Every stage is one
module; the core is stdlib-only (yaml is imported inside `utils.read_contract`,
Harbor only inside `scripts/agent/harbor_env.py`, which runs under
`COMPRESSION_BENCH_HARBOR_PY`).

| stage | module | what it does |
|---|---|---|
| identity, journal, states | `record.py` | run id `<task hash>-<UTC stamp>-<hex6>` claimed by `mkdir`; `identity.json`; append-only fsynced `journal.jsonl`; the run state machine `CREATED → WORKSPACE_READY → AGENT_RUNNING → VERDICT → COMPLETED / FAILED / CANCELLED` (why the agent phase ended lives only in `agent_phase_ended`, the verdict only in `FINAL.json`) |
| GPU | `gpus.py` | a card is free when sampled readings show no compute process and little memory, and no other run's `flock` claim; container reservations are reported, not decisive |
| disk | `disk.py` | a run refuses to start below the free-space floors on the Docker root and the run root; every driver write waits on a full disk (`utils.until_space`) instead of failing |
| workspace | `workspace.py` | the run directory (`INTERFACES.md`), `work/AGENT_TASK.md`, `WORKSPACE.json`, the `version` and `clock` tools, the boot proof |
| credentials | `credentials.py` | the CLI login staged into `<run>/secrets/`, bound read-only, refreshed in place; or, for a provider route (`--agent-env`), the route's env file instead |
| container | `containers.py`, `backend.py` + `scripts/agent/harbor_env.py` | Harbor composes `environment/` (main + public Checker) behind its egress allowlist sidecar, plus the per-run overlay (CLI binary, credentials, gost timeout, HF-cache blob mounts); after that the driver talks to `main` by `docker exec` (`backend.py`) |
| agent | `agent.py`, `instructions.py` | the CLI launched with the task's specification (its `instruction.md` up to `## Continuation`), continued (Claude Code `--resume`, `codex exec resume`) until the deadline with the task's own session sections (no idle cap); idle and cold restarts; terminal, session-limit and credit detection (a credit hold pauses the container and restores the agent's remaining time). The harness composes no agent-facing prose |
| versions | `boundary.py` | `./version "why"` → manifest (full SHA-256), composition, the public check of exactly those files, and code copies in `compression_output/<lineage>/<lineage>_vN/`; a reflink checkpoint in the gitignored `.objects/<lineage>/vN/`, kept until its verdict is recorded |
| trajectory | `trajectory.py`, `telemetry.py` | the agent's steps, transcripts and notes into `trajectory/`; its memory notes, `resources.jsonl` (sampled while it runs) and `TIMING.json` |
| harvest | `harvest.py`, `verdicts.py` | after the run: every metric (each verdict in detail) into `METRICS.json` (run + lineage), read back, then every weight a verdict measured is deleted (checkpoint verdicts: `final_validate/tools/verdict_checkpoints.py`) |
| summary | `report.py`, `pricing.py` | `SUMMARY.json`, usage and cost from the CLI's stream (closed models at their CLI's price, open-weight models at OpenRouter's, `openrouter_prices.json`) |
| verdict | `final_validate/tools/validate_evolve.py` | the hidden Verifier once, in its own no-network container, after the agent environment is down and the lock is free; `FINAL.json` |
| record | `evaluation_progress/record_run.py` | one CSV row per production run |

```bash
. compression_bench/scripts/env.sh
compression_bench/scripts/compress_evolve/evolve-task.sh qwen3_8_27b_12g_standard_1xH200NVL          # GPU=auto, contract hours
HOURS=1 GPU=7 compression_bench/scripts/compress_evolve/evolve-task.sh 8a661afb --dry-run  # build + boot only
"$COMPRESSION_BENCH_PY" -m pytest compression_bench/tests/lifecycle -q                             # CPU only
```

A run lives at `compression_bench/evolve_space/<evolve|evolve-scratch>/<task>/<run-id>/`;
its files are listed in [`docs/INTERFACES.md`](../../docs/INTERFACES.md). How a
run proceeds, what it costs in GPU time after the deadline and how it is read
afterwards is in [`OPERATIONS.md`](../../OPERATIONS.md). An exit code is not a
verdict: `final_validation` in the journal and `FINAL.json` are.
