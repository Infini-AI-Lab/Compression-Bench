# compression_bench — the compression-bench harness

An **autonomous model-compression harness**, built on the framework of
MLE-Bench's ServerBench: the same task tooling, run lifecycle, container
composition through Harbor, and host-side verdict. A coding agent gets one
frozen compression task — a pinned BF16 checkpoint, a capacity limit, an offline
container with the inference runtime, training data and a papers skill, a public Checker and a wall clock — and compresses
the model into a submission (weights plus `inference.py`). When the wall clock
runs out, the harness stops the environment and runs the hidden Verifier once,
host-side, in its own no-network container: package and GPU-resident capacity,
the submission's own compression, a smoke generation, perplexity on hidden
BF16-continued text, the hidden correctness gate, and single-request decode
latency (reported, not scored). The reward is fidelity to the task's own BF16
checkpoint, `10 × PPL_bf16 / PPL`, behind those gates.

The tasks are in [`compression_tasks/`](../compression_tasks/README.md).
[`OPERATIONS.md`](OPERATIONS.md) is the runbook: setting up a host, running a
task, reading the results. [`docs/INTERFACES.md`](docs/INTERFACES.md) fixes the
seams between the task tooling and the run lifecycle.

## The one boundary that matters

```
                HARNESS                    │              CANDIDATE
  ─────────────────────────────────────────┼──────────────────────────────────
  task definition, inputs and baselines    │  compression method and recipe
  agent execution and its container        │  any training on the supplied data
  network allowlist, single-copy lock      │  weight format and packing
  public Checker and hidden Verifier       │  inference.py and its kernels
  versions and trajectory logging          │  what to keep in /app/submission
```

The harness ships the inference runtime (vLLM as released), training data
(`/opt/datasets`) and, for `standard` tasks, papers (`/opt/compression-papers`,
searched and read through the `compression-papers` skill). It ships no
compression toolkit, no third-party source code and no ready-made quantizer,
never a recipe or a guide; no task's starter or tools contain a method.

## The loop

```
  create run ──► claim GPU ──► build workspace ──► start environment (Harbor)
                                                          │   main + public-checker,
                                                          │   egress allowlist sidecar
                                                    boot proof
                                                          │
                     the agent compresses, checks itself with the public
                     Checker, and records versions with ./version "why"
                                                          │
           deadline ──► final version ──► stop the environment (lock free)
                                                          │
                                                          ▼
         hidden Verifier, once, host-side, in its own no-network container:
         check → smoke → ppl → accuracy → decode → score  ──►  FINAL.json
                                                          │
                         every ./version checkpoint verdicted, progress row,
                         METRICS.json written, measured weights deleted
```

An agent session that ends with time left is resumed into the same workspace
until the deadline. The verdict is never fed back to the agent.

## Layout

```
harness/final_validate/   the evaluator: ONE source, byte-copied into every task's
                          tests/ (Verifier) and environment/public-checker/ (Checker);
                          task_policy.py reads the per-task evaluator-assets/task_policy.json;
                          tools/validate_evolve.py runs the hidden verdict for a run,
                          tools/verdict_checkpoints.py the verdict of every checkpoint
harness/lifecycle/        the evolve driver (run.py) and its libraries
scripts/                  evolve (the one entrypoint), compress_evolve/evolve-task.sh (launcher),
                          env.sh, deploy/ (host setup, images, task inputs and BF16 baselines),
                          agent/ (harbor_env.py, image_freshness.py, the egress config,
                          provider route templates), scheduler.py and the run-maintenance
                          tools beside it, baselines/ (GPTQ and released quantizations),
                          studies/ (ablations and supplementary studies), and the run
                          analysis scripts (ANALYSIS.md)
environments/             agent-base (vLLM 0.32.0 + /opt/compression-papers and its skill + /opt/datasets),
                          platform-gguf (+ the pinned vllm-gguf-plugin; every task runs on it)
                          and its pinned patches,
                          images.json (the tags every task renders FROM),
                          reference-lock.json (where each library paper is published)
tests/                    the CPU suite
docs/                     INTERFACES.md: the seams between the task tooling and the lifecycle
evolve_space/             runtime, gitignored: every run's workspace, journal and verdict
```

## Requirements and tests

Two interpreters, resolved by `scripts/env.sh` (bind them for a host in the
gitignored `scripts/env.local.sh`): `COMPRESSION_BENCH_PY` runs the harness,
`taskctl` and the tests (stdlib + pyyaml + pytest; no torch),
`COMPRESSION_BENCH_HARBOR_PY` has Harbor 0.21 and runs only
`scripts/agent/harbor_env.py`.

```bash
. compression_bench/scripts/env.sh
"$COMPRESSION_BENCH_PY" -m pytest compression_bench/tests -q        # CPU only: no GPU, model or network
bash compression_bench/scripts/correctness-gate.sh                  # taskctl check + imports + suite
```
