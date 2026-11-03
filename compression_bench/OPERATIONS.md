# Operations — running compression-bench

The runbook. Every command runs from the repository root. Paths, device indices
and interpreters are derived on the host, never copied from here.

## 1. A host from nothing

Requirements: Linux with NVIDIA GPUs (one H200-class card per run), Docker with
the NVIDIA container runtime, network access during setup, `pdftotext`
(poppler-utils), and the agent CLI (`claude` or `codex`) installed on the host
and logged in (a provider route, §2, needs the `claude` binary but no login).

### Interpreters

Two interpreters, neither of them a bare `python3`. `COMPRESSION_BENCH_PY` runs
the harness, `taskctl` and the tests (stdlib + pyyaml + pytest; the harness core
carries no torch). `COMPRESSION_BENCH_HARBOR_PY` has Harbor 0.21 and runs only
`compression_bench/scripts/agent/harbor_env.py`. `compression_bench/scripts/env.sh`
resolves both (the gitignored `compression_bench/scripts/env.local.sh`, whose
assignments win, then the environment, then `compression_bench/.venv`) and
refuses when either is missing or `COMPRESSION_BENCH_HARBOR_PY` lacks exactly
`harbor==0.21.0`.

```bash
# Either one venv for both:
uv venv compression_bench/.venv && uv pip install --python compression_bench/.venv/bin/python \
    "harbor==0.21.0" pyyaml pytest
# or bind existing interpreters for this host in compression_bench/scripts/env.local.sh:
cat > compression_bench/scripts/env.local.sh <<'EOF'
COMPRESSION_BENCH_PY=/path/to/python3                       # pyyaml, pytest
COMPRESSION_BENCH_HARBOR_PY=/path/to/harbor-venv/bin/python # harbor==0.21.0
export COMPRESSION_BENCH_MODELS_ROOT=/path/to/models        # optional: <local_name>-bf16 checkpoints
export COMPRESSION_BENCH_AUTO_EXCLUDED_GPUS="0"             # optional: cards never picked automatically
export COMPRESSION_BENCH_EXCLUDED_GPUS=""                   # optional: cards never used
EOF
. compression_bench/scripts/env.sh
```

The deploy tools that download, filter or build data (`fetch_model.py`,
`prepare_datasets.py`, `benchmark_guard.py`, `build_*_assets.py`,
`create_model_task.py`) need a third, ordinary interpreter with `pyarrow`,
`numpy`, `huggingface_hub`, `zstandard` and `transformers`, written `$DEPLOY_PY`
below; `01-build-images.sh` reads it from `DEPLOY_PY` (default `python3`).

### Pinned inputs that are not stored in git

```bash
export HF_TOKEN=...      # gated checkpoints and tokenizers (or: huggingface-cli login)

# The official tokenizer files (compression_tasks/README.md, "Fetching the tokenizers").
"$COMPRESSION_BENCH_PY" compression_bench/scripts/deploy/fetch_tokenizers.py
"$COMPRESSION_BENCH_PY" compression_tasks/_manage/taskctl.py check

# The paper library: every paper listed with its publisher's URL in
# compression_bench/environments/reference-lock.json and agent-base/papers/supplement.json
# (arXiv at fixed versions), downloaded from there; text extracts are made with pdftotext.
"$COMPRESSION_BENCH_PY" compression_bench/scripts/deploy/prepare_references.py   # --verify checks only

# The training data for /opt/datasets.
"$DEPLOY_PY" compression_bench/scripts/deploy/prepare_datasets.py                # --verify checks only
```

`prepare_datasets.py` downloads pinned dataset revisions and drops every row
that holds an item of the hidden correctness gate's benchmarks (any split;
`benchmark_guard.py`), every training row sharing a word 13-gram with a
document of the perplexity blocks, and every FineWeb document sharing a URL or a
word 13-gram with the C4 validation split; `DATASETS.json` records what was kept
and dropped. The perplexity documents are not distributed: the rows they rule
out are listed by row index in
`compression_bench/scripts/deploy/perplexity_overlap.json`, which
`prepare_datasets.py --pin-perplexity-overlap` writes (§6).

`01-build-images.sh` runs `prepare_references.py`, `prepare_datasets.py` and
`benchmark_guard.py check` itself before it hashes the agent image's inputs, so
a fresh host can skip straight to the build.

### Checkpoints

A run mounts the task's pinned BF16 checkpoint read-only at `/input/model`.
`compression_bench/scripts/model_for_task.py <task dir>` finds it:
`$COMPRESSION_BENCH_MODELS_ROOT/<local_name>-bf16` (or `<local_name>`), else the
Hugging Face cache snapshot at the pinned revision; a directory counts only if
it holds `config.json` and every weight shard its index names. `MODEL_DIR=...`
overrides. The run refuses (`MODEL_MISMATCH`) a directory without `config.json`
or weight files and a cache snapshot of another revision. To download one into
the cache:

```bash
"$DEPLOY_PY" compression_bench/scripts/deploy/fetch_model.py compression_tasks/<task>
```

Tasks with a `debug_model` (Qwen3.5-9B) mount it at `/input/debug-model` when
launched with `DEBUG_MODEL=1`.

### Images

```bash
bash compression_bench/scripts/deploy/01-build-images.sh          # agent-base, platform-gguf, pull vllm-runtime
bash compression_bench/scripts/deploy/00-preflight.sh qwen3_8_27b_12g_standard_1xH200NVL   # read-only checks
```

`01-build-images.sh` pulls `vllm/vllm-openai:v0.32.0` and builds
`compression_bench/agent-base:v1` (vLLM 0.32.0, the papers in
`/opt/compression-papers` with the `compression-papers` skill, `/opt/datasets`;
no toolkit, author source or llama.cpp) and `compression_bench/platform-gguf:v1`
(agent base + the pinned vllm-gguf-plugin; every task runs on it). Each built
image is labelled with the digest of its build context. A run refuses an image
whose label differs from the checkout's inputs; after editing anything under
`compression_bench/environments/` (or restoring other papers or datasets),
rebuild.

### The runtime: vLLM 0.32.0 as released

The agent, the Checker and the Verifier run the same stack (the Checker and
Verifier images are built FROM the agent image and assert the versions):
`vllm/vllm-openai:v0.32.0` with the humming-kernels 0.1.17 it pins, nothing
patched. `pip check` reports only the release's own pins (torch's NCCL,
OpenTelemetry's SDK). The GGUF platform adds the pinned vllm-gguf-plugin
(`e2b8ad53`) with one compatibility edit applied at its build
(`platform-gguf/compat_vllm_0_32.py`: its two Qwen3.5 config imports point at
Transformers' classes); the per-task platform patches
(`platform-gguf/patches/`) are pinned repairs of the plugin, each refusing a
file it was not written for.

A submission registers its own weight format with vLLM
(`register_quantization_config`), its kernels shipped in the package. vLLM's
built-in formats, the GGUF loader, Humming and every other package's formats
stay in the runtime but serve no weight: the package stage refuses code that
imports or names one (an import, a `quantization=` name, a
`VLLM_HUMMING_*_QUANT_CONFIG` switch, a built-in `quant_method`), and the census
refuses a loaded layer whose method, or a class it inherits from, comes from
one (`final_validate/own_compression.py`, `model_memory.foreign_quant_methods`,
`compression_bench/tests/test_own_compression.py`).

## 2. One run

```bash
# Check the card first; a reservation by an idle container is not occupancy.
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader

GPU=0 AGENT_CLI=claude AGENT_MODEL=claude-opus-5-5 AGENT_EFFORT=max \
    setsid nohup bash compression_bench/scripts/compress_evolve/evolve-task.sh \
    qwen3_8_27b_12g_standard_1xH200NVL < /dev/null > evolve.log 2>&1 &
ps -eo pid,etime,cmd | grep '[s]cripts/evolve'        # record the driver's pid (setsid forks, so $! is wrong)
```

Another provider runs through Claude Code's Anthropic-compatible route, given a
`KEY=VALUE` route file (`ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY` /
`ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_MODEL` and the model aliases, the provider's
context window as `CLAUDE_CODE_MAX_CONTEXT_TOKENS`). Templates for the
open-weight agents' routes are in `compression_bench/scripts/agent/routes/*.env.example`: copy
one outside the repository, fill in the key, `chmod 600` it, and pass it as
`AGENT_ENV`:

```bash
install -m 600 compression_bench/scripts/agent/routes/glm-5p3-flash-fireworks.env.example \
    ~/.config/compression_bench/glm-5p3-flash-fireworks.env           # then fill in the key
GPU=1 AGENT_CLI=claude AGENT_MODEL=accounts/fireworks/models/glm-5p3-flash AGENT_EFFORT=max \
    AGENT_ENV=~/.config/compression_bench/glm-5p3-flash-fireworks.env \
    setsid nohup bash compression_bench/scripts/compress_evolve/evolve-task.sh \
    qwen3_8_27b_8g_standard_1xH200NVL < /dev/null > evolve.log 2>&1 &
```

With `AGENT_ENV` the file is the run's only credential: it is staged into
`<run>/secrets/`, bound read-only at `/run/compression-bench-agent.env`, sourced
inside the launch command (no key on any command line), and deleted when the
agent's environment stops; the CLI's own login is not staged, and egress admits
the base URL's host alone.

| knob | default | meaning |
|---|---|---|
| `GPU` | `auto` | a card with no compute process (sampled), not claimed by another run, not excluded |
| `HOURS` | contract `budget.agent_hours` (8) | the agent's budget; the verdict runs after it |
| `AGENT_CLI` | `claude` | `claude` or `codex`; the host's native binary is mounted read-only |
| `AGENT_MODEL` / `AGENT_EFFORT` | `claude-opus-5-5` / `max` (codex: `gpt-5.6-sol` / `max`) | passed to the CLI; the model and effort are part of the lineage id |
| `AGENT_ENV` | none | a provider route file (above; templates in `scripts/agent/routes/`) |
| `SEED_WORKSPACE` | none | an ended run of the same lineage whose `work/` files (code, notes, logs; never weights) the fresh run starts with |
| `MODEL_DIR` | `model_for_task.py` | the pinned BF16 checkpoint |
| `DEBUG_MODEL` | `0` | `1` mounts the task's debug checkpoint |
| `COMPRESSION_BENCH_EXCLUDED_GPUS` / `COMPRESSION_BENCH_AUTO_EXCLUDED_GPUS` | none | cards never used / never picked automatically |
| extra args | | `--hours H`, `--dry-run` (build and boot only), `--scratch` (scratch namespace, no CSV row), `--lineage-root DIR`, `--lineage-id ID`, `--no-verdict`, `--keep-weights` (no cleanup), `--keep-checkpoints N` |

What happens: the run id is allocated, the GPU claimed, the checkpoint and
image freshness checked, the workspace built (`work/AGENT_TASK.md`, `./clock`,
`./version`, staged credentials), Harbor starts `main` + `public-checker`
behind the egress allowlist, the boot proof runs in the container (model
readable, one GPU visible, the CLI runs, home and submission writable, the paper
collection searchable, the Checker answering
through its socket, the provider reachable and everything else blocked), and the
agent CLI starts with the task's instruction and this run's real budget. A
session that ends early is resumed in place, as often as it ends, until the
deadline (a session that ends idle again waits 30 s more each time, at most
5 min; a platform session limit is waited out); resumability is read from the
session's transcript file, and a session that cannot be resumed (its transcript
is gone) is cold-restarted. At the deadline the final version is captured, the
environment is stopped (freeing the single-copy lock), the hidden Verifier runs
once in its own no-network container on the GPU, and then each checkpoint's
verdict (§3).

A provider that refuses for want of credit (HTTP 402 or its credit messages)
puts the run on hold: the container is paused with its files, GPU memory and
card claim kept, `<run>/CREDIT_HOLD.json` keeps the janitor off it, and the
remaining agent time is restored once the route's key has credit again. The
hold ends without credit after `COMPRESSION_BENCH_CREDIT_HOLD_MAX_H` hours
(default 72) or when `<run>/END_CREDIT_HOLD` is created; the agent phase then
ends (`credit`) and the run is verdicted as it is.

`--hours` bounds the agent, not the run: after the deadline the Verifier runs
(up to the task's `budget.verifier_timeout_s` plus its build) and then every
checkpoint verdict, so a card stays busy well past `deadline_utc`.

A smoke check of the whole loop, without polluting the record:

```bash
GPU=0 HOURS=1.5 AGENT_EFFORT=high bash compression_bench/scripts/compress_evolve/evolve-task.sh \
    qwen3_8_27b_12g_standard_1xH200NVL --scratch --lineage-root compression_bench/evolve_space/smoke-output
```

### Without the lifecycle: plain `harbor run`

Every rendered task is a valid Harbor 0.21 task (separate no-network Verifier
environment, `artifacts = []`), so a Harbor-installed agent can run it directly.
The lifecycle's versions, resume, budget prompt and records are then absent, and
Harbor runs the Verifier right after the agent:

```bash
export COMPRESSION_BENCH_RUN_DIR=$PWD/jobs/<new run name> COMPRESSION_BENCH_GPU_DEVICE=0 \
       COMPRESSION_BENCH_MODEL_DIR=/path/to/Qwen3.8-27B-bf16 \
       COMPRESSION_BENCH_HOST_UID=$(id -u) COMPRESSION_BENCH_HOST_GID=$(id -g)
mkdir -p $COMPRESSION_BENCH_RUN_DIR/work/.agent-home $COMPRESSION_BENCH_RUN_DIR/work/.tmp $COMPRESSION_BENCH_RUN_DIR/submission
touch $COMPRESSION_BENCH_RUN_DIR/.submission.lock
"$COMPRESSION_BENCH_HARBOR_PY" -m harbor.cli.main run -p compression_tasks/qwen3_8_27b_12g_standard_1xH200NVL \
    -a <harbor agent> -m <model> --env docker --n-attempts 1 --n-concurrent 1 \
    --max-retries 0 --job-name harbor --jobs-dir $COMPRESSION_BENCH_RUN_DIR
```

## 3. What a run did

```bash
R=compression_bench/evolve_space/evolve/<task>/<run-id>
"$COMPRESSION_BENCH_PY" -c "import json,sys; [print(json.loads(l)['event']) for l in open(sys.argv[1])]" $R/journal.jsonl | tail -20
cat $R/SUMMARY.json
cat $R/verifier/score.json                                 # the hidden verdict: reward, R_ppl, the correctness gate; stages in verifier/*.json
cat compression_output/<lineage>/FINAL.json            # the verdict pointer, matched version
cat compression_output/<lineage>/versions.jsonl        # the agent's ./version history
cat evaluation_progress/<task>.csv                         # one row per production run
cat $R/METRICS.json                                        # everything, kept after the weights go
cat $R/TIMING.json                                         # phases, launches, public checks, verdict stages, time per tool
tail $R/resources.jsonl                                    # GPU memory/utilisation, container RAM, disk, every 60 s
ls $R/trajectory/                                          # steps.jsonl, transcripts/, notes/, memory/, TRAJECTORY.json
```

`FINAL.json` status is `SCORED` (reward > 0), `ZERO` (a gate failed; the stage
and reason are recorded) or `NOT_RUN` (infrastructure); it is the one verdict
vocabulary (journal `final_validation.verdict`, `SUMMARY.json` `verdict`, the
progress CSV). The run's own states are `CREATED → WORKSPACE_READY →
AGENT_RUNNING → VERDICT → COMPLETED / FAILED / CANCELLED`; why the agent phase
ended is `agent_phase_ended.reason` (`deadline`, `terminal`, `credit` or
`launch_cap`).

### Weights are cleaned; metrics are kept

Every run adds full-size weights: the agent's candidates in `work/`, the
submission, and one reflink checkpoint per distinct `./version` (an unchanged
submission is not recorded: `./version` answers `unchanged`; without reflink
support, or below `COMPRESSION_BENCH_SNAPSHOT_MIN_FREE_GB` free, default 100, a
version is manifest-only). Checkpoints are measured, then deleted: a snapshot
stays while the run lives unless the scheduler's janitor cleans it (§4); after
the final verdict `final_validate/tools/verdict_checkpoints.py` runs the same full hidden
verdict on each distinct one on the run's card, in
`<run>/checkpoint-verdicts/<vN>/` (evidence: `<lineage>/CHECKPOINT_VERDICTS/<vN>/`,
`checkpoint_verdicts.jsonl`, `verdict` in each `VERSION.json`); a checkpoint
identical to the final submission takes the final verdict, and when the final
verdict did not measure (the host failed) the final snapshot's verdict becomes
`FINAL.json`. The driver's last step writes `METRICS.json` (run dir, and
`compression_output/<lineage>/METRICS/<run id>.json`): verdict and every
hidden stage, `verdicts` (the final and each checkpoint verdict in detail: the
per-block hidden PPLs, the gate's per-benchmark counts and predictions, every
decode request, the resident census by module, stage timings), every version's
size, composition and public Checker result, every public check, agent usage and
cost, timeline, disk footprint. It reads the file back, then deletes the
workspace's weight files (any file with a weight suffix of 1 MiB or more, any
non-text file of 256 MiB or more), `work/.tmp`, `work/.agent-home/.cache`, every
checkpoint a verdict measured (`COMPRESSION_BENCH_KEEP_CHECKPOINTS`, default 0,
keeps more) and the submission's weights once the final verdict measured them,
and lists what went in `METRICS.json`. An unmeasured checkpoint, and the
submission while the final verdict has not measured it, stay. A Verifier stage
that fails with `No space left on device` is `NOT_RUN`, never a 0. Code, logs,
manifests (every file's SHA-256) and the verdict evidence stay.

```bash
# The checkpoint verdicts of an ended run, again or for the first time:
"$COMPRESSION_BENCH_PY" compression_bench/harness/final_validate/tools/verdict_checkpoints.py --run $R --gpu N
# Collect metrics and free the weights of ended runs:
"$COMPRESSION_BENCH_PY" compression_bench/scripts/harvest_runs.py --all --dry-run   # what would go
"$COMPRESSION_BENCH_PY" compression_bench/scripts/harvest_runs.py --run $R          # one finished run
"$COMPRESSION_BENCH_PY" compression_bench/scripts/harvest_runs.py --all             # every finished run
```

A live run is skipped by `harvest_runs.py`. A checkpoint still on disk restores with
`cp -a --reflink=auto compression_output/.objects/<lineage>/vN/. $R/submission/`.

A run refuses to start below 60 GiB free on the Docker root (every container's
own filesystem) or 100 GiB on the run root (`COMPRESSION_BENCH_MIN_FREE_DOCKER_GIB`,
`COMPRESSION_BENCH_MIN_FREE_RUN_GIB`).

Re-run only the verdict of a finished run (for example after an infrastructure
failure; the previous attempt is kept as `verifier.previous.<n>/`):

```bash
"$COMPRESSION_BENCH_PY" compression_bench/harness/final_validate/tools/validate_evolve.py --run $R
"$COMPRESSION_BENCH_PY" evaluation_progress/record_run.py --run $R
```

## 4. A sweep: the scheduler and its helpers

`compression_bench/scripts/scheduler.py` keeps a sweep running until every item
has finished. Its state is a queue file; its log (`scheduler.log`), `report.json`
and pid file go beside it:

```json
{"items": [{"id": "qwen8g-opus", "task": "qwen3_8_27b_8g_standard_1xH200NVL", "agent": "opus",
            "status": "queued", "attempts": 0}]}
```

`agent` names an entry of the scheduler's `AGENTS` table (CLI, model, effort and,
for a provider route, its env file: the filled-in copy of
`scripts/agent/routes/<name>.env.example` saved as `<name>.env` under
`COMPRESSION_BENCH_CONFIG_DIR`, default `~/.config/compression_bench`).

```bash
"$COMPRESSION_BENCH_PY" compression_bench/scripts/scheduler.py \
    --queue compression_bench/evolve_space/scheduler/queue.json --max-live 4 --exclude-gpus 0 --dry-run --once
setsid -f nohup bash compression_bench/scripts/scheduler_watchdog.sh \
    compression_bench/evolve_space/scheduler/queue.json < /dev/null > /dev/null 2>&1   # restarts a dead scheduler
setsid -f nohup bash compression_bench/scripts/sweep_guardian.sh \
    compression_bench/evolve_space/scheduler/queue.json opus < /dev/null >> guardian.out 2>&1   # disk and alerts
```

At most `--max-live` evolve drivers run at once, counting this checkout's and
those of any checkout named in `COMPRESSION_BENCH_SHARED_CHECKOUTS` (paths
separated by `:`) that shares the host's GPUs and disk. The scheduler launches
`queued` items as a slot, a free GPU and the disk allow, follows each run to its
end, and queues a run that a system fault ended (GPU taken, failed setup, killed
container, dead driver, full disk) again, seeded from its own earlier workspace,
up to `--attempts` times. Every `--clean-every` seconds it cleans: cold
checkpoints of live runs (`checkpoint_janitor.py`: the newest checkpoint and
anything written in the last `--cold-min` minutes stay; the data of a deleted
version goes to `<lineage>/CHECKPOINT_NOTES/`, of a workspace directory to
`<run>/cleaned_checkpoints.jsonl`; `checkpoint_probe.py` measures a checkpoint on
a spare card before its weights go), the weights of ended runs once measured,
compose projects of ended or driverless runs, and exited leftovers; a second
copy of a run's BF16 checkpoint in its workspace is removed
(`bf16_copy_guard.py`). The disk tiers are `--low-gib` (clean, prune Docker),
`--launch-gib` (launch nothing below it) and `--critical-gib` (clean at once).
The agents are told the cleaning rule in their instruction (`## Checkpoints on
disk`). A driver whose disk is full pauses and continues, it is never killed for
it.

`model_cache_ram.py spill compression_tasks/<task>` moves an idle cached
checkpoint's verified blobs to RAM (tmpfs) and validates the container mounts;
`model_cache_ram.py watch <record.json>` frees the RAM once the scheduler has
evicted that cache. `openrouter_prices.py` refreshes the
price snapshot open-weight agents are costed by.

## 5. Stopping a run

`kill -TERM <driver pid>` cancels: the agent environment is stopped by its
compose project, the run is journalled `CANCELLED`, and no verdict runs. Stop a
process by its recorded pid: `pkill -f` or `pgrep -f` on task text matches the
agent's own command line (and the shell running it). Never stop a container you
did not start; every container of a run is in the compose project named in
`$R/container.json`.

## 6. Changing tasks and the harness

Tasks change only through `taskctl` (`show`, `check`, `render`, `set`, `new`,
`rename`, `model-facts`; see
[`compression_tasks/_manage/README.md`](../compression_tasks/_manage/README.md)),
and every change passes `bash compression_bench/scripts/correctness-gate.sh`.
An evaluator change is made once in `compression_bench/harness/final_validate/`,
then `taskctl render`; `compression_bench/tests/test_evaluator_parity.py` and
`compression_bench/tests/test_fidelity_reward.py` must stay green.

### A task for a new model

```bash
# 1. CPU, with network once: the contract, the official tokenizer, starting perplexity
#    blocks and decode prompts (C4 validation), and the hidden correctness gate.
CTL="$COMPRESSION_BENCH_PY compression_tasks/_manage/taskctl.py"
"$DEPLOY_PY" compression_bench/scripts/deploy/create_model_task.py --task-id <slug>_<N>g_standard_1xH200NVL \
    --model-dir <snapshot> --model-id <org/name> --revision <sha> --title <Name> \
    --local-name <Name> --keyword <family> [--mtp-note]
$CTL model-facts <task> --model-dir <snapshot>

# 2. On a free GPU (inspect it first): the fidelity blocks. 512 tokens of Dolmino-mix text per
#    block, continued for 1536 tokens by the BF16 checkpoint itself, and BF16's perplexity on
#    them, measured through the task's starter and the evaluator's own code.
"$DEPLOY_PY" compression_bench/scripts/deploy/build_fidelity_assets.py --gpu N --install compression_tasks/<task>
$CTL set <task> --ppl-protocol mixed-bf16-continued-256x2048-v1
# ...then re-pin the training rows that overlap the perplexity documents (every model's build),
# and rebuild agent-base.
"$DEPLOY_PY" compression_bench/scripts/deploy/prepare_datasets.py --pin-perplexity-overlap \
    compression_bench/evolve_space/baselines/fidelity-dolmino-*/source/*/documents.jsonl
bash compression_bench/scripts/deploy/01-build-images.sh agent-base platform-gguf

# 3. The BF16 accuracy on the hidden gate (one card), or every BF16 baseline for a checkpoint
#    larger than the card: part of the weights in host memory, or split over cards.
"$COMPRESSION_BENCH_PY" compression_bench/scripts/deploy/measure_bf16_accuracy.py --gpu N compression_tasks/<task>
"$COMPRESSION_BENCH_PY" compression_bench/scripts/deploy/measure_bf16_baselines.py --gpu N --cpu-offload-gb 130 compression_tasks/<task>
"$COMPRESSION_BENCH_PY" compression_bench/scripts/deploy/measure_bf16_baselines.py --gpu N,M --tensor-parallel 2 compression_tasks/<task>

# 4. Render, check, and clone the other capacity tiers.
$CTL render <task>; $CTL check <task>
$CTL new <other tier> --from <task> --capacity-gib M
```

The perplexity protocols are `final_validate/task_policy.LIVE_PROTOCOLS`:
`mixed-bf16-continued-256x2048-v1`, and `mixed-gemma-chat-bf16-continued-256x2048-v1`
for Gemma, whose blocks open with its 19-token chat prefix (a model whose blocks
need another prefix needs its own protocol entry there).
`measure_bf16_baselines.py --measure-only` prints a measurement beside the
recorded values and records nothing (a runtime change is checked this way before
any baseline moves).

### Correctness-gate inputs and BF16 baselines

The gate's questions and the BF16 checkpoint's accuracy on them are task sources
under `assets/hidden/` only; the Checker and the agent never hold them:

```bash
# Once, with network: pinned dataset revisions, sampled and tokenized per task (CPU).
"$DEPLOY_PY" compression_bench/scripts/deploy/build_accuracy_assets.py compression_tasks/<task> [...]
# On a free GPU: the BF16 checkpoint through the starter and the evaluator's own accuracy code,
# in the runtime image; recorded in every task given.
"$COMPRESSION_BENCH_PY" compression_bench/scripts/deploy/measure_bf16_accuracy.py --gpu N compression_tasks/<task> [...]
$CTL render
```

The tasks passed to one measurement must share the model and the inputs (the
capacity tiers of one model). A rebuilt split whose inputs changed
has no baseline until it is measured again; `taskctl check` refuses it. The
evidence of every measurement is kept under `compression_bench/evolve_space/baselines/`.
