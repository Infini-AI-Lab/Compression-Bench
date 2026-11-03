# `compression_tasks/` — the compression problems agents are judged on

One directory per task. A task is the frozen definition of one compression
problem: the source model and its pinned revision, the capacity limit, the
evaluation policy, the fixed public and hidden inputs with their measured BF16
baselines, the starter `inference.py`, and the evaluator that judges it. What
agents build for a task is recorded in `compression_output/`; the harness
that runs them is `compression_bench/`.

## The tasks

| task | hash | model | capacity | treatment | debug model |
|---|---|---|---:|---|---|
| `qwen3_8_27b_12g_standard_1xH200NVL` | `8a661afb` | `Qwen/Qwen3.8-27B` | 12 GiB | standard | Qwen3.5-9B |
| `qwen3_8_27b_8g_standard_1xH200NVL` | `b3ccdaaa` | `Qwen/Qwen3.8-27B` | 8 GiB | standard | Qwen3.5-9B |
| `gemma4_26b_a4b_it_16g_standard_1xH200NVL` | `46ebc480` | `google/gemma-4-26B-A4B-it` | 16 GiB | standard | — |
| `gemma4_26b_a4b_it_10g_standard_1xH200NVL` | `e477c2f1` | `google/gemma-4-26B-A4B-it` | 10 GiB | standard | — |
| `qwen3_5_122b_a10b_48g_standard_1xH200NVL` | `f0362287` | `Qwen/Qwen3.5-122B-A10B` | 48 GiB | standard | — |
| `qwen3_5_122b_a10b_32g_standard_1xH200NVL` | `3518d21e` | `Qwen/Qwen3.5-122B-A10B` | 32 GiB | standard | — |
| `nemotron3_5_super_vl_120b_a12b_48g_standard_1xH200NVL` | `28430d3e` | `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` | 48 GiB | standard | — |
| `nemotron3_5_super_vl_120b_a12b_32g_standard_1xH200NVL` | `834f23a2` | `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` | 32 GiB | standard | — |

Every task scores with `bf16-sample-ppl-gated-v5` behind the
`hidden-gen-gate-v2` correctness gate, runs on the `gguf` platform image with
its pinned repairs on one H200 NVL, and has an 8-hour agent budget and a
60-minute timeout for each public check and for the Verifier. `taskctl show
<task>` prints any task as the tooling reads it; the hashes are
`sha256(task id)[:8]` and are accepted wherever a task id is.

### Task ids

A task id has ServerBench's form, `<model slug>_<capacity>g_<treatment>_<N>x<GPU>`:
`qwen3_8_27b_8g_standard_1xH200NVL` is Qwen3.8-27B at 8 GiB, the standard
treatment, on one H200 NVL. Every part is a contract field (`model.slug`,
`capacity_gib`, `treatment`, `hardware`), and `taskctl check` refuses an id that
disagrees with them.

Every task is the `standard` treatment: the specification and nothing else.
The agent's container holds the inference runtime, the training and check
datasets, and the paper collection with its `compression-papers` skill.

## The policy every task follows

- **Capacity.** One limit, applied independently to the submission package
  (weights, code and the official tokenizer) and to the GPU-resident model
  storage counted by the evaluator-owned census inside the vLLM worker.
- **Reward.** Fidelity to the task's own BF16 checkpoint,
  `reward = 10 × PPL_bf16 / PPL` with no maximum, on 256 blocks of 2048 tokens.
  Each block opens with 512 tokens of text from the Dolma 3 Dolmino mix (public
  and hidden blocks from disjoint shards; for Gemma, after the official chat
  prefix), followed by 1,536 tokens the BF16 checkpoint sampled itself at
  temperature 1; only those 1,536 are scored. On such text
  `PPL_bf16 / PPL ≈ exp(−KL)`, so 10 is a submission that predicts like BF16.
  The public Checker reports it on the public blocks, the verdict on the hidden ones.
- **Correctness gate (verdict only, undisclosed to the agent).** 150 items
  answered by greedy generation through the submission's `inference.py`: 50
  GSM8K test problems (4-shot, final number), 50 HumanEval problems (unit tests
  run) and 50 TriviaQA closed-book questions (5-shot, alias match). A
  submission keeping less than 80% of BF16's mean accuracy
  (`max_accuracy_drop` 0.2) scores 0.
- **Gates.** Capacity; the compression is the submission's own (no weight is
  served through a format, loader or kernel that vLLM or another package
  ships); correctness; every stage completes. A failed gate scores 0, whatever
  the reward.
- **Speed.** Single-request decode, 2048 tokens in and 128 out, one token per
  streaming event, median of three rounds' means; speculative decoding
  prohibited. Measured, reported and required to complete; not part of the reward.
- **Public vs hidden.** The public Checker and the hidden Verifier run the same
  code on disjoint inputs. Hidden inputs exist only in a task's `tests/`; the
  evaluator and its inputs are read-only in both containers and not visible
  from the agent's.
- **The agent's container.** The task's workspace, the inference runtime
  (vLLM 0.32.0 with PyTorch and Transformers), `/opt/datasets` (WikiText-103,
  UltraChat-200k and a FineWeb sample, without any row holding an item of the
  correctness gate's benchmarks or sharing a word 13-gram with a perplexity
  document, plus MATH-500's test split as check data;
  `compression_bench/scripts/deploy/prepare_datasets.py`) and, in the standard
  treatment, `/opt/compression-papers` with the `compression-papers` skill. No
  compression toolkit, no third-party source, no guide or worked example. The
  instruction states the checkpoint's structure (`assets/model_facts.json`).

The instruction tells the agent only that a hidden corpus of mixed text
supplies the perplexity blocks and that a hidden correctness check exists;
benchmark names, sizes and the threshold appear only in this README and in each
task's repository-side `README.md`.

## Anatomy of a task

| path | source/generated | what it is |
|---|---|---|
| `contract.yaml` | SOURCE | identity, model pin, capacity, treatment, platform, evaluation policy, reference, budget, `task.toml` metadata |
| `assets/public/`, `assets/hidden/` | SOURCE | the fixed perplexity and decode inputs and their measured BF16 baselines, per split; the correctness gate's inputs (`accuracy_*.json`) in `hidden/` only |
| `assets/debug-9b/` | SOURCE | optional Qwen3.5-9B debug inputs (public Checker only) |
| `assets/tokenizer/` | SOURCE | the official tokenizer files |
| `assets/model_facts.json` | SOURCE | the checkpoint's configuration and tensor inventory (`taskctl model-facts`) |
| `task.toml` | generated | the Harbor task definition |
| `instruction.md` | generated | the one document the agent is given, then the session prompts the harness sends on a later launch |
| `README.md`, `MODE.md` | generated | repository-side overview; where each container part comes from |
| `environment/` | generated | the agent image (`Dockerfile`), compose (`main` + `public-checker`), agent tools, starter, and the Checker with the packaged evaluator and **public** inputs |
| `tests/` | generated | the hidden Verifier: packaged evaluator, **hidden** inputs, `test.sh` |

The evaluator is one source, `compression_bench/harness/final_validate/`, copied
byte for byte into `environment/public-checker/` and `tests/`; each copy's only
per-task input is `evaluator-assets/task_policy.json`.

## Fetching the tokenizers

The large tokenizer files (`tokenizer.json`) are not stored in the repository.
They are downloaded from the Hugging Face Hub at the revision each task's
`contract.yaml` pins, into `assets/tokenizer/` and both evaluator copies, with

```bash
"$COMPRESSION_BENCH_PY" compression_bench/scripts/deploy/fetch_tokenizers.py            # every task
"$COMPRESSION_BENCH_PY" compression_bench/scripts/deploy/fetch_tokenizers.py <task> ...  # selected tasks
```

Gated checkpoints need `HF_TOKEN` (or a `huggingface-cli login` token). `taskctl check` and `taskctl render` need
the tokenizer files present.

## Creating and changing tasks

Use [`_manage/`](_manage/README.md); never hand-edit a generated file.
`taskctl check` refuses a task whose generated files drift from its sources.

```bash
CTL="$COMPRESSION_BENCH_PY compression_tasks/_manage/taskctl.py"
$CTL check                                            # every task consistent
$CTL show 8a661afb                                    # a task by id or hash
$CTL set qwen3_8_27b_12g_standard_1xH200NVL --agent-hours 10 --reason "..."   # loosening needs a reason
$CTL new qwen3_8_27b_20g_standard_1xH200NVL --from qwen3_8_27b_12g_standard_1xH200NVL --capacity-gib 20
```

A task for a model no task uses yet is built from its checkpoint with
`compression_bench/scripts/deploy/create_model_task.py` and the GPU steps after
it ([`OPERATIONS.md`](../compression_bench/OPERATIONS.md) §6).

The tests that hold the tasks to this are in `compression_bench/tests/`:
`test_task_tooling.py` (sources, rendering, taskctl), `test_evaluator_parity.py`
(the Checker's copy measures what the Verifier's does), `test_fidelity_reward.py`
(the reward's formula), `test_the_instruction_only_states_facts.py` (the
instruction) and `test_agent_image_is_clean.py` (the agent's container).

## Directory map

```text
<model>_<capacity>g_<treatment>_<N>x<GPU>/   one directory per task
_manage/                        taskctl and the renderers
_template/                      instruction template, container files, starters
```
