# `_manage/` — the task tooling

A task's sources are `contract.yaml` and `assets/`. Everything else in a task
directory is generated from them. These tools are the only writers.

| file | role |
|---|---|
| `taskctl.py` | the command line: `show`, `check`, `render`, `set`, `new`, `rename`, `model-facts` |
| `render.py` | every generated file of a task, as bytes; `write()` and `drift()` |
| `instruction_render.py` | `instruction.md` (the fact-only specification and its session sections) and `README.md`, from the contract and the assets |
| `model_facts.py` | `assets/model_facts.json` (the checkpoint's configuration and tensor inventory) and its table in the instruction |
| `sites.py` | task paths and the registry of settable scalars (one site each) |
| `blocks.py` | edits one YAML key in place, without reserialising the file |

```bash
CTL="$COMPRESSION_BENCH_PY compression_tasks/_manage/taskctl.py"
$CTL show  <task>                  # contract, rendered policy, profiles, images
$CTL check [task]                  # sources valid, generated files byte-identical to a fresh render
$CTL render [task]                 # regenerate; refuses a task whose sources are invalid
$CTL set   <task> [--capacity-gib N] [--agent-hours H] [--verifier-timeout-s S]
                  [--public-check-timeout-s S] [--build-timeout-s S] [--model-revision SHA]
                  [--version V] [--description D] [--starter S] [--platform-patches P,Q]
                  [--ppl-protocol P] [--payload-rule R] [--link-rule R] [--reason WHY]
$CTL new   <task> --from <task> --capacity-gib N [--version V] [--description D]
$CTL rename <task> --to <new id>
$CTL model-facts <task> [--model-dir DIR]   # read the pinned checkpoint's config and shard headers
```

`<task>` is a task id or its 8-hex hash.

`set` writes each parameter to its one registered site (`sites.SCALARS`),
re-renders, reads the contract back, and refuses if a value did not land.
A change that loosens the task (more capacity, more agent hours, another
payload or link rule) needs `--reason`; the reason belongs in the commit message.

A task id is ServerBench's form, `<model slug>_<capacity>g_<treatment>_<N>x<GPU>`
(`qwen3_8_27b_8g_standard_1xH200NVL`), with the treatment `standard`. `check` refuses an id that disagrees with the contract's
`model.slug`, `capacity_gib`, `treatment` and `hardware`; `rename` and `new`
write those fields from the id (a capacity change is a new task, not a `set`).

`check` validates the contract (schema, hash, policy via
`final_validate/task_policy.validate`, the scoring and perplexity protocols, the
`gguf` platform and its patches,
starter, budget), the assets (every split present, protocols matching the
policy, measured BF16 baselines, the correctness gate's inputs in the hidden
split only with their recorded digest and measured BF16 accuracy, the model
facts matching the pinned checkpoint, public and hidden inputs disjoint,
official tokenizer files present), and then that every generated file is
byte-identical to a fresh render with no extra files.

`new` clones an existing task's sources for a new capacity tier of the same
model (same tokenizer, inputs and baselines). It drops the diagnostic reference
row's label, which belongs to the source tier.
A task for a new model needs its own measured assets: the perplexity blocks
(`compression_bench/scripts/deploy/build_fidelity_assets.py`), the decode
inputs and the correctness gate's inputs
(`compression_bench/scripts/deploy/build_accuracy_assets.py`) built with that
model's tokenizer, BF16 baselines on both splits, and its model facts.

`COMPRESSION_BENCH_TASK_ROOT` points the tooling at a disposable copy of the task tree
(the tests use it).
