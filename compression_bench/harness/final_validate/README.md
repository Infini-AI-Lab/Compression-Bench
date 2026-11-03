# `final_validate/` — the evaluator, one source for every task

Every task's hidden Verifier (`tests/`) and public Checker
(`environment/public-checker/`) carry byte copies of these modules, written by
`taskctl render`; `taskctl check` refuses a copy that differs. The only
per-task evaluator input is `evaluator-assets/task_policy.json`, rendered from
the task's `contract.yaml` and read by `task_policy.py`.

| module | in | role |
|---|---|---|
| `task_policy.py` | both | reads and validates `task_policy.json`; the PPL protocol registry |
| `evaluate.py` | both | the stage pipeline `check → smoke → ppl → accuracy → decode → score` (accuracy in the hidden verdict only); the hidden `run` entry writes `/logs/verifier/<stage>.{json,log}` and `reward.txt` |
| `inference_runtime.py` | both | runs the submitted `inference.py` in an isolated child process; trusted token-arrival timestamps; resident-storage RPCs outside token timing |
| `ppl_benchmark.py` | both | the fixed 256×2048 perplexity blocks; PPL from submitted log-probabilities, prefix tokens unscored |
| `accuracy_benchmark.py` | both (items in the Verifier only) | the hidden correctness gate (`hidden-gen-gate-v2`): fixed items (math, code, closed-book QA; 50 each) answered by greedy generation and judged by final number, alias match or unit tests |
| `fidelity.py` | both | the `bf16-fidelity` reward (`bf16-sample-ppl-gated-v5`): `10 × PPL_bf16/PPL`, no maximum, on BF16-sampled continuations of hidden text against the split's BF16 PPL, and in the verdict the correctness gate (`max_accuracy_drop`) |
| `decode_benchmark.py` | both | single-request decode (warmup + 3×3), one token per event; the final score (the reward from `fidelity.py`, decode latency reported beside it) |
| `model_memory.py` | both | the evaluator-owned census of GPU-resident model storage inside the vLLM worker |
| `own_compression.py` | both | the package-stage check that no built-in or packaged quantization implementation serves a weight |
| `tokenizer_assets.py` | both | the official tokenizer substituted into every package; the private evaluation view; post-verdict packaging |
| `evaluate_public.py`, `server.py` | Checker | the Unix-socket Checker service and the public response |
| `tools/validate_evolve.py` | host | runs the hidden verdict for a finished run and records it in the lineage |
| `tools/verdict_checkpoints.py` | host | runs the hidden verdict on every distinct `./version` checkpoint of a run |

## The task policy

| field | values | meaning |
|---|---|---|
| `schema` | `compression-bench-task-policy-v1` | the file's format |
| `task_id` | string | the task this copy belongs to |
| `size_limit_gib` | number | package and resident-storage limit |
| `payload_rule` | `safetensors-or-gguf`, `any-payload` | what counts as weights in `submission/model/` |
| `link_rule` | `safetensors-or-gguf`, `model-payload` | which files the private view links instead of copying |
| `official_tokenizer_files` | list | task-owned tokenizer files substituted into every package |
| `ppl_protocol` | `task_policy.LIVE_PROTOCOLS`: `mixed-bf16-continued-256x2048-v1` or `mixed-gemma-chat-bf16-continued-256x2048-v1` (the debug-9b profile uses `c4-raw-256x2048-v1`) | the fixed PPL protocol (and its unscored prefix) |
| `scoring_protocol` | string | the version label every score carries |
| `quality_rule` + `quality` | `bf16-fidelity {max_accuracy_drop}` | the reward: BF16 fidelity behind the correctness gate |
| `smoke_prompt` | `plain`, `chat` | how the smoke prompt is tokenized |
| `hidden_reference_fields` | bool | report the reference PPL alongside hidden PPL |

A scored run requires the split's BF16 perplexity and the hidden gate's BF16
accuracy to be measured. The resident-storage census accepts the pinned GGUF
plugin's empty type placeholders (they hold no storage) and counts everything else.

`taskctl check` refuses a rendered copy that differs from these sources;
`compression_bench/tests/test_fidelity_reward.py` holds the reward and the gate to
their formula.
