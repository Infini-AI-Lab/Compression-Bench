# Studies

The paper's supplementary experiments, run after the agent runs have finished and been recorded. Each script's
docstring documents its arguments and record formats in full; this page lists what to run, in order, and what each
step reads and writes. Analysis directory: `evaluation_progress/trajectory_analysis/` (the selected runs, the
version ledger and the byte allocation come from `compression_bench/scripts/`: `trajectory_features.py`,
`version_ledger.py`, `byte_allocation.py`).

Host-side scripts run with the harness interpreter (`. compression_bench/scripts/env.sh`; figures need matplotlib,
`insights/` needs torch, numpy and safetensors). Scripts marked *container* are not run by hand: a launcher runs
them in the platform image (`compression_bench/platform-gguf:v1`) or the released llama.cpp runtime, with this
directory mounted read-only at `/study`. Every GPU step claims an idle card through the harness
(`compression_bench/harness/lifecycle/gpus.py`) and never contends with another process.

Shared modules: `common.py` (repository paths, JSON records and portable archives, digests, run and checkpoint
lookup, GPU claims, the `docker run` prefix) and `study_tensors.py` (tensor digests and safetensors indexes for the
container scripts).

## 1. Quantization error (`insights/`)

Weight error of controlled grids and of each analysed run's final grid on sampled real matrices, and controlled
quantization interventions on Qwen3.8-27B. Scratch: `compression_bench/evolve_space/analysis/quant_error/`.

| Step | Entry point | Reads | Writes (scratch unless noted) |
|---|---|---|---|
| sample tensors | `insights/fetch_sampled_tensors.py` | Hugging Face (range requests at the contract revisions) | `tensors/<model>/` |
| VQ codebook | `insights/train_vq_codebook.py` | nothing | `cb8_16_gauss_retrained.pt` |
| Nemotron-48 grids | `insights/grid_mse.py` | the Nemotron BF16 checkpoint (`model_for_task.py`) | `grid_mse.json` |
| controlled grids | `insights/grid_mse_all.py controlled` | `tensors/`, the local Qwen3.8-27B BF16 copy | `quant_error_grid_controlled.json` |
| run grids | `insights/grid_mse_all.py runs [AGENT\|KEY,...]`, then `pruning KEY:KEPT ...` | as above, `version_ledger.json`, the runs' work files (`insights/run_grids.py`) | `quant_error_grid_runs.json` |
| interventions | `insights/gpu_worker.py --gpu N --queue DIR` running jobs that call `insights/quant_interventions.py` | Qwen3.8-27B BF16, the task's public PPL blocks, the FineWeb sample | `interventions/<variant>.json` |
| assemble | `insights/quant_error_assemble.py` | the above, `selected_runs.json`, `version_ledger.json`, `byte_allocation.json` | `quant_error.json` in the analysis directory (read by `findings_figures.py`) |

## 2. Agent decisions

| Step | Entry point | Reads | Writes |
|---|---|---|---|
| comparison pairs | `insights/make_pairs.py SPEC.json --out PAIRS.json` | the reader's pair specification, each run's `public-checks/checks.jsonl` | the local-vs-public comparison pairs |
| audit and figure | `agent_decisions.py --pairs PAIRS.json --annotations FILE` (`--figure-only` redraws) | `selected_runs.json`, `version_ledger.json`, the runs' trajectories and public checks, the manual review | `agent_decisions/`: decision events, trace evidence, coverage, summary, `agent_decisions.{pdf,svg,png}` |

## 3. Inference implementation

| Step | Entry point | Reads | Writes |
|---|---|---|---|
| compare and figure | `inference_implementation.py extract --lineage ID --run RUN --baseline vA --variant vB`, then `figure` | two versions of one lineage with identical weights: manifests, public checks, checkpoint verdicts | `inference_implementation/` in the analysis directory |
| kernel numerics (container, runtime image) | `inference_kernel_check.py [--versions vA vB]` | each version's package at `/<version>`, the packing reference at `--helpers` | `numerical_kernel_check.json` in `--out` |

## 4. Controlled single-decision ablation

Two pairs (A selected, B with one decision ablated) built from selected runs, frozen, then freshly verified.
Artifacts: `compression_bench/evolve_space/studies/controlled_ablation/`; records:
`evaluation_progress/controlled_ablation/decisions/`, which must hold `preregistration.json` before step 3.

| Step | Entry point | Reads | Writes |
|---|---|---|---|
| 1. dense pair | `controlled_dense_launcher.py --source RUN_DIR [--model-dir DIR]` (runs `controlled_build.py`, container) | the selected dense run's `submission/` and `work/`, the BF16 checkpoint | `dense_A`, `dense_B`; `dense_build.json`, tensor audit, reproduction check |
| 2. MoE pair | `controlled_moe_launcher.py --source RUN_DIR [--model-dir DIR]` (runs `controlled_moe_build.py`, then `source_reproduction_audit.py`, container) | the selected MoE run's `submission/` and `work/` (its encoder, edited by `controlled_moe_build.py` and recorded as `moe_encoder_instrumented.py` in the study records), the BF16 checkpoint | `moe_A`, `moe_B`; `moe_build.json`, `moe_source_reproduction_audit.json` |
| 3. freeze and verify | `controlled_parallel.py` (runs `controlled_audit.py` and `controlled_verify.py`, container) | the four artifacts, each task's `tests/` | `pair_freeze_*.json`, `all_four_frozen.json`, `verification/<artifact>/`, `parallel_state.json` |
| 4. analysis | `controlled_ablation.py collect` | the verifications | `results.*`, `paired_results.*`, `paired_questions.csv`, `REPORT.md` |
| 5. finalize | `controlled_finalize.py` | everything above | `completion_checks.json`, `integrity_audit.json`, portable archives, `evidence_manifest.json` |

## 5. MoE GPTQ-compensation ablation

B rounds the source routed-expert weights to moe_A's selected grids without GPTQ compensation. Needs the finished
decision ablation. Artifact: `compression_bench/evolve_space/studies/controlled_ablation_moe_gptq/`; records:
`evaluation_progress/controlled_ablation/moe_gptq/` (with `preregistration.json` before `run`).

| Step | Entry point | Reads | Writes |
|---|---|---|---|
| build, freeze, verify | `moe_gptq_study.py run [--model-dir DIR]` (runs `moe_gptq_build.py`, `moe_gptq_representation_audit.py`, `controlled_verify.py`, container) | moe_A and the decision ablation's records, the BF16 checkpoint | B, `freeze.json`, `verification/`, `state.json` |
| analysis | `moe_gptq_study.py collect` | the verifications | paired results, `paired_sequences.csv`, `REPORT.md`, `../CURRENT_REPORT.md` (all three ablations) |
| finalize | `moe_gptq_study.py archive` | everything above | `completion_checks.json`, portable archives, `evidence_manifest.json` |

## 6. Repeatability trials and peak memory

Records: `evaluation_progress/supplemental_studies/evidence/`; work files: `compression_bench/evolve_space/studies/evidence/`.

| Step | Entry point | Reads | Writes |
|---|---|---|---|
| plan | `evidence_studies.py prepare --config TRIALS.json` | the trial definition, an isolated harness checkout and agent CLI under the work directory, the original runs | `plan.json`, `repeat_state.json` |
| run | `evidence_studies.py run` | the plan | one scratch run per trial (in the checkout), `repeat_state.json` |
| peak memory | `memory_study.py released`, `memory_study.py summarize` (runs `memory_probe.py` under Nsight Systems, container) | the released GGUF baselines (`compression_bench/scripts/baselines/`) | `memory/<task>/<artifact>/rep<N>/`, `memory_results.json` |
| status | `study_status.py [--watch]` | the above, optional `provider_pricing.json` | `repeat_results.*`, `repeat_variation.json`, `provider_costs.csv`, `peak_memory.csv`, `STATUS.md` |
| finalize | `finalize_studies.py [--decision-events agent_decisions/decision_events.json --event ID ...]` | everything above | `complete_results.csv`, `repeat_verification_evidence.json`, `original_storage.json`, `final_summary.json`, `README.md`, `evidence_manifest.json` |
