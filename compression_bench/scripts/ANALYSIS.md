# Run analysis and baselines

The scripts in this directory read finished runs and write the tables and figures of the paper. They run on
the host interpreter with no GPU. `insight_data.py` needs `tokenizers`, the figure scripts need `matplotlib`
and `numpy`, and `generation_failures.py` needs Docker. Their inputs are the harness's own run records:

- `compression_bench/evolve_space/evolve/<task>/<run>/`: workspaces, journals, `METRICS.json`
- `compression_output/<lineage>/`: versions, checkpoint verdicts, trajectories
- `compression_tasks/<task>/`: the tasks

They write into one analysis directory, `evaluation_progress/trajectory_analysis/` by default; pass
`--data DIR` or `--out DIR` to use another. Shared paths, agent and model names, and readers live in
[`analysis_common.py`](analysis_common.py).

## Baselines

| Baseline | Commands |
|---|---|
| GPTQ, packed and served natively by vLLM | `baselines/gptq_baseline.py plan`, then `quantize`, then `validate` |
| Latency of BF16 and GPTQ under the evaluator's decode protocol | `baselines/baseline_latency.py`, then `baselines/archive_baseline_setting.py`, then `baselines/collect_baseline_latency.py` |
| Released quantizations: Unsloth Dynamic (Nemotron 48 GiB, Qwen3.5), GSQ-RCO (Qwen3.8) | `baselines/images/build_released_images.sh`, then `baselines/released_baseline.py fetch`, then `run`, then `collect` |

`baselines/released_models.py` pins one release per task (Unsloth UD-IQ2_XXS, UD-IQ1_M, UD-IQ3_S; GSQ-RCO IQ2_XS,
IQ3_S); Nemotron 32 GiB has no release that fits. Baseline records go to `evaluation_progress/baselines/` (work files under
`compression_bench/evolve_space/baselines/`).

## Analysis pipeline

| # | Script | Writes | Needs |
|--:|---|---|---|
| 1 | `trajectory_features.py [--runs FILE \| --latest]` | `features.json`, `features.csv`, `selected_runs.json` (the run selection every later step uses; default: every finished run) | |
| 2 | `byte_allocation.py` | `byte_allocation.json` | 1 |
| 3 | `search_paths.py` | `search_paths.json` | 1 |
| 4 | `paper_usage.py` | `paper_usage.json` (with `paper_catalog.json`) | 1 |
| 5 | `gate_accuracy.py` | `gate_accuracy.json` | 1 |
| 6 | `insight_data.py` | `insights_hidden.json` | 1 |
| 7 | `version_ledger.py` | `version_ledger.json` | 5 |
| 8 | `design_pairs.py --pairs PAIRS.json` | `design_pairs.json` | 7 |
| 9 | `generation_failures.py` | `generation_failures.json` | 7 |
| 10 | `verified_results_table.py` | `verified_results.md`, `verified_results.json` | 1, baselines |
| 11 | `methods_report.py [--runs FILE]` | a methods report | |
| 12 | `analysis_figures.py [--families FILE]` | `figures/` | 1–7, released baselines |
| 13 | `findings_figures.py` | `figures/` | 7–9, `studies/insights/quant_error_assemble.py` |

To reproduce the paper's selection, pass `trajectory_features.py --runs` a file listing the 36 runs of the
main sweep (six agents × six `standard` tasks). `--latest` takes the latest run per agent and task.

The ablations and supplementary studies are in [`studies/`](studies/README.md).
