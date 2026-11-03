# evaluation_progress/

## The paper's results

- `leaderboard.csv`: the final verification of every model in the paper's six
  settings (Nemotron-3.5 120B-A12B and Qwen3.5 122B-A10B at 32 and 48 GiB,
  Qwen3.8 27B at 8 and 12 GiB): the BF16 reference, GPTQ, the released
  quantization of each setting and the six agents. Columns: perplexity on the
  hidden BF16-generated continuations (`ppl`, lower is better), mean GSM8K,
  HumanEval and TriviaQA accuracy (`accuracy_pct`), package and GPU-resident
  model storage (GiB), batch-one decode tokens per second (`decode_tps`,
  reported, not scored) and the gate outcome (`Pass`, `Fail`, `Timeout`). An
  empty cell is a value the paper does not report.
- `ablations.csv`: the paper's repeated eight-hour designs (GLM 5.3 and
  DeepSeek V4.1 on Qwen3.8 27B / 12 GiB and Qwen3.5 122B-A10B / 48 GiB) and the
  controlled quantization ablations of GLM's repeated designs (three-bit MLP
  clipping range; GPTQ error compensation), with per-benchmark accuracy.

## Run records

After a production run's verdict, the harness calls `record_run.py`, which
writes one row per run into `evaluation_progress/<task>.csv` (MLE-Bench's
ServerBench row shape). The row holds the reward, the gate measurements,
latency, agent cost and pointers to the run's evidence in
`compression_output/`; it is replaced in place when the run is recorded
again. Scratch and dry runs get no row. These CSVs, and the baseline and
analysis records the scripts under `compression_bench/scripts/` write here, are
produced locally and not tracked.

    "$COMPRESSION_BENCH_PY" evaluation_progress/record_run.py --run <run root>
    "$COMPRESSION_BENCH_PY" evaluation_progress/record_run.py --all      # every finished production run
