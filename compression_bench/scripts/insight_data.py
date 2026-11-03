#!/usr/bin/env python3
"""Hidden-set measurements of the selected runs, per run and per hidden PPL block.

    python3 compression_bench/scripts/insight_data.py [--data DIR]      # writes DIR/insights_hidden.json

Runs: `DIR/selected_runs.json` (trajectory_features.py; DIR defaults to evaluation_progress/trajectory_analysis/).
Per run: the hidden verdict (PPL above BF16, answers correct per benchmark). Per hidden PPL block: the run's excess
cross-entropy over BF16 on the same block (BF16's per-block NLL from the build that produced the blocks,
`ppl_config.json: baseline_evidence.result`), and whether the block is code (a line-pattern classifier over the block's
decoded tokens, prompt and continuation scored separately; only the label is kept). From these: how much a run's excess
concentrates on code blocks (mean excess on code blocks / mean excess on all blocks, with a one-sided permutation p over
20,000 random block sets of the same size, seeded by the run id) and the tail of its per-block excess. Per run also the
resident bytes by component (byte_allocation.allocation). Needs `tokenizers` (any host Python with it; no GPU).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import byte_allocation  # noqa: E402
from analysis_common import DATA, REPO, selected_runs, task_dir  # noqa: E402

CODE_THRESHOLD = 0.3     # share of a block's non-empty lines that look like code
CODE_LINE = re.compile(r"(^\s*(def |class |import |from \S+ import|public |private |static |#include|function |const |"
                       r"let |var |return\b|if \(|for \(|while \()|[{};]\s*$|^\s*[}\]]\s*$|=>|::|\(\)\s*[{:])", re.M)


def code_labels(task: str) -> list[bool]:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(task_dir(task) / "assets" / "tokenizer" / "tokenizer.json"))
    blocks = json.loads((task_dir(task) / "assets" / "hidden" / "ppl_inputs.json").read_text())   # token ids per block
    prompt = json.loads((task_dir(task) / "assets" / "hidden" / "ppl_config.json").read_text())["prompt_tokens"]

    def share(text: str) -> float:
        lines = [line for line in text.split("\n") if line.strip()]
        return sum(1 for line in lines if CODE_LINE.search(line)) / len(lines) if lines else 0.0
    # A block is code when its document prompt or BF16's continuation of it is: code seeds code.
    return [max(share(tok.decode(ids[:prompt])), share(tok.decode(ids[prompt:]))) >= CODE_THRESHOLD
            for ids in blocks]


def bf16_block_nll(task: str) -> list[float]:
    config = json.loads((task_dir(task) / "assets" / "hidden" / "ppl_config.json").read_text())
    result = REPO / config["baseline_evidence"]["result"]
    windows = json.loads(result.read_text())["windows"]
    return [w["nll"] / w["scored_tokens"] for w in sorted(windows, key=lambda w: w["block"])]


def percentile(values: list[float], q: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA, help="the analysis directory (selected_runs.json in, output out)")
    args = parser.parse_args()
    runs = selected_runs(args.data)
    out = {"code_threshold": CODE_THRESHOLD, "tasks": {}, "runs": []}
    labels_of = {}
    for task in sorted({r["task"] for r in runs}):
        labels_of[task] = code_labels(task)
        out["tasks"][task] = {"code_blocks": sum(labels_of[task]), "blocks": len(labels_of[task]),
                              "bf16_block_nll": bf16_block_nll(task)}
    for r in runs:
        m = json.loads((Path(r["run_dir"]) / "METRICS.json").read_text())
        detail = m["verdicts"]["final"]["detail"]
        ppl = (detail.get("ppl") or {})
        bf16 = out["tasks"][r["task"]]["bf16_block_nll"]
        row = {k: r[k] for k in ("task", "model", "cap", "agent", "run")}
        row["passed"] = bool((detail.get("score") or {}).get("reward"))
        row["ppl"] = ppl.get("ppl")
        bench = (detail.get("accuracy") or {}).get("benchmarks") or {}
        row["correct"] = {k: v.get("correct") for k, v in bench.items()}
        alloc = byte_allocation.allocation(Path(r["run_dir"]))
        if alloc:
            row["allocation"] = {"bits_per_weight": alloc["bits_per_weight"], "share": alloc["share"],
                                 "resident_gib": alloc["resident_gib"]}
        if ppl.get("block_ppl"):
            excess = [math.log(p) - b for p, b in zip(ppl["block_ppl"], bf16)]
            row["ppl_above_bf16_pct"] = 100 * (ppl["ppl"] / math.exp(st.mean(bf16)) - 1)
            code = [e for e, c in zip(excess, labels_of[r["task"]]) if c]
            mean = st.mean(excess)
            ratio = st.mean(code) / mean
            n, idx = len(code), list(range(len(excess)))
            rng = random.Random(r["run"])      # per run: a run's p does not depend on which runs are selected
            hits = sum(st.mean(excess[i] for i in rng.sample(idx, n)) / mean >= ratio for _ in range(20000))
            pct = [100 * (math.exp(e) - 1) for e in excess]
            row["block"] = {"mean_excess_nats": mean, "code_excess_ratio": ratio, "code_ratio_p": hits / 20000,
                            "pct_above_bf16": {"median": st.median(pct), "p95": percentile(pct, 0.95),
                                               "p99": percentile(pct, 0.99), "max": max(pct)}}
        out["runs"].append(row)
    for task in out["tasks"].values():
        task.pop("bf16_block_nll")
    (args.data / "insights_hidden.json").write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", args.data / "insights_hidden.json", len(out["runs"]), "runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
