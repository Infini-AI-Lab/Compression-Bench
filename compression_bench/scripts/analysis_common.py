"""Paths, names and readers shared by the trajectory-analysis scripts (stdlib only), and the order they run in.

The analysis reads finished runs (`compression_bench/evolve_space/evolve/*/<run>/`), their lineages
(`compression_output/<lineage>/`) and the task directories, and writes into one analysis directory DIR (default
`evaluation_progress/trajectory_analysis/`; every script takes `--data DIR` or `--out DIR`). Each script's docstring
documents its inputs, outputs and rules. Host interpreter, no GPU; `insight_data.py` needs `tokenizers`, the figure
scripts `matplotlib` and `numpy`, `generation_failures.py` Docker.

Run order (script: what it writes to DIR):
   1 trajectory_features.py [--runs FILE | --latest]  features.json, features.csv, selected_runs.json (the selection
                                                     every later step uses)
   2 byte_allocation.py                               byte_allocation.json
   3 search_paths.py                                  search_paths.json
   4 paper_usage.py                                   paper_usage.json (with paper_catalog.json)
   5 gate_accuracy.py                                 gate_accuracy.json
   6 insight_data.py                                  insights_hidden.json
   7 version_ledger.py                                version_ledger.json (needs 5)
   8 design_pairs.py --pairs PAIRS.json               design_pairs.json (needs 7)
   9 generation_failures.py                           generation_failures.json (needs 7)
  10 verified_results_table.py                        verified_results.md, verified_results.json (with the baseline
                                                     records of baselines/released_baseline.py and baseline_latency.py)
  11 methods_report.py [--runs FILE]                  methods_<date>.md
  12 analysis_figures.py [--families FILE]            figures/ (needs 1-7 and baselines/ released/RESULTS.json)
  13 findings_figures.py                              figures/ (needs 7-9 and studies/insights/quant_error_assemble.py's
                                                     quant_error.json)
studies/agent_decisions.py also reads version_ledger.json.
"""
from __future__ import annotations

import gzip
import json
import re
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVOLVE = REPO / "compression_bench" / "evolve_space" / "evolve"
PROTOTYPES = REPO / "compression_output"
DATA = REPO / "evaluation_progress" / "trajectory_analysis"     # the analysis directory every script defaults to
BASELINES = REPO / "evaluation_progress" / "baselines"

#: Agent model id (METRICS.json `agent.agent_model`) -> short key, and the key's full name, in table order.
AGENT_KEY = {"claude-sonnet-5-5": "sonnet", "claude-opus-5-5": "opus", "accounts/fireworks/models/glm-5p3": "glm",
             "accounts/fireworks/models/glm-5p3-flash": "glm-flash", "moonshotai/kimi-k3.1-pro": "kimi",
             "accounts/fireworks/models/minimax-m3p1-flash": "minimax",
             "accounts/fireworks/models/deepseek-v4p1-flash-1017": "deepseek"}
AGENT_NAME = {"sonnet": "Claude Sonnet 5.5", "opus": "Claude Opus 5.5", "glm": "GLM 5.3", "glm-flash": "GLM 5.3 Flash",
              "kimi": "Kimi K3.1 Pro", "minimax": "MiniMax M3.1 Flash", "deepseek": "DeepSeek V4.1 Flash 1017"}
#: Task model slug (the task id's prefix) -> short key, and the model's name.
MODEL_KEY = {"nemotron3_5_super_vl_120b_a12b": "nemotron", "qwen3_5_122b_a10b": "qwen122", "qwen3_8_27b": "qwen8",
             "gemma4_26b_a4b_it": "gemma"}
MODEL_LABEL = {"nemotron3_5_super_vl_120b_a12b": "Nemotron-3.5 120B-A12B", "qwen3_5_122b_a10b": "Qwen3.5 122B-A10B",
               "qwen3_8_27b": "Qwen3.8 27B", "gemma4_26b_a4b_it": "Gemma-4 26B-A4B"}


def agent_key(agent_model: str) -> str:
    return AGENT_KEY.get(agent_model, agent_model)


def model_key(task_id: str) -> str:
    return next((v for k, v in MODEL_KEY.items() if task_id.startswith(k)), task_id)


def short_task(task_id: str) -> str:
    """A task's short id: its full id without the standard treatment's suffix."""
    return task_id.removesuffix("_standard_1xH200NVL")


def task_dir(task: str) -> Path:
    """A task's directory from its full id or its short id."""
    path = REPO / "compression_tasks" / task
    return path if path.is_dir() else REPO / "compression_tasks" / f"{task}_standard_1xH200NVL"


def utc_seconds(stamp: str) -> float:
    """POSIX seconds of a harness timestamp (`YYYY-MM-DDTHH:MM:SSZ`)."""
    return datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").timestamp()


def read_selection(path: Path) -> set[str]:
    """Run ids from a JSON list, {"runs": [id or {"run": id}]}, or a text file with one id per line."""
    text = Path(path).read_text()
    try:
        got = json.loads(text)
    except json.JSONDecodeError:
        return {line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")}
    items = got.get("runs", []) if isinstance(got, dict) else got
    return {i["run"] if isinstance(i, dict) else str(i) for i in items}


def run_dir(run: str) -> Path | None:
    """The run's directory under EVOLVE, or None."""
    found = sorted(EVOLVE.glob(f"*/{run}"))
    return found[0] if found else None


def trajectory_dir(run: str) -> Path | None:
    """The run's trajectory in its lineage (`<lineage>/TRAJECTORY/<run>/`), or None."""
    found = sorted(PROTOTYPES.glob(f"*/TRAJECTORY/{run}"))
    return found[0] if found else None


def steps(run: str) -> list[dict]:
    """The run's trajectory steps (`steps.jsonl.gz`); empty without a trajectory."""
    found = trajectory_dir(run)
    path = found / "steps.jsonl.gz" if found else None
    if not path or not path.is_file():
        return []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def selected_runs(data: Path = DATA) -> list[dict]:
    """(task, task_id, model, cap, agent, run, run_dir) of every selected run: `selected_runs.json` (trajectory_features.py)."""
    path = Path(data) / "selected_runs.json"
    if not path.is_file():
        raise SystemExit(f"{path} is missing: run trajectory_features.py first")
    out = [dict(r) for r in json.loads(path.read_text())["runs"]]
    for r in out:
        found = run_dir(r["run"])
        if found is None:
            raise SystemExit(f"run {r['run']} is not under {EVOLVE}")
        r["run_dir"] = str(found)
    return out


def lineage_checkpoint_verdicts(lineage: Path, run: str, statuses: tuple[str, ...] | None = None) -> dict[str, dict]:
    """version -> the run's last checkpoint verdict measured after the run (`<lineage>/checkpoint_verdicts.jsonl`),
    only those whose status is in `statuses` when given."""
    path = Path(lineage) / "checkpoint_verdicts.jsonl"
    rows = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("run_id") == run and (statuses is None or row.get("status") in statuses):
                rows[row["version"]] = row
    return rows


def created(run_id: str) -> str:
    """A run id's creation stamp (`<hash>-<YYYYMMDDTHHMMSSZ>-<suffix>`), for ordering a task's runs."""
    found = re.search(r"\d{8}T\d{6}Z", run_id)
    return found.group(0) if found else run_id
