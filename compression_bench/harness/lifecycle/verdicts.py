"""What one full hidden verdict measured, in detail, and whether it measured at all.

Every checkpoint of a run (and the final submission) goes through the task's own
Verifier -- package check, smoke, hidden PPL, the correctness gate, decode --
and its outputs are kept whole under the lineage (`FINAL_VALIDATION/<run id>/`,
`CHECKPOINT_VERDICTS/<version>/`). `detail` turns one such folder into the record
METRICS.json carries: every stage's scalars plus the per-block perplexities, the
gate's per-benchmark counts and predictions, every decode request, the resident
census grouped by module, and the stage timings. Host-side only: none of this
reaches an agent.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

from .boundary import identity_of
from .utils import read_json

STAGES = ("check", "smoke", "ppl", "accuracy", "decode", "score")
#: A Verifier failure with one of these in its error is the host's, not the submission's.
INFRA_MARKERS = ("No space left on device", "[Errno 28]", "ENOSPC", "Disk quota exceeded",
                 # another process held the card's memory when the engine started
                 "Free memory on device", "less than desired GPU memory utilization")
#: Paths that do not change what a submission computes: interpreter caches, and the
#: tokenizer assets the evaluator replaces with the official ones (tokenizer_assets.py).
IGNORED = re.compile(r"(^|/)__pycache__/|\.pyc$|^model/chat_templates/|^model/(tokenizer\.json|"
                     r"tokenizer_config\.json|vocab\.json|merges\.txt|special_tokens_map\.json|"
                     r"added_tokens\.json|tokenizer\.model|spiece\.model|chat_template\.jinja|"
                     r"chat_template\.json)$")


def is_infra_error(text) -> bool:
    return isinstance(text, str) and any(marker in text for marker in INFRA_MARKERS)


def content_identity(manifest: list[dict]) -> str | None:
    """One digest over what a submission computes with: every path, size and file
    digest except interpreter caches (two versions differing only there are one)."""
    rows = [entry for entry in manifest if not IGNORED.search(entry.get("path", ""))]
    if not rows:
        return None
    return identity_of(sorted(rows, key=lambda e: e["path"]))


def measured(summary: dict) -> bool:
    """A verdict is a measurement when the Verifier ran to a reward on the host's terms."""
    return (summary.get("status") in ("SCORED", "ZERO") and not summary.get("infra_error")
            and not is_infra_error(str(summary.get("error"))))


def _stats(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)

    def quantile(q: float) -> float:
        return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]
    return {"n": len(ordered), "mean": sum(ordered) / len(ordered), "min": ordered[0],
            "p10": quantile(0.1), "median": quantile(0.5), "p90": quantile(0.9),
            "max": ordered[-1]}


def _module_key(path: str) -> str:
    parts = [p for p in path.split(".") if p not in ("_modules", "_parameters", "_buffers", "module")]
    for anchor in ("layers", "embed_tokens", "lm_head", "norm", "visual", "mtp"):
        if anchor in parts:
            parts = parts[parts.index(anchor):]
            break
    return re.sub(r"\.\d+(?=\.|$)", ".N", ".".join(parts))[-120:]


def census(memory: dict, top: int = 30) -> dict:
    """The resident-storage census of the largest snapshot, grouped by module."""
    if not isinstance(memory, dict):
        return {}
    snapshots = memory.get("snapshots") or []
    out = {"max_resident_bytes": memory.get("max_resident_bytes"),
           "limit_gib": memory.get("limit_gib"), "passed": memory.get("passed"),
           "snapshots": [{"phase": s.get("phase"), "resident_bytes": s.get("resident_bytes"),
                          "passed": s.get("passed")} for s in snapshots if isinstance(s, dict)]}
    workers = [w for s in snapshots if isinstance(s, dict) for w in (s.get("workers") or [])]
    if not workers:
        return out
    worker = max(workers, key=lambda w: w.get("bytes") or 0)
    groups: dict[tuple, list[int]] = {}
    by_dtype: dict[str, int] = {}
    for storage in worker.get("storages") or []:
        key = (_module_key(storage.get("first_path", "")), storage.get("dtype"))
        slot = groups.setdefault(key, [0, 0])
        slot[0] += storage.get("bytes") or 0
        slot[1] += 1
        by_dtype[storage.get("dtype")] = by_dtype.get(storage.get("dtype"), 0) + (storage.get("bytes") or 0)
    out.update(storage_count=worker.get("storage_count"), worker_bytes=worker.get("bytes"),
               bytes_by_dtype=dict(sorted(by_dtype.items(), key=lambda kv: -kv[1])),
               largest_groups=[{"module": k[0], "dtype": k[1], "bytes": v[0], "storages": v[1]}
                               for k, v in sorted(groups.items(), key=lambda kv: -kv[1][0])[:top]])
    return out


def detail(folder: Path | str) -> dict:
    """Everything one Verifier run measured, from its output folder."""
    folder = Path(folder)
    stages = {name: read_json(folder / f"{name}.json", None) for name in STAGES}
    pick = lambda data, keys: {k: (data or {}).get(k) for k in keys if k in (data or {})}  # noqa: E731
    ppl = stages["ppl"] or {}
    windows = [w for w in ppl.get("windows") or [] if isinstance(w, dict) and w.get("scored_tokens")]
    block_ppl = [math.exp(w["nll"] / w["scored_tokens"]) for w in windows]
    accuracy = stages["accuracy"] or {}
    decode = stages["decode"] or {}
    score = stages["score"] or {}
    memory = (score.get("model_memory") or decode.get("model_memory") or ppl.get("model_memory")
              or accuracy.get("model_memory"))
    harbor = read_json(folder / "harbor-verify.json", {}) or {}
    return {
        "evidence": str(folder),
        "stages_present": [name for name in STAGES if stages[name] is not None],
        "package": pick(stages["check"], ("valid", "total_bytes", "total_gib", "limit_gib",
                                          "file_count", "tokenizer_bytes", "error")),
        "smoke": pick(stages["smoke"], ("valid", "generated_text", "error")),
        "ppl": {**pick(ppl, ("valid", "ppl", "nll", "predicted_tokens", "input_tokens", "blocks",
                             "ppl_split", "ppl_protocol", "bf16_ppl", "batch_size", "error")),
                "block_ppl": block_ppl, "block_ppl_stats": _stats(block_ppl)},
        "accuracy": {**pick(accuracy, ("valid", "accuracy", "chance", "items", "sequences",
                                       "scored_tokens", "input_tokens", "accuracy_split", "error")),
                     "benchmarks": accuracy.get("benchmarks"),
                     "predictions": accuracy.get("predictions")},
        "decode": {**pick(decode, ("valid", "complete", "device", "input_tokens", "output_tokens",
                                   "concurrency", "measured_requests", "engine_startup_seconds",
                                   "warmup_seconds", "median_round_mean_seconds",
                                   "decode_tokens_per_second", "round_mean_seconds", "round_cv",
                                   "round_spread_fraction", "error")),
                   "requests": [pick(r, ("round", "request", "input_tokens", "output_tokens",
                                         "time_to_first_token_seconds", "decode_seconds",
                                         "decode_tokens_per_second"))
                                for r in decode.get("requests") or [] if isinstance(r, dict)]},
        "score": {k: v for k, v in score.items()
                  if not isinstance(v, (dict, list)) or k == "round_mean_seconds"},
        "census": census(memory),
        "stage_seconds": {s.get("stage"): s.get("seconds")
                          for s in (read_json(folder / "timing.json", {}) or {}).get("stages") or []
                          if isinstance(s, dict)},
        "harbor": pick(harbor, ("build_and_start_seconds", "seconds", "returncode", "timeout_s",
                                "started_utc", "finished_utc")),
    }
