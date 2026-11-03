"""Exercise one rendered evaluator copy's PPL reward and hidden correctness gate; print JSON.

    python3 fidelity_probe.py <evaluator dir> <public assets dir> <hidden assets dir>

Imports only the copy under test, needs no GPU, model, torch or vLLM, and
drives the measurement entry points through fake runtimes.
`test_fidelity_reward.py` checks the output against the formula computed from
the task's own assets.
"""
from __future__ import annotations

import json
import math
import os
import sys
import types
from pathlib import Path

copy_dir, public_assets, hidden_assets = (Path(a).resolve() for a in sys.argv[1:4])
sys.path.insert(0, str(copy_dir))
os.chdir(copy_dir)

transformers = types.ModuleType("transformers")
transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: types.SimpleNamespace(
    decode=lambda ids, **kw: ""))
sys.modules["transformers"] = transformers

import accuracy_benchmark  # noqa: E402
import decode_benchmark  # noqa: E402
import model_memory  # noqa: E402

policy = model_memory.read_policy(copy_dir)
out: dict[str, object] = {}


def outcome(fn, *args, **kwargs):
    try:
        return {"ok": fn(*args, **kwargs)}
    except Exception as exc:  # noqa: BLE001 - the error text is the observable
        return {"error": f"{type(exc).__name__}: {exc}"}


def memory_report():
    worker = {"bytes": 1000, "storage_count": 1,
              "storages": [{"bytes": 1000, "device": "cuda:0", "dtype": "torch.uint8",
                            "first_path": "model.w"}]}
    snapshots = [model_memory.check_snapshot({"protocol": model_memory.PROTOCOL, "workers": [worker]},
                                             policy, phase) for phase in ("load", "token_logprobs")]
    return model_memory.report(snapshots, policy)


def ppl_result(ppl, split, assets):
    cfg = json.loads((assets / "ppl_config.json").read_text())
    return {"valid": True, "ppl": ppl, "ppl_protocol": cfg["protocol"], "ppl_split": split,
            "predicted_tokens": cfg["scored_tokens"], "model_memory": memory_report()}


def decode_result(seconds, assets):
    cfg = json.loads((assets / "decode_config.json").read_text())
    rows = []
    for r in range(cfg["rounds"]):
        for p in range(cfg["requests_per_round"]):
            arrivals = [0.05 + seconds * i / (cfg["output_tokens"] - 1) for i in range(cfg["output_tokens"])]
            rows.append({"round": r, "request": p, "input_tokens": cfg["input_tokens"],
                         "output_tokens": cfg["output_tokens"], "arrival_seconds": arrivals,
                         "chunk_sizes": [1] * cfg["output_tokens"], "chunk_arrival_seconds": arrivals,
                         "decode_seconds": arrivals[-1] - arrivals[0],
                         "decode_intervals": cfg["output_tokens"] - 1})
    return {"valid": True, "complete": True, "measurement_protocol": cfg["measurement_protocol"],
            "requests": rows, "model_memory": memory_report()}


def accuracy_result(accuracy, split, assets):
    cfg = json.loads((assets / "accuracy_config.json").read_text())
    return {"valid": True, "accuracy": accuracy, "chance": cfg["chance"],
            "accuracy_protocol": cfg["protocol"], "accuracy_split": split, "items": cfg["items"],
            "benchmarks": {b["name"]: {"items": b["items"], "correct": 0, "accuracy": accuracy,
                                       "chance": b["chance"], "kind": b["kind"]}
                           for b in cfg["benchmarks"]},
            "model_memory": memory_report()}


# 1. The reward across the PPL range on both splits; in the hidden verdict, behind the gate.
gate_cfg = json.loads((hidden_assets / "accuracy_config.json").read_text())
bf16_acc, chance = gate_cfg["bf16"]["accuracy"], gate_cfg["chance"]
lift = bf16_acc - chance
GATE = {"bf16": bf16_acc, "kept-0.9": chance + 0.9 * lift, "at-floor": chance + 0.8 * lift,
        "below-floor": chance + 0.79 * lift, "chance": chance, "above-bf16": min(1.0, bf16_acc + 0.02)}
for split, assets, public in (("hidden", hidden_assets, False), ("public", public_assets, True)):
    bf16_ppl = json.loads((assets / "ppl_config.json").read_text())["bf16_ppl"]
    for p_name, ppl in {"bf16": bf16_ppl, "x1.1": bf16_ppl * 1.1, "x2": bf16_ppl * 2,
                        "below-bf16": bf16_ppl * 0.95}.items():
        for a_name, acc in (GATE.items() if not public else [("none", None)]):
            out[f"score/{split}/{p_name}/{a_name}"] = outcome(
                decode_benchmark.final_score, ppl_result(ppl, split, assets),
                decode_result(1.0, assets), public=public, ppl_assets=assets,
                accuracy_result=None if acc is None else accuracy_result(acc, "hidden", hidden_assets),
                accuracy_assets=None if public else hidden_assets)
    out[f"inputs/{split}"] = {"bf16_ppl": bf16_ppl, "bf16_accuracy": bf16_acc, "chance": chance}

# 2. The verdict refuses missing, stale or mixed gate evidence.
assets = hidden_assets
good = lambda: (ppl_result(1.5 * 7, "hidden", assets), decode_result(1.0, assets),  # noqa: E731
                accuracy_result(bf16_acc, "hidden", assets))
for label, mutate in {
    "no-accuracy": lambda p, d, a: None,
    "invalid-accuracy": lambda p, d, a: a.update(valid=False),
    "wrong-split": lambda p, d, a: a.update(accuracy_split="public"),
    "wrong-protocol": lambda p, d, a: a.update(accuracy_protocol="other"),
    "wrong-items": lambda p, d, a: a.update(items=1),
    "no-memory": lambda p, d, a: a.pop("model_memory"),
    "accuracy-above-one": lambda p, d, a: a.update(accuracy=1.5),
}.items():
    p, d, a = good()
    mutate(p, d, a)
    out[f"refuse/{label}"] = outcome(decode_benchmark.final_score, p, d, public=False, ppl_assets=assets,
                                     accuracy_result=None if label == "no-accuracy" else a,
                                     accuracy_assets=assets)

# 3. The accuracy stage on the real inputs, through a fake runtime whose
#    generations are empty text.
class FakeRuntime:
    calls = []

    def __init__(self, submission, **options):
        FakeRuntime.options = options

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def generate(self, batch, *, max_new_tokens, vocab_size, ignore_eos=False):
        FakeRuntime.calls.append(len(batch))
        return [[1] * 3 for _ in batch]

    def memory_report(self):
        return memory_report()


FakeRuntime.calls = []
result = outcome(accuracy_benchmark.measure, "sub", hidden_assets, FakeRuntime,
                 tokenizer_dir=hidden_assets / "tokenizer", expected_split="hidden")
if "ok" in result:
    result["ok"] = {k: v for k, v in result["ok"].items() if k not in ("generations", "correct_items", "model_memory")}
out["accuracy-measure"] = result
out["accuracy-calls"] = {"count": len(FakeRuntime.calls), "max_tokens": max(FakeRuntime.calls or [0]),
                         "options": FakeRuntime.options}
out["accuracy-load/wrong-split"] = outcome(accuracy_benchmark.load_inputs, hidden_assets, expected_split="public")
out["public-has-gate-inputs"] = (public_assets / "accuracy_config.json").exists()

# 4. The public response.
if (copy_dir / "evaluate_public.py").is_file():
    import evaluate_public
    bf16_ppl = json.loads((public_assets / "ppl_config.json").read_text())["bf16_ppl"]
    stage = {"check": {"valid": True, "total_bytes": 1, "total_gib": 1e-9},
             "smoke": {"valid": True, "generated_text": "Paris"},
             "ppl": ppl_result(bf16_ppl * 1.1, "public", public_assets),
             "decode": dict(decode_result(1.0, public_assets), median_round_mean_seconds=1.0)}
    stage["score"] = decode_benchmark.final_score(stage["ppl"], decode_result(1.0, public_assets),
                                                  public=True, ppl_assets=public_assets)
    evaluate_public.shared.evaluate_submission = lambda *a, **k: stage
    profile = evaluate_public.load_profile("formal")
    response = evaluate_public.evaluate(Path("sub"), profile=profile)
    out["public"] = {k: response.get(k) for k in (
        "public_reward", "ppl_fidelity", "public_ppl", "public_bf16_ppl", "decode_latency_seconds",
        "decode_in_reward", "scoring_version", "reward_scale", "correctness_gate")}
    out["public-response-keys"] = sorted(response)
    out["public-profile-accuracy-assets"] = str(profile.get("accuracy_assets"))

print(json.dumps(out, sort_keys=True, default=lambda v: repr(v) if isinstance(v, float) and not math.isfinite(v) else str(v)))
