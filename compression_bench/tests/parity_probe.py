"""Exercise one evaluator copy on synthetic inputs and print what it does, as JSON.

    python3 parity_probe.py <evaluator dir> <public assets dir> <hidden assets dir> <scratch dir>

Run against each evaluator copy of a task (the Verifier's `tests/` and the
public Checker's); `test_evaluator_parity.py` compares the outputs. The probe
imports only the copy under test (its directory is
the first entry on sys.path) and needs no GPU, model, torch or vLLM: the
measurement entry points are driven through fake runtimes, and a stub
`transformers` module stands in for the tokenizer.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import types
from pathlib import Path

copy_dir, public_assets, hidden_assets, scratch = (Path(a).resolve() for a in sys.argv[1:5])
sys.path.insert(0, str(copy_dir))
os.chdir(copy_dir)

out: dict[str, object] = {}


def outcome(fn, *args, **kwargs):
    try:
        return {"ok": normalise(fn(*args, **kwargs))}
    except Exception as exc:  # noqa: BLE001 - the error text is the observable
        return {"error": f"{type(exc).__name__}: {exc}"}


def normalise(value):
    if isinstance(value, dict):
        return {str(k): normalise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalise(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value


# A stub tokenizer: plain prompts and chat prompts tokenize differently, so the
# smoke stage's choice between them is observable.
class FakeTokenizer:
    calls: list = []

    def __init__(self, vocab):
        self.vocab = vocab

    def __len__(self):
        return self.vocab

    def __call__(self, text, add_special_tokens=False):
        FakeTokenizer.calls.append(["plain", text, add_special_tokens])
        return types.SimpleNamespace(input_ids=[1, 2, 3])

    def apply_chat_template(self, messages, **kwargs):
        FakeTokenizer.calls.append(["chat", messages, sorted(kwargs.items())])
        return [7, 8, 9, 10]

    def decode(self, ids, skip_special_tokens=True):
        return "Paris"


transformers = types.ModuleType("transformers")
transformers.AutoTokenizer = types.SimpleNamespace(
    from_pretrained=lambda *a, **k: FakeTokenizer(300000))
sys.modules["transformers"] = transformers

import decode_benchmark  # noqa: E402
import evaluate  # noqa: E402
import inference_runtime  # noqa: E402
import model_memory  # noqa: E402
import ppl_benchmark  # noqa: E402
import tokenizer_assets  # noqa: E402

policy = model_memory.read_policy(copy_dir)
limit_bytes = int(policy["limit_gib"] * 2**30)


def memory_report(resident=1000):
    worker = {"bytes": resident, "storage_count": 1,
              "storages": [{"bytes": resident, "device": "cuda:0", "dtype": "torch.uint8",
                            "first_path": "model.w"}]}
    snapshots = [model_memory.check_snapshot({"protocol": model_memory.PROTOCOL, "workers": [worker]},
                                             policy, phase)
                 for phase in ("load", "token_logprobs")]
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
            d = seconds * (1 + 0.01 * r + 0.001 * p)
            arrivals = [0.05 + d * i / (cfg["output_tokens"] - 1) for i in range(cfg["output_tokens"])]
            rows.append({"round": r, "request": p, "input_tokens": cfg["input_tokens"],
                         "output_tokens": cfg["output_tokens"], "arrival_seconds": arrivals,
                         "chunk_sizes": [1] * cfg["output_tokens"], "chunk_arrival_seconds": arrivals,
                         "decode_seconds": arrivals[-1] - arrivals[0],
                         "decode_intervals": cfg["output_tokens"] - 1})
    return {"valid": True, "complete": True, "measurement_protocol": cfg["measurement_protocol"],
            "requests": rows, "model_memory": memory_report()}


# 1. The final score on both splits, across a range of PPL and decode latency.
for split, assets, public in (("hidden", hidden_assets, False), ("public", public_assets, True)):
    bf16 = json.loads((assets / "ppl_config.json").read_text())["bf16_ppl"]
    for name, ppl in {"bf16": bf16, "x1.05": bf16 * 1.05, "x1.12": bf16 * 1.12, "x1.15": bf16 * 1.15,
                      "x1.2": bf16 * 1.2, "x1.25": bf16 * 1.25, "x1.3": bf16 * 1.3, "x1.31": bf16 * 1.31,
                      "below-one": 0.5}.items():
        for seconds in (0.6, 1.4):
            out[f"score/{split}/{name}/{seconds}"] = outcome(
                decode_benchmark.final_score, ppl_result(ppl, split, assets),
                decode_result(seconds, assets), public=public, ppl_assets=assets)

# 2. Scoring refuses inconsistent or stale inputs.
assets = hidden_assets
good_ppl = ppl_result(json.loads((assets / "ppl_config.json").read_text())["bf16_ppl"], "hidden", assets)
for label, mutate in {
    "wrong-split": lambda p, d: p.update(ppl_split="public"),
    "wrong-tokens": lambda p, d: p.update(predicted_tokens=1),
    "invalid-ppl": lambda p, d: p.update(valid=False),
    "decode-incomplete": lambda p, d: d.update(complete=False),
    "decode-protocol": lambda p, d: d.update(measurement_protocol="other"),
    "missing-memory": lambda p, d: p.pop("model_memory"),
    "two-token-chunk": lambda p, d: d["requests"][0].update(chunk_sizes=[2] + [1] * 126),
}.items():
    p, d = json.loads(json.dumps(good_ppl)), decode_result(1.0, assets)
    mutate(p, d)
    out[f"refuse/{label}"] = outcome(decode_benchmark.final_score, p, d, public=False, ppl_assets=assets)

# 3. Fixed perplexity inputs: loading, the scored-token arithmetic, and PPL from fake probabilities.
class FakeRuntime:
    def __init__(self, submission, **options):
        self.options = options

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def token_logprobs(self, batch):
        return [[None] + [-0.5 - 0.001 * ((i + j) % 7) for j in range(1, len(ids))]
                for i, ids in enumerate(batch)]

    def memory_report(self):
        return memory_report()


for split, assets in (("hidden", hidden_assets), ("public", public_assets)):
    def load(split=split, assets=assets):
        cfg, rows = ppl_benchmark.load_inputs(assets, assets / "tokenizer", expected_split=split)
        return {"blocks": len(rows), "first": rows[0][:24], "protocol": cfg["protocol"]}
    out[f"ppl-load/{split}"] = outcome(load)
    out[f"ppl-load/{split}/wrong-split"] = outcome(ppl_benchmark.load_inputs, assets, assets,
                                                    expected_split="other")
    out[f"ppl-measure/{split}"] = outcome(lambda split=split, assets=assets: {
        k: v for k, v in ppl_benchmark.measure("sub", assets, assets, FakeRuntime,
                                               expected_split=split).items() if k != "windows"})
debug = public_assets / "debug-9b"
if debug.is_dir() and (copy_dir / "evaluate_public.py").is_file():  # a Checker-only profile
    # measure_quality always uses the real runtime class; drive the debug split
    # through ppl_benchmark directly.
    def debug_measure():
        result = ppl_benchmark.measure("sub", debug, debug, FakeRuntime, expected_split="debug-9b")
        return {k: v for k, v in result.items() if k != "windows"}
    out["ppl-measure/debug-9b"] = outcome(debug_measure)

# 4. Package inspection and the private evaluation view.
tokenizer_dir = hidden_assets / "tokenizer"


def make_submission(name, files, *, link=None):
    root = scratch / name
    if root.exists():
        shutil.rmtree(root)
    (root / "model").mkdir(parents=True)
    (root / "inference.py").write_text("INFERENCE_API_VERSION = 1\n")
    for rel, size in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            handle.truncate(size)
    if link:
        (root / "model" / "link.bin").symlink_to(root / "inference.py")
    return root


cases = {
    "safetensors": {"model/config.json": 10, "model/model.safetensors": 4096},
    "gguf": {"model/config.json": 10, "model/model.gguf": 4096},
    "custom-bin": {"model/config.json": 10, "model/weights.bin": 4096},
    "nested-payload": {"model/config.json": 10, "model/parts/a.npy": 4096},
    "metadata-only": {"model/config.json": 10, "model/tokenizer.json": 10},
    "no-config": {"model/model.safetensors": 4096},
    "with-tokenizer": {"model/config.json": 10, "model/model.safetensors": 4096,
                       "model/tokenizer.json": 999999, "model/chat_template.jinja": 50,
                       "model/vocab.json": 77, "helpers/kernel.py": 300},
    "too-large": {"model/config.json": 10, "model/model.safetensors": limit_bytes + 1},
    "at-limit-minus-tokenizer": {"model/config.json": 10, "model/model.safetensors":
                                 limit_bytes - 10 - 64 - 2 * 2**20},
}
for name, files in cases.items():
    root = make_submission(name, files)
    out[f"inspect/{name}"] = outcome(evaluate.inspect_submission, root, policy["limit_gib"],
                                     tokenizer_dir=tokenizer_dir)

    def view(root=root):
        with tokenizer_assets.prepared_submission(root, tokenizer_dir) as package:
            return sorted([p.relative_to(package).as_posix(), "link" if p.is_symlink() else "copy"]
                          for p in package.rglob("*") if p.is_file() or p.is_symlink())
    out[f"view/{name}"] = outcome(view)
root = make_submission("symlink", cases["safetensors"], link=True)
out["inspect/symlink"] = outcome(evaluate.inspect_submission, root, policy["limit_gib"],
                                 tokenizer_dir=tokenizer_dir)
out["default-limit"] = evaluate.DEFAULT_LIMIT_GIB
out["scoring-version"] = evaluate.SCORING_PROTOCOL

# 5. Hidden-PPL reporting fields and the setup gate in front of the pipeline.
real_measure = evaluate.measure_ppl
evaluate.measure_ppl = lambda *a, **k: {"valid": True, "ppl": 9.5, "ppl_split": k.get("expected_split")}
out["measure-quality/hidden"] = outcome(evaluate.measure_quality, "sub", tokenizer_dir, hidden_assets,
                                        split="hidden")
evaluate.measure_ppl = real_measure
gate_dir = scratch / "gate-assets"
for label, edit in {"as-shipped": lambda c: None,
                    "bf16-unmeasured": lambda c: c.update(bf16_ppl=None)}.items():
    if gate_dir.exists():
        shutil.rmtree(gate_dir)
    shutil.copytree(hidden_assets, gate_dir)
    cfg = json.loads((gate_dir / "ppl_config.json").read_text())
    edit(cfg)
    (gate_dir / "ppl_config.json").write_text(json.dumps(cfg))
    profile = dict(tokenizer=tokenizer_dir, size_limit_gib=policy["limit_gib"], score=True,
                   ppl_assets=gate_dir, ppl_split="hidden", decode_assets=gate_dir)
    broken = make_submission("gate-broken", {"model/model.safetensors": 10})
    out[f"setup-gate/{label}"] = outcome(evaluate.evaluate_submission, broken, profile)

# 6. The smoke stage's prompt form (plain text vs chat template).
class SmokeRuntime(FakeRuntime):
    def generate(self, token_ids, *, max_new_tokens, vocab_size, ignore_eos=False):
        return [[42] * 3 for _ in token_ids]


real_runtime = inference_runtime.SubmittedRuntime
inference_runtime.SubmittedRuntime = SmokeRuntime
FakeTokenizer.calls.clear()
out["smoke"] = outcome(inference_runtime.generate_smoke, "sub", tokenizer_dir,
                       "Complete with one city name: The capital of France is", 16, 60)
out["smoke-calls"] = normalise(FakeTokenizer.calls)
inference_runtime.SubmittedRuntime = real_runtime

# 7. The public Checker's response shape, from fixed stage results.
if (copy_dir / "evaluate_public.py").is_file():
    import evaluate_public
    measured = {
        "check": evaluate.inspect_submission(make_submission("public", cases["safetensors"]),
                                             policy["limit_gib"], tokenizer_dir=tokenizer_dir),
        "smoke": {"valid": True, "generated_text": "Paris"},
    }
    cfg = json.loads((public_assets / "ppl_config.json").read_text())
    for name, ppl in {"bf16": cfg["bf16_ppl"], "x1.15": cfg["bf16_ppl"] * 1.15,
                      "x1.5": cfg["bf16_ppl"] * 1.5}.items():
        stage = dict(measured, ppl=ppl_result(ppl, "public", public_assets),
                     decode=dict(decode_result(1.0, public_assets),
                                 median_round_mean_seconds=1.0))
        stage["score"] = decode_benchmark.final_score(stage["ppl"], decode_result(1.0, public_assets),
                                                      public=True, ppl_assets=public_assets)
        evaluate_public.shared.evaluate_submission = lambda *a, stage=stage, **k: stage
        profile = evaluate_public.load_profile("formal")
        out[f"public/{name}"] = outcome(evaluate_public.evaluate, Path("sub"), profile=profile)

print(json.dumps(out, sort_keys=True, default=str))
