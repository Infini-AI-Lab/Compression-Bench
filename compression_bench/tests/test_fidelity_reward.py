"""The reward and the correctness gate (fidelity.py).

    reward = 10 * PPL_bf16 / PPL on BF16-continued hidden text (~ 10 exp(-KL))   public Checker and verdict
             0 in the verdict when (accuracy - chance) < (1 - max_accuracy_drop) * (accuracy_bf16 - chance)

The gate's questions exist only in the hidden Verifier's copy, are never run by
the public Checker, and never reach the agent's instruction. The pure functions
are tested on their own; every live task's rendered evaluator copies (hidden
Verifier and public Checker) are then driven through `fidelity_probe.py` and
held to the same formula, computed here from the task's own assets.
"""
from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
FINAL_VALIDATE = REPO / "compression_bench" / "harness" / "final_validate"
TASKS = REPO / "compression_tasks"
PROBE = Path(__file__).with_name("fidelity_probe.py")
sys.path.insert(0, str(TASKS / "_manage"))
import render  # noqa: E402
import sites  # noqa: E402

LIVE = sites.live_tasks()
GATE = dict(bf16_accuracy=0.62, chance=0.233, max_accuracy_drop=0.2)


def load(name):
    spec = importlib.util.spec_from_file_location(f"fv_{name}", FINAL_VALIDATE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fidelity = load("fidelity")
accuracy_benchmark = load("accuracy_benchmark")


def expected(ppl, bf16_ppl, accuracy=None, bf16_accuracy=None, chance=None, drop=None):
    if accuracy is not None and (accuracy - chance) / (bf16_accuracy - chance) < (1 - drop) - 1e-12:
        return 0.0
    return 10 * bf16_ppl / ppl


def test_a_submission_that_measures_like_bf16_scores_ten():
    got = fidelity.score(9.9, bf16_ppl=9.9, gate=dict(GATE, accuracy=0.62))
    assert got["reward"] == 10.0 and got["ppl_fidelity"] == 1.0 and got["accuracy_gate_pass"] is True


def test_the_reward_is_the_perplexity_ratio_so_model_scale_drops_out():
    assert fidelity.score(9.9 * 1.25, bf16_ppl=9.9)["reward"] == pytest.approx(
        fidelity.score(20.3 * 1.25, bf16_ppl=20.3)["reward"]) == pytest.approx(8.0)
    assert fidelity.ppl_term(12.0, 10.0) == pytest.approx(math.exp(-(math.log(12.0) - math.log(10.0))))
    assert fidelity.ppl_term(9.0, 9.9) == pytest.approx(1.1)  # beating BF16 scores above 10


def test_the_reward_is_uncapped_and_scored_on_bf16_continuations():
    """The blocks are text continued by the BF16 checkpoint itself, where excess
    cross-entropy is KL from BF16, so changing the model (a temperature, training toward
    other text) cannot beat BF16 in expectation. Not capped: a value above 10 is sampling
    noise, not a better model."""
    assert fidelity.PROTOCOL == "bf16-sample-ppl-gated-v5"
    better = fidelity.score(9.0, bf16_ppl=10.0, gate=dict(GATE, accuracy=0.62))
    assert better["reward"] == pytest.approx(10 / 0.9) and better["ppl_fidelity"] == pytest.approx(10 / 9)
    assert "min(1" not in fidelity.FORMULA and "BF16-sampled continuations" in fidelity.FORMULA


def test_on_bf16_samples_no_model_beats_bf16_in_expectation():
    """Gibbs' inequality, the reason the reward cannot be gamed by changing the distribution:
    E_P[-log Q] >= E_P[-log P], with equality only at Q = P (a temperature here)."""
    p = [0.6, 0.25, 0.1, 0.05]
    def ce(q):
        return -sum(pi * math.log(qi) for pi, qi in zip(p, q))
    for t in (0.7, 0.9, 1.1, 1.36, 1.5):
        q = [pi ** (1 / t) for pi in p]
        z = sum(q)
        assert ce([qi / z for qi in q]) > ce(p)


def test_accuracy_is_not_part_of_the_reward():
    keep = fidelity.score(11.0, bf16_ppl=10.0, gate=dict(GATE, accuracy=0.62))
    lose = fidelity.score(11.0, bf16_ppl=10.0, gate=dict(GATE, accuracy=0.233 + 0.81 * (0.62 - 0.233)))
    assert keep["reward"] == lose["reward"] == pytest.approx(10 / 1.1)


def test_the_gate_rejects_below_the_floor_whatever_the_reward():
    floor = 0.233 + 0.8 * (0.62 - 0.233)
    at = fidelity.score(10.0, bf16_ppl=10.0, gate=dict(GATE, accuracy=floor))
    assert at["accuracy_gate_pass"] is True and at["reward"] == 10.0
    below = fidelity.score(10.0, bf16_ppl=10.0, gate=dict(GATE, accuracy=floor - 0.001))
    assert below["accuracy_gate_pass"] is False and below["reward"] == 0.0
    assert below["quality_rejection_reason"] == "accuracy_below_80_percent_of_bf16"


def test_without_a_gate_nothing_about_it_is_reported():
    got = fidelity.score(11.0, bf16_ppl=10.0)
    assert got["correctness_gate"] == "not_run"
    assert not {k for k in got if "accuracy" in k}


@pytest.mark.parametrize("call", [
    lambda: fidelity.ppl_term(float("nan"), 10.0),
    lambda: fidelity.ppl_term(0.5, 10.0),
    lambda: fidelity.accuracy_retention(0.5, 0.2, 0.233),      # BF16 at or below chance
    lambda: fidelity.accuracy_retention(1.2, 0.62, 0.233),
    lambda: fidelity.reward(-0.1),                                    # a ratio is never negative
    lambda: fidelity.score(10, bf16_ppl=10, gate=dict(GATE, accuracy=0.5, max_accuracy_drop=1.5)),
])
def test_invalid_measurements_are_refused(call):
    with pytest.raises(ValueError):
        call()


def test_generated_answers_are_judged_by_kind():
    """The gate generates its answers -- a final number, a short answer, code run
    against unit tests -- rather than choosing among options."""
    judge = accuracy_benchmark.judge
    assert judge("final-number", {"number": "18"}, " 16 - 3 - 4 = 9.\n9 * 2 = $18\n#### 18\n\nQuestion: x 7")
    assert judge("final-number", {"number": "1200"}, " so 1,200 dollars.\n\nQuestion: 5")   # last number
    assert not judge("final-number", {"number": "18"}, " #### 17\n\nQuestion: #### 18")
    assert judge("short-answer", {"aliases": ["eiffel tower"]}, " The Eiffel Tower.\nQuestion: next")
    assert not judge("short-answer", {"aliases": ["paris"]}, " Lyon\nParis")
    item = {"prompt": "def add(a, b):\n", "test": "def check(f):\n    assert f(2, 3) == 5\n",
            "entry_point": "add"}
    assert accuracy_benchmark.completion_code("    return a + b\n\ndef extra():\n    pass") == "    return a + b\n"
    assert accuracy_benchmark.passes(item, "    return a + b\n")
    assert not accuracy_benchmark.passes(item, "    return a - b\n")
    assert not accuracy_benchmark.passes(item, "    while True:\n        pass\n")
    assert not accuracy_benchmark.passes(item, "    return len(bytearray(8 * 2**30))\n")
    assert not accuracy_benchmark.passes(item, "    return a - b\nimport os\nos._exit(0)\n")


def test_the_gate_scores_the_mean_of_its_benchmarks_with_chance_zero():
    cfg = {"benchmarks": [{"name": "math", "kind": "final-number", "items": 2},
                          {"name": "knowledge", "kind": "short-answer", "items": 1}]}
    items = [{"b": 0, "answer": {"number": "4"}}, {"b": 0, "answer": {"number": "5"}},
             {"b": 1, "answer": {"aliases": ["oslo"]}}]
    got = accuracy_benchmark.score(cfg, items, {0: " 2+2=4\n#### 4", 1: " #### 6", 2: " Oslo"})
    assert got["benchmarks"]["math"]["correct"] == 1 and got["benchmarks"]["knowledge"]["correct"] == 1
    assert got["accuracy"] == pytest.approx((0.5 + 1.0) / 2) and got["chance"] == 0.0


@pytest.mark.parametrize("task", LIVE)
def test_the_gate_inputs_are_hidden_only(task):
    root = TASKS / task
    cfg, items = accuracy_benchmark.load_inputs(root / "assets" / "hidden", expected_split="hidden")
    assert [b["name"] for b in cfg["benchmarks"]] == ["code", "knowledge", "math"]
    assert [b["kind"] for b in cfg["benchmarks"]] == ["unit-tests", "short-answer", "final-number"]
    assert [b["items"] for b in cfg["benchmarks"]] == [50, 50, 50] and len(items) == 150
    assert not (root / "assets" / "public" / "accuracy_config.json").exists()
    assert not list((root / "environment").rglob("accuracy_inputs.json"))
    assert (root / "tests" / "evaluator-assets" / "accuracy_inputs.json").is_file()


def test_a_tampered_input_is_refused(tmp_path):
    source = TASKS / LIVE[0] / "assets" / "hidden"
    for name in ("accuracy_config.json", "accuracy_inputs.json"):
        (tmp_path / name).write_bytes((source / name).read_bytes())
    data = (tmp_path / "accuracy_inputs.json").read_bytes()
    (tmp_path / "accuracy_inputs.json").write_bytes(data.replace(b'"b":0', b'"b":1', 1))
    with pytest.raises(ValueError, match="digest"):
        accuracy_benchmark.load_inputs(tmp_path, expected_split="hidden")


def probe(copy_dir: Path, public: Path, hidden: Path) -> dict:
    done = subprocess.run([sys.executable, "-B", str(PROBE), str(copy_dir), str(public), str(hidden)],
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-4000:]
    return json.loads(done.stdout)


@pytest.fixture(scope="module", params=[(t, side) for t in LIVE for side in ("hidden", "public")],
                ids=lambda p: f"{p[0]}-{p[1]}")
def probed(request):
    task, side = request.param
    root = TASKS / task
    copy = root / "tests" if side == "hidden" else root / "environment" / "public-checker"
    public = root / "environment" / "public-checker" / "evaluator-assets"
    hidden = root / "tests" / "evaluator-assets"
    return task, side, probe(copy, public, hidden)


def test_every_copy_scores_the_formula(probed):
    task, side, out = probed
    drop = render.load_contract(TASKS / task)["evaluation"]["quality"]["max_accuracy_drop"]
    scored = {"hidden": 0, "public": 0}
    for key, value in out.items():
        if not key.startswith("score/"):
            continue
        _, split, _, _ = key.split("/")
        inputs = out[f"inputs/{split}"]
        assert "ok" in value, (key, value)
        got = value["ok"]
        if split == "hidden":
            want = expected(got["ppl"], inputs["bf16_ppl"], got["accuracy"], inputs["bf16_accuracy"],
                            inputs["chance"], drop)
        else:
            want = expected(got["ppl"], inputs["bf16_ppl"])
            assert got["correctness_gate"] == "not_run" and "accuracy" not in got
        assert got["reward"] == pytest.approx(want, abs=1e-9), key
        assert got["decode_in_reward"] is False and got["scoring_version"] == fidelity.PROTOCOL
        scored[split] += 1
    assert scored == {"hidden": 24, "public": 4}
    assert out["score/hidden/bf16/bf16"]["ok"]["reward"] == 10.0
    assert out["score/hidden/below-bf16/above-bf16"]["ok"]["reward"] == pytest.approx(10 / 0.95)
    assert out["score/hidden/x2/kept-0.9"]["ok"]["reward"] == pytest.approx(5.0)
    assert out["score/hidden/bf16/at-floor"]["ok"]["accuracy_gate_pass"] is True
    assert out["score/hidden/bf16/below-floor"]["ok"]["reward"] == 0.0
    assert out["score/hidden/bf16/chance"]["ok"]["reward"] == 0.0


def test_the_verdict_refuses_bad_gate_evidence(probed):
    _, _, out = probed
    refused = [k for k in out if k.startswith("refuse/")]
    assert len(refused) == 7
    for key in refused:
        assert "error" in out[key], key


def test_the_gate_is_measured_in_bounded_calls(probed):
    _, _, out = probed
    measured = out["accuracy-measure"]["ok"]
    assert measured["items"] == 150 and measured["accuracy_split"] == "hidden"
    calls = out["accuracy-calls"]
    assert calls["count"] == measured["calls"] <= 6
    assert calls["max_tokens"] <= 50     # sequences in one generate call: one benchmark's items
    assert calls["options"] == {"max_model_len": 2049, "max_num_seqs": 32,
                                "gpu_memory_utilization": 0.9, "timeout": 600}
    assert "error" in out["accuracy-load/wrong-split"]
    assert out["public-has-gate-inputs"] is False


def test_the_checker_reports_the_public_reward_and_nothing_of_the_gate(probed):
    task, side, out = probed
    if side != "public":
        pytest.skip("the public response exists only in the Checker copy")
    public, inputs = out["public"], out["inputs/public"]
    assert public["public_reward"] == pytest.approx(expected(public["public_ppl"], inputs["bf16_ppl"]))
    assert public["decode_in_reward"] is False and public["reward_scale"] == 10.0
    assert public["correctness_gate"] == "not_run"
    assert not [k for k in out["public-response-keys"] if "accuracy" in k or "gate" in k.replace("correctness_gate", "")]
    assert out["public-profile-accuracy-assets"] == "None"


def test_the_public_response_gives_a_perplexity_and_a_reward_not_the_benchmark():
    """Mid-run, the agent learns its perplexity and reward only -- no per-block losses,
    protocol id, split, token counts or reward-formula text."""
    source = (REPO / "compression_bench" / "harness" / "final_validate" / "evaluate_public.py").read_text()
    # The bf16-fidelity branch is the live one; the branch after it serves the other quality rules.
    source = source.split('elif profile.get("score"):')[0]
    for leaked in ('quality=quality', '"ppl_protocol"', '"ppl_split"', '"predicted_tokens"', '"reward_mapping"'):
        assert leaked not in source, leaked
    assert 'public_ppl=quality["ppl"]' in source and 'public_reward=score["reward"]' in source
