"""Every checkpoint gets the full verdict, recorded in detail; a full disk is not a verdict."""
import importlib.util
import json
import math
from pathlib import Path

from lifecycle import agent, disk, verdicts

REPO = Path(__file__).resolve().parents[3]
ENOSPC = "OSError: [Errno 28] No space left on device: '/tmp/quant-inference-work/vllm'"


def load_validate_evolve():
    path = REPO / "compression_bench" / "harness" / "final_validate" / "tools" / "validate_evolve.py"
    spec = importlib.util.spec_from_file_location("validate_evolve_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verifier_folder(root: Path, *, error: str | None = None) -> Path:
    """A Verifier output folder in the shapes the real evaluator writes."""
    root.mkdir(parents=True)
    census = {"max_resident_bytes": 900, "limit_gib": 8, "passed": True, "snapshots": [
        {"phase": "after load", "resident_bytes": 900, "passed": True, "workers": [{
            "bytes": 900, "storage_count": 3, "storages": [
                {"bytes": 500, "dtype": "torch.uint8",
                 "first_path": "model._modules.layers._modules.0._modules.mlp._parameters.qweight"},
                {"bytes": 300, "dtype": "torch.uint8",
                 "first_path": "model._modules.layers._modules.1._modules.mlp._parameters.qweight"},
                {"bytes": 100, "dtype": "torch.bfloat16",
                 "first_path": "model._modules.embed_tokens._parameters.weight"}]}]}]}
    files = {
        "check": {"valid": True, "total_gib": 7.5, "total_bytes": 8, "file_count": 5},
        "smoke": {"valid": error is None, "generated_text": "Paris.", **({"error": error} if error else {})},
        "ppl": {"valid": True, "ppl": 10.5, "nll": 2.0, "windows": [
            {"block": 0, "nll": 2047 * math.log(10.0), "scored_tokens": 2047},
            {"block": 1, "nll": 2047 * math.log(11.0), "scored_tokens": 2047}]},
        "accuracy": {"valid": True, "accuracy": 0.7, "chance": 0.23, "items": 150,
                     "benchmarks": {"math": {"items": 50, "correct": 30}},
                     "predictions": {"math": [1, 2, 0]}},
        "decode": {"valid": True, "decode_tokens_per_second": 40.0, "model_memory": census,
                   "requests": [{"round": 0, "request": 0, "output_tokens": 128, "decode_seconds": 3.2,
                                 "arrival_seconds": list(range(128))}]},
        "score": {"valid": error is None, "reward": 0.0 if error else 9.4, "ppl": 10.5,
                  "accuracy_gate_pass": True, "model_memory": census,
                  **({"error": error} if error else {})},
    }
    for name, body in files.items():
        (root / f"{name}.json").write_text(json.dumps(body))
    (root / "timing.json").write_text(json.dumps({"stages": [{"stage": "ppl", "seconds": 456.0}]}))
    (root / "reward.txt").write_text("0.0" if error else "9.4")
    return root


def test_a_full_disk_is_not_a_verdict(tmp_path):
    validate = load_validate_evolve()
    failed = validate.summarize_verifier(verifier_folder(tmp_path / "a", error=ENOSPC))
    assert failed["status"] == "NOT_RUN" and failed["infra_error"] is True
    assert not verdicts.measured(failed)
    scored = validate.summarize_verifier(verifier_folder(tmp_path / "b"))
    assert scored["status"] == "SCORED" and verdicts.measured(scored)
    # A pointer written before the rule existed is judged by its error text too.
    assert not verdicts.measured({"status": "ZERO", "error": ENOSPC})
    assert verdicts.measured({"status": "ZERO", "error": "ValueError: wrong shape"})


def test_the_detail_carries_what_the_scalars_drop(tmp_path):
    detail = verdicts.detail(verifier_folder(tmp_path / "v"))
    assert [round(x, 6) for x in detail["ppl"]["block_ppl"]] == [10.0, 11.0]
    assert detail["ppl"]["block_ppl_stats"]["n"] == 2
    assert detail["accuracy"]["benchmarks"]["math"]["correct"] == 30
    assert detail["accuracy"]["predictions"]["math"] == [1, 2, 0]
    assert detail["decode"]["requests"][0]["decode_seconds"] == 3.2
    assert "arrival_seconds" not in detail["decode"]["requests"][0]
    assert detail["census"]["largest_groups"][0] == {"module": "layers.N.mlp.qweight",
                                                     "dtype": "torch.uint8", "bytes": 800, "storages": 2}
    assert detail["stage_seconds"] == {"ppl": 456.0}
    assert detail["smoke"]["generated_text"] == "Paris."


def test_content_identity_ignores_caches_and_tokenizer_assets():
    base = [{"path": "inference.py", "type": "file", "size": 1, "sha256": "a"},
            {"path": "model/w.safetensors", "type": "file", "size": 9, "sha256": "w"}]
    noisy = base + [{"path": "__pycache__/x.pyc", "type": "file", "size": 2, "sha256": "p"},
                    {"path": "model/tokenizer.json", "type": "file", "size": 3, "sha256": "t"}]
    assert verdicts.content_identity(base) == verdicts.content_identity(noisy)
    changed = [dict(base[0]), dict(base[1], sha256="v")]
    assert verdicts.content_identity(changed) != verdicts.content_identity(base)


def test_resumability_is_read_from_the_transcript_not_asked_of_the_model(monkeypatch):
    """A probe turn would run without the provider route's settings, fail `Not logged in`
    and turn every resume of an API-key run into a cold restart; the transcript is read instead."""
    commands = []

    class Done:
        returncode, stdout = 0, "/app/work/.agent-home/.claude/projects/-app-work/s1.jsonl\n"
    monkeypatch.setattr(agent.backend, "run", lambda container, command, timeout: commands.append(command) or Done())
    assert agent.can_continue(container="c", session_id="s1", agent_cli="claude")
    assert "find" in commands[0] and "s1.jsonl" in commands[0] and "claude --resume" not in commands[0]


def test_the_disk_check_reports_floors(tmp_path):
    space = disk.check(tmp_path, docker_floor=0, run_floor=0)
    assert space["ok"] and space["run_free_gib"] > 0
    assert not disk.check(tmp_path, docker_floor=0, run_floor=10 ** 9)["ok"]


def test_a_busy_card_at_engine_start_is_the_hosts_fault_not_a_zero():
    error = ("submitted inference failed: ValueError: Free memory on device cuda:0 (86.17/139.79 GiB) on "
             "startup is less than desired GPU memory utilization (0.8, 111.83 GiB).")
    assert verdicts.is_infra_error(error)
    assert not verdicts.measured({"status": "ZERO", "reward": 0.0, "error": error})
