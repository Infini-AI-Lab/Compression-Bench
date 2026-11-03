import importlib.util
import json
import subprocess
import sys

import pytest

from lifecycle import run
from lifecycle.utils import task_hash

from conftest import REPO


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_task_dir_by_path_id_and_hash(tmp_path, make_task):
    task = make_task(root=tmp_path / "compression_tasks")
    assert run.resolve_task_dir(str(task)) == task.resolve()
    assert run.resolve_task_dir("COMPRESS-TEST-4G", repo=tmp_path) == task.resolve()
    assert run.resolve_task_dir(task_hash("compress-test-4g"), repo=tmp_path) == task.resolve()
    with pytest.raises(SystemExit):
        run.resolve_task_dir("nope", repo=tmp_path)


def test_model_identity_refuses_incomplete_or_wrong_checkpoints(tmp_path):
    contract = {"model": {"id": "Org/M", "revision": "abc"}}
    model = tmp_path / "m"
    model.mkdir()
    with pytest.raises(run.SetupFailed):
        run.model_identity(model, contract)              # no config.json
    (model / "config.json").write_text("{}")
    with pytest.raises(run.SetupFailed, match="no weight files"):
        run.model_identity(model, contract)
    (model / "model.safetensors").write_bytes(b"x")
    found = run.model_identity(model, contract)
    assert found["weight_files"] == 1
    snap = tmp_path / "models--Org--M" / "snapshots" / "other"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "w.safetensors").write_bytes(b"x")
    with pytest.raises(run.SetupFailed, match="snapshot revision"):
        run.model_identity(snap, contract)


def test_the_parser_requires_task_gpu_and_model():
    parser = run.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--task", "t"])
    args = parser.parse_args(["--task", "t", "--gpu", "0", "--model-dir", "/m"])
    assert args.agent_cli == "claude" and not args.dry_run


def test_model_for_task_prefers_the_models_root_then_the_hf_cache(tmp_path, make_task, monkeypatch):
    module = load(REPO / "compression_bench/scripts/model_for_task.py", "model_for_task")
    task = make_task()
    root = tmp_path / "models"
    (root / "Model-1B-bf16").mkdir(parents=True)
    (root / "Model-1B-bf16" / "config.json").write_text("{}")
    (root / "Model-1B-bf16" / "model.safetensors").write_bytes(b"x")
    assert module.resolve(task, env={"COMPRESSION_BENCH_MODELS_ROOT": str(root)}) == root / "Model-1B-bf16"
    cache = tmp_path / "hub"
    snap = cache / "models--Org--Model-1B" / "snapshots" / "0123456789abcdef0123456789abcdef01234567"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    # a snapshot that fetched only the config is not a checkpoint (run.model_identity refuses it)
    assert module.resolve(task, env={}) is None
    (snap / "model-00001-of-00001.safetensors").symlink_to(tmp_path / "missing-blob")
    assert module.resolve(task, env={}) is None
    (tmp_path / "missing-blob").write_bytes(b"x")
    assert module.resolve(task, env={}) == snap
    assert module.resolve(task, env={}, debug=True) is None


def test_validate_evolve_reads_the_verifiers_outputs(tmp_path):
    module = load(REPO / "compression_bench/harness/final_validate/tools/validate_evolve.py",
                  "validate_evolve")
    verifier = tmp_path / "v"
    verifier.mkdir()
    assert module.summarize_verifier(verifier)["status"] == "NOT_RUN"
    (verifier / "reward.txt").write_text("0\n")
    (verifier / "check.json").write_text(json.dumps({"valid": False, "error": "missing submission/model/"}))
    zero = module.summarize_verifier(verifier)
    assert zero["status"] == "ZERO" and zero["failed_stage"] == "check"
    assert zero["error"] == "missing submission/model/"
    (verifier / "reward.txt").write_text("0.0755\n")
    (verifier / "check.json").write_text(json.dumps({"valid": True, "total_gib": 17.34}))
    (verifier / "score.json").write_text(json.dumps({
        "valid": True, "ppl": 10.23, "decode_latency_seconds": 1.197, "reward": 0.0755,
        "quality_pass": True, "model_memory": {"max_resident_bytes": 2 ** 34}}))
    scored = module.summarize_verifier(verifier)
    assert scored["status"] == "SCORED" and scored["package_gib"] == 17.34
    assert scored["resident_gib"] == 16.0 and scored["failed_stage"] is None


def test_validate_evolve_lock_check(tmp_path):
    import fcntl
    module = load(REPO / "compression_bench/harness/final_validate/tools/validate_evolve.py",
                  "validate_evolve2")
    lock = tmp_path / ".submission.lock"
    lock.touch()
    assert module.lock_is_free(lock)
    with lock.open("rb") as held:
        fcntl.flock(held, fcntl.LOCK_SH)
        assert not module.lock_is_free(lock)


def test_launcher_scripts_parse():
    for script in ("compression_bench/scripts/compress_evolve/evolve-task.sh", "compression_bench/scripts/env.sh"):
        assert subprocess.run(["bash", "-n", str(REPO / script)]).returncode == 0
    for script in ("compression_bench/scripts/evolve", "compression_bench/scripts/agent/harbor_env.py"):
        compile((REPO / script).read_text(), script, "exec")


def test_the_core_imports_without_yaml_at_module_scope():
    """The lifecycle is stdlib-only at import time (yaml inside functions)."""
    code = ("import sys, builtins; real = builtins.__import__\n"
            "def guard(name, *a, **k):\n"
            "    if name.split('.')[0] in ('yaml', 'harbor', 'torch'): raise ImportError(name)\n"
            "    return real(name, *a, **k)\n"
            "builtins.__import__ = guard\n"
            "import lifecycle.run, lifecycle.agent, lifecycle.boundary, lifecycle.gpus\n")
    got = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={"PYTHONPATH": str(REPO / "compression_bench" / "harness")})
    assert got.returncode == 0, got.stderr


def test_tokenizer_packaging_runs_the_tasks_own_helper(tmp_path):
    """The post-verdict step that adds the official tokenizer to the final submission."""
    module = load(REPO / "compression_bench/harness/final_validate/tools/validate_evolve.py",
                  "validate_evolve_packaging")
    task = REPO / "compression_tasks" / "qwen3_8_27b_12g_standard_1xH200NVL"
    submission = tmp_path / "submission"
    (submission / "model").mkdir(parents=True)
    (submission / "model" / "config.json").write_text("{}")
    got = module.package_tokenizer(task, submission)
    assert got["status"] == "packaged", got["output"]
    assert (submission / "model" / "tokenizer.json").exists()


def test_the_checkpoint_verdict_timeout_is_one_bound_per_checkpoint_and_fits_subprocess(tmp_path):
    """MAX_VERSIONS * 3 h (5.4e6 s) would overflow poll()'s int milliseconds and fail the run
    (HARNESS) right after its final verdict; the bound is capped."""
    from lifecycle import record
    driver = object.__new__(run.Lifecycle)
    driver.contract = {"budget": {"verifier_timeout_s": 3600}}
    driver.journal = record.Journal(tmp_path)
    for version in ("v1", "v2"):
        driver.journal.append("version_captured", version=version)
    driver.journal.append("final_captured", version="v3")
    assert driver.checkpoint_verdicts_timeout() == 3 * (3 * 3600 + 3600)
    for index in range(1000):
        driver.journal.append("version_captured", version=f"x{index}")
    longest = driver.checkpoint_verdicts_timeout()
    assert longest == run.POLL_TIMEOUT_MAX_S
    # The real wait path: capture_output waits through a selector's poll().
    subprocess.run([sys.executable, "-c", "pass"], capture_output=True, text=True,
                   timeout=longest, check=True)


def test_a_print_to_a_vanished_terminal_never_raises():
    """A driver started from a terminal that later went away must not record
    `harvest_failed: OSError [Errno 5]` (or a BrokenPipeError) over a print."""
    class Gone:
        def write(self, text):
            raise OSError(5, "Input/output error")

        def flush(self):
            raise BrokenPipeError(32, "Broken pipe")

    out = run.Unbreakable(Gone())
    assert out.write("metrics: ...\n") == len("metrics: ...\n")
    out.flush()


def test_ctrl_c_cancels_a_run_like_sigterm():
    """A Ctrl-C cancels a run; it is not a FAILED (HARNESS, KeyboardInterrupt) run."""
    import signal
    assert set(run.CANCEL_SIGNALS) == {signal.SIGTERM, signal.SIGHUP, signal.SIGINT}
    assert issubclass(run.Cancelled, BaseException) and not issubclass(run.Cancelled, Exception)
