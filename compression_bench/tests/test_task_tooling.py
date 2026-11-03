"""The task tooling: the live tasks render from their sources, and taskctl keeps them in step."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "compression_tasks"
MANAGE = TASKS / "_manage"
TASKCTL = MANAGE / "taskctl.py"
sys.path.insert(0, str(MANAGE))

import instruction_render  # noqa: E402
import render  # noqa: E402
import sites  # noqa: E402

LIVE = sites.live_tasks()


def taskctl(*args, root=None):
    env = dict(os.environ)
    if root is not None:
        env["COMPRESSION_BENCH_TASK_ROOT"] = str(root)
    return subprocess.run([sys.executable, "-B", str(TASKCTL), *args], capture_output=True,
                          text=True, env=env, timeout=300)


@pytest.mark.parametrize("task", LIVE)
def test_task_ids_are_named_as_serverbench_names_them(task):
    """<model slug>_<capacity>g_<treatment>_<N>x<GPU>, every part from the contract."""
    import taskctl
    contract = render.load_contract(TASKS / task)
    assert taskctl.TASK_ID.fullmatch(task)
    assert task == taskctl.task_id_for(contract)
    assert contract["treatment"] == "standard" and contract["hardware"] == "1xH200NVL"


@pytest.mark.parametrize("task", LIVE)
def test_every_live_task_scores_fidelity_to_bf16(task):
    contract = render.load_contract(TASKS / task)
    assert contract["evaluation"]["quality_rule"] == "bf16-fidelity"
    # Fidelity to BF16, on hidden text continued by the BF16 checkpoint itself.
    assert contract["evaluation"]["scoring_protocol"] == "bf16-sample-ppl-gated-v5"
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tooling_task_policy", REPO / "compression_bench" / "harness" / "final_validate" / "task_policy.py")
    task_policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(task_policy)
    assert contract["evaluation"]["ppl_protocol"] in task_policy.LIVE_PROTOCOLS
    assert contract["evaluation"]["quality"] == {"max_accuracy_drop": 0.2}
    assert contract["platform"]["image"] == "gguf"
    cfg = json.loads((TASKS / task / "tests/evaluator-assets/accuracy_config.json").read_text())
    assert cfg["split"] == "hidden" and cfg["baseline_status"] == "measured"
    # The correctness gate is the Verifier's alone: the Checker holds no question of it.
    assert not (TASKS / task / "environment/public-checker/evaluator-assets/accuracy_config.json").exists()


@pytest.mark.parametrize("task", LIVE)
def test_generated_files_are_in_step(task):
    assert render.drift(TASKS / task) == []


@pytest.mark.parametrize("task", LIVE)
def test_hidden_inputs_never_reach_the_agent_or_the_checker(task):
    root = TASKS / task
    hidden = {p.read_bytes() for p in (root / "assets" / "hidden").glob("*_inputs.json")}
    for path in (root / "environment").rglob("*"):
        if path.is_file() and path.suffix == ".json":
            assert path.read_bytes() not in hidden, path


@pytest.mark.parametrize("task", LIVE)
def test_evaluator_copies_are_the_one_source(task):
    source = REPO / "compression_bench" / "harness" / "final_validate"
    for copy in (TASKS / task / "tests", TASKS / task / "environment" / "public-checker"):
        for module in render.VERIFIER_MODULES:
            assert (copy / module).read_bytes() == (source / module).read_bytes()


def test_template_engine_blocks_nest_and_refuse_leftovers():
    text = "a\n@@IF X@@\nb @@IF Y@@c@@END@@\n@@END@@\nd {{V}}\n"
    assert instruction_render.expand(text, {"X": True, "Y": False}, {"V": "1"}) == "a\nb \nd 1\n"
    assert instruction_render.expand(text, {"X": False, "Y": True}, {"V": "1"}) == "a\nd 1\n"
    with pytest.raises(KeyError):
        instruction_render.expand("{{MISSING}}", {}, {})
    with pytest.raises(KeyError):
        instruction_render.expand("@@IF NOPE@@x@@END@@", {}, {})
    with pytest.raises(ValueError):
        instruction_render.expand("@@IF X@@x", {"X": True}, {})


def test_check_passes_on_the_live_tasks():
    done = taskctl("check")
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.fixture
def scratch_root(tmp_path):
    """Two tasks' sources in a disposable root; the large measured assets are
    linked, not copied (taskctl only reads them)."""
    for task in ("qwen3_8_27b_12g_standard_1xH200NVL", "gemma4_26b_a4b_it_16g_standard_1xH200NVL"):
        (tmp_path / task).mkdir()
        shutil.copy2(TASKS / task / "contract.yaml", tmp_path / task / "contract.yaml")
        (tmp_path / task / "assets").symlink_to(TASKS / task / "assets", target_is_directory=True)
        render.write(tmp_path / task)
    return tmp_path


def test_check_refuses_a_hand_edit(scratch_root):
    path = scratch_root / "qwen3_8_27b_12g_standard_1xH200NVL" / "tests" / "evaluate.py"
    path.write_text(path.read_text() + "\n# hand edit\n")
    done = taskctl("check", "qwen3_8_27b_12g_standard_1xH200NVL", root=scratch_root)
    assert done.returncode == 1 and "stale generated file tests/evaluate.py" in done.stdout


def test_check_refuses_an_extra_file_in_a_generated_directory(scratch_root):
    (scratch_root / "qwen3_8_27b_12g_standard_1xH200NVL" / "environment" / "leftover.py").write_text("x")
    done = taskctl("check", "qwen3_8_27b_12g_standard_1xH200NVL", root=scratch_root)
    assert done.returncode == 1 and "extra file" in done.stdout


def test_set_refuses_a_loosening_without_a_reason(scratch_root):
    done = taskctl("set", "qwen3_8_27b_12g_standard_1xH200NVL", "--capacity-gib", "40", root=scratch_root)
    assert done.returncode != 0 and "loosens the task" in (done.stdout + done.stderr)
    assert "capacity_gib: 12\n" in (scratch_root / "qwen3_8_27b_12g_standard_1xH200NVL" / "contract.yaml").read_text()


def test_accepting_another_payload_format_needs_a_reason(scratch_root):
    task = scratch_root / "qwen3_8_27b_12g_standard_1xH200NVL"
    done = taskctl("set", task.name, "--payload-rule", "any-payload", root=scratch_root)
    assert done.returncode != 0 and "loosens the task" in (done.stdout + done.stderr)
    done = taskctl("set", task.name, "--payload-rule", "any-payload", "--link-rule",
                   "model-payload", "--reason", "test", root=scratch_root)
    assert done.returncode == 0, done.stdout + done.stderr
    for copy in ("tests", "environment/public-checker"):
        policy = json.loads((task / copy / "evaluator-assets" / "task_policy.json").read_text())
        assert (policy["payload_rule"], policy["link_rule"]) == ("any-payload", "model-payload")


def test_set_tightens_and_rerenders_everything(scratch_root):
    done = taskctl("set", "8a661afb", "--agent-hours", "2", root=scratch_root)
    assert done.returncode == 0, done.stdout + done.stderr
    task = scratch_root / "qwen3_8_27b_12g_standard_1xH200NVL"
    text = (task / "instruction.md").read_text()
    assert text.startswith("# Compress Qwen3.8-27B to 12 GiB")
    assert "**12 GiB** (12,884,901,888 bytes" in text
    for copy in ("tests", "environment/public-checker"):
        policy = json.loads((task / copy / "evaluator-assets" / "task_policy.json").read_text())
        memory = json.loads((task / copy / "evaluator-assets" / "model_memory_policy.json").read_text())
        assert policy["size_limit_gib"] == memory["limit_gib"] == 12
    assert "timeout_sec = 7200.0" in (task / "task.toml").read_text()


def test_new_drops_the_source_tiers_diagnostic_reference(scratch_root):
    done = taskctl("new", "qwen3_8_27b_20g_standard_1xH200NVL", "--from", "qwen3_8_27b_12g_standard_1xH200NVL",
                   "--capacity-gib", "20", root=scratch_root)
    assert done.returncode == 0, done.stdout + done.stderr
    task = scratch_root / "qwen3_8_27b_20g_standard_1xH200NVL"
    assert (task / "instruction.md").read_text().startswith("# Compress Qwen3.8-27B to 20 GiB")
    assert render.load_contract(task)["reference"]["label"] is None


def test_new_refuses_an_id_that_disagrees_with_its_contract(scratch_root):
    for wrong in ("compress-qwen3.8-27b-20g", "qwen3_8_27b_16g_standard_1xH200NVL"):
        done = taskctl("new", wrong, "--from", "qwen3_8_27b_12g_standard_1xH200NVL",
                       "--capacity-gib", "20", root=scratch_root)
        assert done.returncode != 0, wrong


def test_rename_rehashes_and_rerenders(scratch_root):
    new = "gemma4_26b_a4b_it_16g_standard_2xH200NVL"
    done = taskctl("rename", "gemma4_26b_a4b_it_16g_standard_1xH200NVL", "--to", new, root=scratch_root)
    assert done.returncode == 0, done.stdout + done.stderr
    contract = render.load_contract(scratch_root / new)
    assert contract["task_id"] == new and contract["hardware"] == "2xH200NVL"
    import hashlib
    assert contract["task_hash"] == hashlib.sha256(new.encode()).hexdigest()[:8]


def test_every_evaluator_module_an_image_imports_is_copied_into_it():
    """A module the evaluator imports but an image does not carry breaks that image's
    boot, so every sibling module an evaluator copy imports is COPYed with it."""
    import ast
    import re
    final_validate = REPO / "compression_bench" / "harness" / "final_validate"
    siblings = {p.stem for p in final_validate.glob("*.py")}
    for task in sorted(p for p in (REPO / "compression_tasks").iterdir()
                       if (p / "contract.yaml").exists() and not p.name.startswith("_")):
        for dockerfile in (task / "environment" / "public-checker" / "Dockerfile", task / "tests" / "Dockerfile"):
            copied = {Path(name).stem for line in dockerfile.read_text().splitlines()
                      if line.startswith("COPY ") for name in line.split()[1:-1] if name.endswith(".py")}
            for name in sorted(copied & siblings):
                tree = ast.parse((final_validate / f"{name}.py").read_text())
                imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
                imported |= {n.module.split(".")[0] for n in ast.walk(tree)
                             if isinstance(n, ast.ImportFrom) and n.module and not n.level}
                imported |= set(re.findall(r'with_name\("(\w+)\.py"\)', (final_validate / f"{name}.py").read_text()))
                missing = (imported & siblings) - copied
                assert not missing, f"{dockerfile.relative_to(REPO)}: {name}.py imports {sorted(missing)}, not copied"


def test_the_agent_containers_tmp_is_its_runs_own_folder():
    """/tmp is the run's work/.tmp, on the run disk, as $TMPDIR is: an agent writing to
    /tmp must not fill the Docker root every container on the host shares."""
    import yaml
    for task in LIVE:
        compose = yaml.safe_load((TASKS / task / "environment" / "docker-compose.yaml").read_text())
        main = compose["services"]["main"]
        binds = {v["target"]: v["source"] for v in main["volumes"] if isinstance(v, dict) and v.get("type") == "bind"}
        assert binds.get("/tmp", "").endswith("}/work/.tmp"), task
        assert main["environment"]["TMPDIR"] == "/app/work/.tmp", task
