"""What the agent's container carries: the task, the packages and the related files.

The container holds the task folder, the packages and the related files, and no
instructions teaching the agent to use a certain method (a guide that names one
method the default steers every agent to it); training data the agent may use,
but not the benchmark; and the accuracy benchmark and the verdict are read-only
to it.

No third-party source code (no `**/upstream/` author snapshot), no isolated
toolkit interpreter (`/opt/<toolkit>/` venvs) and no llama.cpp; of the library
only the papers are shipped, at `/opt/compression-papers/` with the
`compression-papers` skill (test_papers_skill.py). The base runtime is vLLM as
released, its GGUF plugin and `gguf`.
"""
from __future__ import annotations

import fnmatch
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
AGENT_BASE = REPO / "compression_bench" / "environments" / "agent-base"
LIBRARY = AGENT_BASE / "library"
DEPLOY = REPO / "compression_bench" / "scripts" / "deploy"
sys.path.insert(0, str(REPO / "compression_tasks" / "_manage"))
import sites  # noqa: E402


def load(name: str, path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_library_holds_no_task_authored_guide_or_tool():
    """Tracked files are only provenance manifests; the rest of the library (the papers,
    their text extracts) is third-party, listed with its sources by the reference lock and
    downloaded by prepare_references.py."""
    tracked = subprocess.run(["git", "-C", str(REPO), "ls-files", str(LIBRARY)],
                             capture_output=True, text=True, check=True).stdout.split()
    names = [Path(p).relative_to(LIBRARY.relative_to(REPO)).as_posix() for p in tracked]
    assert names
    for name in names:
        assert name.endswith(".json"), name


def test_benchmark_files_by_name_are_listed_for_removal():
    guard = load("benchmark_guard", DEPLOY / "benchmark_guard.py")
    listed = set(guard.read_exclude())
    if not any(LIBRARY.glob("*/papers/*.txt")):
        pytest.skip("library papers not restored on this host (prepare_references.py)")
    for path in LIBRARY.rglob("*"):
        relative = path.relative_to(LIBRARY)
        if path.is_file() and any(fnmatch.fnmatch(path.name.lower(), p) for p in guard.NAME_PATTERNS):
            assert relative.as_posix() in listed, path
    for name in listed:
        assert (LIBRARY / name).is_file(), f"listed but absent: {name}"


def test_the_image_takes_the_papers_only_from_their_cleaned_stage():
    dockerfile = (AGENT_BASE / "Dockerfile").read_text()
    _, stage, final = re.split(r"^FROM ", dockerfile, flags=re.M)
    assert "AS papers" in stage and "source=library-exclude.txt" in stage and "build_collection.py" in stage
    assert "--exclude /tmp/library-exclude.txt" in stage
    assert "COPY --from=papers /opt/compression-papers/ /opt/compression-papers/" in final
    assert "COPY library/" not in final, "the final stage copies the uncleaned library"
    assert "compression-library" not in stage
    assert "test ! -e /opt/compression-library" in final
    assert "quant-docs" not in dockerfile


def test_the_image_carries_no_author_source_toolkit_or_quantizer():
    """No author source code, toolkit interpreter, Git-installed package or llama.cpp
    is built into it."""
    dockerfile = "\n".join(line for line in (AGENT_BASE / "Dockerfile").read_text().splitlines()
                           if not line.lstrip().startswith("#"))
    for gone in ("upstream", "venv", "/opt/llmcompressor", "llama.cpp", "llama-quantize", "llama-imatrix",
                 "install.py", "tools.json", "requirements.txt", "git+https", "cmake"):
        assert gone not in dockerfile, gone
    installs = re.findall(r"pip install[^\n]*", dockerfile)
    assert installs == ["pip install --no-cache-dir --no-deps pyarrow==25.0.1 \\"], installs
    assert not (AGENT_BASE / "setup").exists() and not (AGENT_BASE / "requirements.txt").exists()
    prepare = load("prepare_references", DEPLOY / "prepare_references.py")
    assert all(p["pdf"].endswith(".pdf") for p in prepare.load_lock(prepare.DEFAULT_LOCK))
    platform = (REPO / "compression_bench" / "environments" / "platform-gguf" / "Dockerfile").read_text()
    assert "llama" not in platform.lower() and "quantize" not in platform.lower()


def test_the_training_data_is_not_an_evaluation_benchmark():
    datasets = load("prepare_datasets", DEPLOY / "prepare_datasets.py")
    for name, (repo, *_rest) in datasets.SOURCES.items():
        assert "gsm8k" not in repo.lower() and "c4" not in repo.lower().split("/")[-1], name
    # Check data (MATH-500) are test splits, never training data,
    # and none of them is an input of a live task's evaluation: the gate's benchmarks
    # (GSM8K, HumanEval, TriviaQA) differ from what the agent is given to test on.
    assert not set(datasets.SOURCES) & set(datasets.CHECK_SOURCES)
    gate = {b["dataset"].lower() for task in sites.live_tasks() for b in json.loads(
        (REPO / "compression_tasks" / task / "assets" / "hidden" / "accuracy_config.json").read_text())["benchmarks"]}
    for name, (repo, *_rest) in datasets.CHECK_SOURCES.items():
        assert repo.lower() not in gate and "c4" not in repo.lower().split("/")[-1], name
        for word in ("gsm8k", "humaneval", "openai_humaneval", "trivia", "mathqa", "math_qa", "codemmlu", "sciq"):
            assert word not in repo.lower(), name
    staged = AGENT_BASE / "datasets" / "DATASETS.json"
    if staged.is_file():
        manifest = json.loads(staged.read_text())
        assert set(manifest["datasets"]) == set(datasets.SOURCES) | set(datasets.CHECK_SOURCES)
        assert all(manifest["datasets"][n]["use"] == "check" for n in datasets.CHECK_SOURCES)
        assert all(manifest["datasets"][n]["use"] == "train" for n in datasets.SOURCES)
        assert "hidden evaluation inputs" in manifest["filter"]
        for name in ("gsm8k", "humaneval", "triviaqa", "trivia_qa", "mathqa", "math_qa", "codemmlu", "sciq", "arc", "hellaswag"):
            assert name not in manifest["filter"].lower(), name


@pytest.mark.parametrize("task", sites.live_tasks())
def test_the_agent_container_carries_no_evaluator_and_no_benchmark(task):
    root = REPO / "compression_tasks" / task
    environment = (root / "environment" / "Dockerfile").read_text()
    assert "evaluator" not in environment and "calibration" not in environment
    assert not (root / "environment" / "calibration").exists()
    for copy in ("environment/public-checker/Dockerfile", "tests/Dockerfile"):
        assert "chmod -R a-w" in (root / copy).read_text(), copy
