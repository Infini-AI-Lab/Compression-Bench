"""The images every task renders FROM, and the hand-written task table that names them."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FRESHNESS = REPO / "compression_bench" / "scripts" / "agent" / "image_freshness.py"


def load_freshness(images_json: Path | None = None):
    spec = importlib.util.spec_from_file_location("image_freshness_under_test", FRESHNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if images_json is not None:
        module.IMAGES = images_json
        module.REPO = images_json.parent
    return module


def test_every_task_names_an_image_in_the_manifest():
    images = json.loads((REPO / "compression_bench" / "environments" / "images.json").read_text())
    tags = {v["tag"] for v in images.values() if isinstance(v, dict)}
    sys.path.insert(0, str(REPO / "compression_tasks" / "_manage"))
    import sites
    dockerfiles = [path for task in sites.live_tasks()
                   for path in sorted((REPO / "compression_tasks" / task).glob("**/Dockerfile"))]
    assert dockerfiles
    for dockerfile in dockerfiles:
        froms = re.findall(r"^FROM (\S+)", dockerfile.read_text(), flags=re.M)
        assert froms and all(f in tags for f in froms), (dockerfile, froms)


def test_digest_covers_content_and_the_base_image(tmp_path):
    (tmp_path / "base").mkdir()
    (tmp_path / "base" / "Dockerfile").write_text("FROM scratch\n")
    (tmp_path / "plat").mkdir()
    (tmp_path / "plat" / "Dockerfile").write_text("ARG BASE_IMAGE\nFROM ${BASE_IMAGE}\n")
    (tmp_path / "plat" / "patches").mkdir()
    (tmp_path / "plat" / "patches" / "p.py").write_text("x")
    manifest = tmp_path / "images.json"
    manifest.write_text(json.dumps({
        "base": {"tag": "t/base:v1", "context": "base"},
        "plat": {"tag": "t/plat:v1", "context": "plat", "build_args": {"BASE_IMAGE": "base"},
                 "exclude": ["patches"]},
    }))
    module = load_freshness(manifest)
    base, plat = module.digest("base"), module.digest("plat")
    (tmp_path / "plat" / "patches" / "p.py").write_text("y")       # excluded: no change
    assert module.digest("plat") == plat
    (tmp_path / "base" / "Dockerfile").write_text("FROM scratch\nRUN true\n")
    assert module.digest("base") != base
    assert module.digest("plat") != plat                             # follows its base
    (tmp_path / "base" / "__pycache__").mkdir()
    (tmp_path / "base" / "__pycache__" / "x.pyc").write_bytes(b"\0")
    assert module.digest("base") == module.digest("base")


def test_task_table_hashes_are_the_task_hashes():
    text = (REPO / "compression_tasks" / "README.md").read_text()
    rows = re.findall(r"\| `([a-z0-9_]+_standard_[0-9]+x[A-Za-z0-9-]+)` \| `([0-9a-f]{8})` \|", text)
    live = sorted(p.parent.name for p in (REPO / "compression_tasks").glob("*/contract.yaml")
                  if not p.parent.name.startswith("_"))
    assert sorted(task for task, _ in rows) == live
    for task, digest in rows:
        assert digest == hashlib.sha256(task.encode()).hexdigest()[:8], task


def test_one_unpatched_runtime_stack_in_every_container():
    """vLLM 0.32.0 as released, with the humming-kernels it pins, built once in agent-base
    and inherited, never reinstalled or patched, by the Checker and the Verifier."""
    dockerfile = (REPO / "compression_bench/environments/agent-base/Dockerfile").read_text()
    assert dockerfile.count("FROM vllm/vllm-openai:v0.32.0") == 2
    assert "assert version('vllm') == '0.32.0'; assert version('humming-kernels') == '0.1.17'" in dockerfile
    assert "pip install --no-cache-dir --no-deps humming" not in dockerfile
    images = json.loads((REPO / "compression_bench/environments/images.json").read_text())
    assert images["vllm-runtime"]["tag"] == "vllm/vllm-openai:v0.32.0"
    sys.path.insert(0, str(REPO / "compression_tasks" / "_manage"))
    import render
    import sites
    assert render.VLLM_VERSION == "0.32.0"
    for task in sites.live_tasks():
        root = REPO / "compression_tasks" / task
        contract = render.load_contract(root)
        assert render.runtime_image(contract) == render.agent_image(contract)
        for copy in ("environment/public-checker/Dockerfile", "tests/Dockerfile"):
            text = (root / copy).read_text()
            assert "assert version('vllm') == '0.32.0'" in text and "pip install" not in text, copy
        for split in ("public", "hidden"):
            assert json.loads((root / "assets" / split / "decode_config.json").read_text())["vllm_version"] == "0.32.0"


def test_the_agent_user_has_a_name_without_a_passwd_entry():
    """The agent runs as the host uid, which the image's /etc/passwd does not hold;
    getpass.getuser() then fails unless USER/LOGNAME are set, and a library that looks
    the user up at import cannot be imported."""
    import yaml
    sys.path.insert(0, str(REPO / "compression_tasks" / "_manage"))
    import sites
    for task in sites.live_tasks():
        compose = yaml.safe_load((REPO / "compression_tasks" / task / "environment" / "docker-compose.yaml").read_text())
        agent = compose["services"]["main"]
        assert str(agent["user"]).startswith("${COMPRESSION_BENCH_HOST_UID")
        assert agent["environment"]["USER"] and agent["environment"]["LOGNAME"], task


def test_the_agent_base_digest_covers_the_papers_and_skill():
    """The skill the harness delivers into the agent's home is a copy of the source
    the image bakes in, so an image built from another skill or paper set is stale."""
    from pathlib import PurePosixPath
    module = load_freshness()
    images = json.loads((REPO / "compression_bench/environments/images.json").read_text())
    patterns = module.ALWAYS_EXCLUDED + tuple(images["agent-base"].get("exclude", ()))
    for kept in ("papers/skill/SKILL.md", "papers/skill/scripts/search.py", "papers/build_collection.py",
                 "library/quantization/papers/a.txt", "library-exclude.txt", ".dockerignore", "Dockerfile"):
        assert not module._excluded(PurePosixPath(kept), patterns), kept
    for dropped in ("papers/skill/scripts/__pycache__/a.pyc",):
        assert module._excluded(PurePosixPath(dropped), patterns), dropped
    import lifecycle.workspace as workspace
    assert workspace.PAPERS_SKILL == REPO / images["agent-base"]["context"] / "papers" / "skill"
