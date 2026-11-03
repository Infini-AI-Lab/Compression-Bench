import json
import os
import subprocess
import sys
from types import SimpleNamespace

from lifecycle import instructions, workspace


def test_layout_creates_the_interfaces_directories(tmp_path):
    paths = workspace.layout(tmp_path)
    for key in ("work", "agent_home", "tmp", "versions", "submission", "verifier"):
        assert paths[key].is_dir()
    assert paths["tmp"] == tmp_path / "work" / ".tmp"   # the compose file's TMPDIR
    assert paths["lock"].is_file()
    assert "debug_submission" not in paths
    debug = workspace.layout(tmp_path / "d", debug_model=True)
    assert debug["debug_submission"] == tmp_path / "d" / "work" / "debug-submission"
    assert debug["debug_submission"].is_dir()


def test_deliver_puts_the_brief_workspace_json_and_tools_in_work(tmp_path, make_task):
    task = make_task()
    workspace.layout(tmp_path)
    workspace.deliver(tmp_path, task_dir=task, task_id="compress-test-4g",
                      deadline_utc="2030-01-01T00:00:00Z", budget_hours=2.0,
                      run_id="r", lineage_id="l")
    work = tmp_path / "work"
    assert (work / "AGENT_TASK.md").read_text() == instructions.brief(task)
    assert "## Continuation" not in (work / "AGENT_TASK.md").read_text()
    data = json.loads((work / "WORKSPACE.json").read_text())
    assert data["deadline_utc"] == "2030-01-01T00:00:00Z" and data["budget_hours"] == 2.0
    for tool in ("version", "clock"):
        assert os.access(work / tool, os.X_OK)
        compile((work / tool).read_text(), tool, "exec")


def test_deliver_puts_the_papers_skill_in_the_clis_home(tmp_path, make_task):
    """Claude Code finds a user skill at ~/.claude/skills/<name>/SKILL.md; $HOME is
    work/.agent-home. The copy is the image's own skill source, byte for byte."""
    task = make_task()
    workspace.layout(tmp_path)
    for _ in range(2):                                   # a re-delivery replaces the copy
        delivered = workspace.deliver(tmp_path, task_dir=task, task_id="compress-test-4g",
                                      deadline_utc="2030-01-01T00:00:00Z", budget_hours=2.0,
                                      run_id="r", lineage_id="l")
    assert delivered["skills"] == ["compression-papers"]
    skill = tmp_path / "work" / ".agent-home" / ".claude" / "skills" / "compression-papers"
    source = workspace.PAPERS_SKILL
    files = sorted(p.relative_to(source).as_posix() for p in source.rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts)
    assert files and "SKILL.md" in files and "scripts/search.py" in files and "scripts/read.py" in files
    assert sorted(p.relative_to(skill).as_posix() for p in skill.rglob("*") if p.is_file()) == files
    for name in files:
        assert (skill / name).read_bytes() == (source / name).read_bytes(), name
    assert source.parent.parent.name == "agent-base"     # inside the agent image's build context


def test_version_tool_writes_a_publish_request(tmp_path):
    """Run the tool's body against a temporary /app/work by rewriting its root."""
    body = workspace.VERSION_TOOL.replace("/app/work/.versions", str(tmp_path / ".versions"))
    body = body.replace("range(120)", "range(1)").replace("time.sleep(0.5)", "pass")
    script = tmp_path / "version"
    script.write_text(body)
    got = subprocess.run([sys.executable, str(script), "quantized", "to", "fp8"],
                         capture_output=True, text=True, check=False)
    assert got.returncode == 0
    requests = list((tmp_path / ".versions").glob("*.json"))
    assert len(requests) == 1
    data = json.loads(requests[0].read_text())
    assert data == {**data, "action": "publish", "note": "quantized to fp8"}


def test_version_tool_without_a_note_prints_help(tmp_path):
    script = tmp_path / "version"
    script.write_text(workspace.VERSION_TOOL)
    got = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert got.returncode == 2 and "version" in got.stdout


def fake_run(replies):
    """A `backend.run` stand-in answering by the first word that matches."""
    def run(container, command, timeout=0):
        for needle, (code, out) in replies.items():
            if needle in command:
                return SimpleNamespace(returncode=code, stdout=out, stderr="")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")
    return run


GOOD = {
    "nvidia-smi": (0, "GPU 0: NVIDIA H200 NVL (UUID: GPU-x)\n"),
    "--version": (0, "2.1.296 (Claude Code)"),
    "check_submission": (1, json.dumps({"valid": False, "error": "missing submission/model/"})),
    "urllib": (0, json.dumps({"https://api.anthropic.com/": "reachable (HTTP 404)",
                              **{u: "blocked (URLError)" for u in workspace.BLOCKED_PROBES}})),
    "compression-papers": (0, "1079 papers\n"),
}


def test_boot_contract_passes_on_a_good_environment():
    checks = workspace.boot_contract("c", agent_cli="claude",
                                     provider_url="https://api.anthropic.com/",
                                     run=fake_run(GOOD))
    ok, summary = workspace.summary_of(checks)
    assert ok, summary
    assert {c["check"] for c in checks} >= {"model readable", "one GPU visible",
                                            "public Checker answers", "egress allowlist",
                                            "paper collection searchable"}


def test_boot_contract_fails_on_two_gpus_open_egress_or_a_silent_checker():
    for needle, reply in (
            ("nvidia-smi", (0, "GPU 0: a\nGPU 1: b\n")),
            ("urllib", (0, json.dumps({"https://api.anthropic.com/": "reachable",
                                       **{u: "reachable" for u in workspace.BLOCKED_PROBES}}))),
            ("check_submission", (1, "Traceback: connection refused")),
            ("compression-papers", (1, "test: SKILL.md: No such file"))):
        checks = workspace.boot_contract("c", agent_cli="claude",
                                         provider_url="https://api.anthropic.com/",
                                         run=fake_run({**GOOD, needle: reply}))
        ok, summary = workspace.summary_of(checks)
        assert not ok, needle
