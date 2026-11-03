"""The agent's document states facts. It does not teach, advise or instruct.

Ported from ServerBench (`serverbench/tests/contracts/
test_the_instruction_only_states_facts.py`): the document carries the tools,
the workspace, the goal, the hard gates, how the verdict runs and its
entrypoint -- and never advises the agent what to run. The same holds for every
compression task, with the reviewer's anti-hacking notice, but with no
compression technique named as hacking, since compression is the task -- and no
method named as a route to the goal: a method or toolkit the document names is
the one an agent reaches for.

Rendered for EVERY live task, so a task added later is held to the same line.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "compression_tasks" / "_manage"))
import sites  # noqa: E402

from lifecycle import instructions  # noqa: E402

LIVE = sites.live_tasks()

#: Advisory shapes, each with the kind of material it is.
FORBIDDEN = (
    (r"\byou should\b", "advice"),
    (r"\bwe recommend\b", "advice"),
    (r"\bit is worth\b", "advice"),
    (r"\byour job\b", "advice"),
    (r"\bmake sure you\b", "advice"),
    (r"\bdo not spend\b", "advice"),
    (r"\bkeep working\b", "advice"),
    (r"\byou are done\b", "advice"),
    (r"\bnext thing to do\b", "advice"),
    (r"\bthe first thing to do\b", "the prescribed order of work"),
    (r"\bRead this first\b", "the reading-list framing"),
    (r"\bStart with\b", "the prescribed order of work"),
    # Coaching a compression brief tends to carry.
    (r"Choose (?:the|your) (?:method|approach) independently", "the method coaching"),
    (r"Keep only the best submission|Preserve only the best submission", "the working-style advice"),
    (r"at most one temporary\s+candidate", "the working-style advice"),
    (r"Use the same managed runner", "the waiting coaching"),
    (r"Do not append `nohup`", "the waiting coaching"),
    (r"repeatedly call the model", "the waiting coaching"),
    (r"Release GPU memory", "the Checker coaching"),
    (r"Reserve separate texts", "the calibration coaching"),
    (r"Use the optional calibration text for", "the calibration coaching"),
    (r"Record the final export recipe", "the record-keeping coaching"),
    (r"report the concrete failure so the operator", "a channel the agent does not have"),
    (r"\bask the operator\b", "a channel the agent does not have"),
    (r"inspect them before you change anything", "the restart coaching"),
    (r"Continue the work\b", "the restart coaching"),
)

#: Compression methods and toolchains: the prose names none as a route. The
#: interpreter inventory names every installed package, alphabetically, and is
#: checked on its own below.
METHODS = (r"\bGPTQ\b", r"\bGPTAQ\b", r"LLM Compressor", r"\bllmcompressor\b", r"/opt/quantize\b",
           r"/opt/gptq\b", r"\bAutoRound\b", r"\bAWQ\b", r"\bHQQ\b", r"\bQTIP\b", r"\bSpinQuant\b",
           r"\bQuaRot\b", r"\bSparseGPT\b", r"\bWanda\b", r"\bbitsandbytes\b", r"\bMarlin\b")

#: What hacking means in a compression task. ServerBench's list opens with
#: "running any part of the model below the checkpoint's own precision"; here
#: that is the task, and must not appear.
NOT_HACKING = (r"quantis", r"quantiz", r"below the checkpoint's own precision", r"\bprun",
               r"distill", r"low-rank")

#: Facts the document must carry.
REQUIRED = (
    ("# Compress ", "the title"),
    ("A more capable model will inspect your work", "the reviewer notice"),
    ("scores **0 `reward`**", "that a hacked submission scores zero"),
    ("Use only that model", "one model for every route"),
    ("This document is the specification for this task", "what the document is"),
    ("## The goal", "the goal"),
    ("Maximise **`reward`**", "the objective"),
    ("R_ppl  = PPL_bf16 / PPL", "the perplexity term"),
    ("`reward` has no maximum", "that a perplexity below BF16's scores higher too"),
    ("reward = 10 * R_ppl", "the reward"),
    ("It is not part\nof `reward`", "that decode latency is reported, not scored"),
    ("## The hard gates", "the hard gates"),
    ("Three hard gates", "that there are three"),
    ("**1. The capacity gate.**", "the capacity gate"),
    ("**2. The correctness gate.**", "the correctness gate"),
    ("whatever its `reward`", "that the correctness gate rejects regardless of the reward"),
    ("The check runs only in the verdict", "that the gate runs only in the verdict"),
    ("not disclosed", "that the gate's details are withheld"),
    ("What you verify about the submission's correctness", "that the agent may check correctness itself"),
    ("**3. The validity gate.**", "the validity gate"),
    ("## How the measurements are made", "the measurement protocols"),
    ("### Perplexity", "the PPL protocol"),
    ("training, fine-tuning or calibrating on any evaluation input or on the test or\nvalidation split of any public benchmark",
     "that training on evaluation or benchmark test data is hacking"),
    ("/opt/datasets/", "the training data"),
    ("None of it is an evaluation input", "that the training data holds no benchmark"),
    ("/opt/compression-papers/", "the papers"),
    ("compression-papers` skill", "the skill that searches and reads them"),
    ("No compression toolkit is installed", "that the container holds no toolkit"),
    ("### Single-request decode", "the decode protocol"),
    ("## What is in your workspace", "the workspace inventory"),
    ("## What is in the container", "the container inventory"),
    ("## The checkpoint", "the checkpoint's structure"),
    ("/input/model/config.json", "where its whole configuration is"),
    ("| tensor | layers | dtype | shape |", "every stored tensor with its layers and precision"),
    ("## Measure the reward the way the verdict does", "the public Checker"),
    ("python3 /app/check_submission.py /app/submission", "the Checker command"),
    ("public_reward", "what the Checker reports"),
    ("## How the final verdict runs", "how the verdict runs"),
    ("## The submission the verdict requires", "the submission contract"),
    ("INFERENCE_API_VERSION = 1", "the API version"),
    ("register_quantization_config", "that a submission may register its own quantization format"),
    # Every agent builds its own compression; a method picked because an
    # implementation exists (vLLM's kernels, a package) is hacking.
    ("The compression is the submission's own work", "that the format and kernels are the submission's own"),
    ("a method picked for convenience is treated like any other hack", "that convenience is hacking"),
    ("never because an implementation of it already exists", "why a method is chosen"),
    # A temperature on either model is hacking; both score at 1.
    ("calibrating, changing or folding in an output temperature", "that a temperature is hacking"),
    ("scored at the same\ntemperature, 1", "that BF16 and the submission score at temperature 1"),
    ("token_logprobs(token_ids)", "the log-probability method"),
    ("stream_generate(token_ids", "the streaming method"),
    ("Speculative decoding of any kind", "the speculative-decoding rule"),
    ("## Versions", "what gets judged"),
    ("## Budget", "the budget"),
    ("`deadline_utc` in\n`/app/work/WORKSPACE.json`", "where the deadline is recorded"),
)


def render_contract(task: str) -> dict:
    import render
    return render.load_contract(sites.TASKS / task)


def _document(task: str) -> str:
    return instructions.brief(sites.TASKS / task)


def test_there_are_tasks_to_check() -> None:
    assert LIVE


@pytest.mark.parametrize("task", LIVE)
def test_the_document_advises_nothing(task: str) -> None:
    body = _document(task)
    found = [f"{pattern!r} ({what})" for pattern, what in FORBIDDEN
             if re.search(pattern, body, re.I)]
    assert not found, f"{task}'s instruction.md carries advisory material:\n  " + "\n  ".join(found)


def _prose(body: str) -> str:
    """The document without its generated interpreter inventory table."""
    head, _, rest = body.partition("Isolated interpreters")
    return head + rest.partition("Packages installed only in this container")[2]


@pytest.mark.parametrize("task", LIVE)
def test_the_document_names_no_compression_method(task: str) -> None:
    body = _prose(_document(task))
    found = [pattern for pattern in METHODS if re.search(pattern, body)]
    assert not found, f"{task}'s instruction.md names {found}"


@pytest.mark.parametrize("task", LIVE)
def test_the_anti_hacking_notice_does_not_forbid_compression(task: str) -> None:
    body = _document(task)
    notice = body.split("A more capable model will inspect your work", 1)[1].split("## The goal", 1)[0]
    found = [pattern for pattern in NOT_HACKING if re.search(pattern, notice, re.I)]
    assert not found, f"{task}'s anti-hacking notice names {found}"


@pytest.mark.parametrize("task", LIVE)
def test_the_document_states_every_required_fact(task: str) -> None:
    body = _document(task)
    missing = [f"{needle!r} ({what})" for needle, what in REQUIRED if needle not in body]
    assert not missing, f"{task}'s instruction.md does not state:\n  " + "\n  ".join(missing)


@pytest.mark.parametrize("task", LIVE)
def test_no_quantization_method_is_named_anywhere_in_the_document(task: str) -> None:
    """Not even a table row, a protocol id or an identifier names a quantization
    method or its calibration loader."""
    body = _document(task).lower()
    for name in ("gptq", "get_c4"):
        assert name not in body, name


@pytest.mark.parametrize("task", LIVE)
def test_no_toolkit_or_method_is_offered_anywhere_in_the_document(task: str) -> None:
    """No toolkit, author source, ready-made quantizer or method name anywhere,
    tables included: the container holds the runtime, data and papers."""
    body = _document(task).lower()
    for name in ("awq", "qtip", "quip", "aqlm", "hqq", "spinquant", "quarot", "smoothquant",
                 "bitsandbytes", "nf4", "marlin", "machete", "exllama", "imatrix", "llama.cpp",
                 "llama-quantize", "q4_k", "iq4", "autoround", "auto-round", "torchao", "modelopt",
                 "llmcompressor", "llm compressor", "/opt/compression-library", "upstream",
                 "group size", "group_size", "4-bit", "8-bit", "/bin/python`"):
        assert name not in body, name


@pytest.mark.parametrize("task", LIVE)
def test_the_public_baseline_is_stated(task: str) -> None:
    import json
    body = _document(task)
    ppl = json.loads((sites.TASKS / task / "assets" / "public" / "ppl_config.json").read_text())["bf16_ppl"]
    assert f"| {ppl:.12f} |" in body


#: What the agent must not learn about the correctness gate and the perplexity
#: text: their benchmarks, datasets, corpus, size, threshold or BF16 result.
GATE_SECRETS = ("mathqa", "math_qa", "math-qa", "codemmlu", "code_mmlu", "sciq", "fsoft-aic", "allenai/sci",
                "hidden-mc-gate", "multiple-choice", "multiple choice", "80%", "0.8 ", "max_accuracy_drop",
                "retention", "above-chance", "chance",
                # The gate's benchmarks (GSM8K, HumanEval, TriviaQA), which also supply the
                # multi-task perplexity contexts; the agent is told only "a hidden
                # multi-task benchmark".
                "gsm8k", "gsm-8k", "grade-school", "grade school", "humaneval", "human_eval",
                "human-eval", "triviaqa", "trivia_qa", "trivia qa", "mandarjoshi",
                "openai/gsm", "openai_humaneval", "hidden-gen-gate", "unit test", "calculator",
                "#### ", "word problem",
                # The mixed-text protocol's contexts are drawn from the Dolma 3 Dolmino mix;
                # the agent is told only "a hidden corpus of mixed text".
                "dolma", "dolmino", "allenai/dolma", "olmo")


@pytest.mark.parametrize("task", LIVE)
def test_nothing_of_the_gate_reaches_the_agent(task: str) -> None:
    import json
    root = sites.TASKS / task
    hidden = root / "assets" / "hidden"
    gate = json.loads((hidden / "accuracy_config.json").read_text())
    secrets = GATE_SECRETS + (f"{gate['bf16']['accuracy']:.4f}", f"{gate['chance']:.4f}",
                              *(b["dataset"].lower() for b in gate["benchmarks"]))
    ppl = json.loads((hidden / "ppl_config.json").read_text())["bf16_ppl"]
    agent_facing = [_document(task)] + [p.read_text(errors="ignore") for p in (root / "environment").rglob("*")
                                        if p.is_file() and p.suffix in (".py", ".md", ".json", ".yaml", ".txt")
                                        and "public-checker" not in p.parts]
    for text in agent_facing:
        low = text.lower()
        assert f"{ppl:.12f}" not in text
        leaked = [s for s in secrets if s.lower() in low]
        assert not leaked, leaked


@pytest.mark.parametrize("task", LIVE)
def test_the_session_prompts_are_the_tasks_own(task: str) -> None:
    """And they are not delivered as part of the specification."""
    body = _document(task)
    for kind, heading in instructions.SESSION_HEADINGS.items():
        assert heading not in body
        text = instructions.session_prompt(sites.TASKS / task, kind, budget_seconds=3600.0,
                                           deadline_utc="2026-10-17T00:00:00Z")
        assert "1.00 h" in text and "2026-10-17T00:00:00Z" in text
        found = [pattern for pattern, _ in FORBIDDEN if re.search(pattern, text, re.I)]
        assert not found, f"{task}/{kind}: {found}"


def test_the_harness_composes_no_agent_facing_prose() -> None:
    """The lifecycle modules that reach the agent carry no prompt of their own."""
    banned = {"compression_bench/harness/lifecycle/instructions.py": 120,
              "compression_bench/harness/lifecycle/agent.py": None,
              "compression_bench/harness/lifecycle/workspace.py": None}
    for path, line_cap in banned.items():
        body = (REPO / path).read_text("utf-8")
        if line_cap is not None:
            assert len(body.splitlines()) < line_cap, f"{path} is a loader, not a template"
        tree = ast.parse(body)
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                first = (node.body or [None])[0]
                if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                        and isinstance(first.value.value, str)):
                    docstrings.add(id(first.value))
        literals = [node.value for node in ast.walk(tree)
                    if isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and id(node) not in docstrings]
        for text in literals:
            for pattern, what in FORBIDDEN:
                assert not re.search(pattern, text, re.I), f"{path} carries {what}: {text[:90]!r}"
