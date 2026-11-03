"""Render a task's `instruction.md` and `README.md` from its contract and assets.

The instruction is the task's specification and states facts only: the goal,
the hard gates, how each measurement is made, what the workspace and the
container hold, how the verdict runs, what the submission must be, and how
versions and the budget work -- the shape of ServerBench's standard task
documents, with no method advice. It ends with the session sections
(`## Continuation`, `## Idle restart`, `## Cold restart`) that the harness
sends on a later launch; they are not part of the delivered specification
(`compression_bench/harness/lifecycle/instructions.py`).

Every number is read from the task's own sources: the capacity and policy from
`contract.yaml`, the BF16 measurements from `assets/public/{ppl,decode}_config.json`,
the checkpoint's structure from `assets/model_facts.json` (`taskctl model-facts`).
Nothing is typed in by hand, so a re-measured baseline reaches the instruction
by `taskctl render` alone. The hidden correctness gate is stated to exist and
nothing more: its questions, their source, its size and its threshold appear
only in the repository-side README.

The body is `_template/instruction/instruction.md.in` followed by
`framework.md.in`, with `{{KEY}}` scalars and `@@IF FLAG@@ ... @@END@@` blocks.
A block marker on a line of its own consumes that line; a marker inside a line
is removed in place. Blocks nest.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import model_facts  # noqa: E402

REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "compression_bench" / "harness" / "final_validate"))
import task_policy  # noqa: E402

TEMPLATE = HERE.parent / "_template" / "instruction"
_TOKEN = re.compile(r"(?m)(^@@(?:IF [A-Z0-9_]+|END)@@\n|@@(?:IF [A-Z0-9_]+|END)@@)")


def _datasets() -> dict:
    """prepare_datasets.SOURCES and CHECK_SOURCES, read without its download machinery."""
    import importlib.util
    path = REPO / "compression_bench" / "scripts" / "deploy" / "prepare_datasets.py"
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location("compression_bench_prepare_datasets", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module.SOURCES, module.CHECK_SOURCES


def dataset_rows(check: bool = False) -> str:
    sources = _datasets()[1 if check else 0]
    return "\n".join(f"| `/opt/datasets/{name}/` | `{repo}` @ `{revision[:12]}` | {content} | {license_} |"
                      for name, (repo, revision, _, license_, content) in sorted(sources.items()))


def expand(text: str, flags: dict[str, bool], values: dict[str, str]) -> str:
    """Apply `@@IF FLAG@@` blocks, then `{{KEY}}` scalars; refuse leftovers."""
    out: list[str] = []
    stack: list[bool] = []
    for piece in _TOKEN.split(text):
        marker = piece.strip()
        if marker.startswith("@@IF ") and marker.endswith("@@"):
            name = marker[5:-2]
            if name not in flags:
                raise KeyError(f"instruction template uses unknown flag {name}")
            stack.append(bool(flags[name]))
            continue
        if marker == "@@END@@":
            if not stack:
                raise ValueError("unbalanced @@END@@ in instruction template")
            stack.pop()
            continue
        if all(stack):
            out.append(piece)
    if stack:
        raise ValueError("unclosed @@IF@@ in instruction template")
    result = "".join(out)
    for key, value in values.items():
        result = result.replace("{{" + key + "}}", value)
    left = re.findall(r"\{\{[A-Z0-9_]+\}\}", result)
    if left:
        raise KeyError(f"instruction template has unfilled values: {sorted(set(left))}")
    return result


def _vllm_version() -> str:
    from render import VLLM_VERSION
    return VLLM_VERSION


def _ppl(value: float) -> str:
    return f"{value:.12f}"


def facts(task_dir: Path, contract: dict) -> tuple[dict[str, bool], dict[str, str]]:
    """The flags and values the template needs, from the task's own sources."""
    evaluation = contract["evaluation"]
    public_dir, hidden_dir = task_dir / "assets" / "public", task_dir / "assets" / "hidden"
    ppl = json.loads((public_dir / "ppl_config.json").read_text())
    decode = json.loads((public_dir / "decode_config.json").read_text())
    gate = json.loads((hidden_dir / "accuracy_config.json").read_text())
    protocol = evaluation["ppl_protocol"]
    if protocol not in task_policy.LIVE_PROTOCOLS:
        raise ValueError(f"{protocol}: the instruction describes the BF16-continued mixed-text blocks only")
    prefix = task_policy.PPL_PROTOCOLS[protocol]["prefix_tokens"]
    chat = task_policy.chat_prefix_tokens(protocol)
    check_timeout = int(contract["budget"]["public_check_timeout_s"])
    if check_timeout != int(contract["budget"]["verifier_timeout_s"]) or check_timeout % 60:
        raise ValueError("the instruction states one execution timeout for public checks and the "
                         "hidden Verifier: public_check_timeout_s must equal verifier_timeout_s "
                         "and be whole minutes")
    latency = decode.get("bf16_latency_seconds")
    flags = {
        "MTP_NOTE": bool(contract["model"].get("mtp_note")),
        "CHAT_PPL": chat > 0,
        "DEBUG_MODEL": contract.get("debug_model") is not None,
        "BF16_LATENCY_KNOWN": latency is not None,
    }
    values = {
        "MODEL_TITLE": contract["model"]["title"],
        "MODEL_ID": contract["model"]["id"],
        "MODEL_REVISION": contract["model"]["revision"],
        "CAPACITY_GIB": f"{contract['capacity_gib']:g}",
        "CAPACITY_BYTES": f"{int(contract['capacity_gib'] * 2**30):,}",
        "BF16_PPL": _ppl(ppl["bf16_ppl"]),
        "BF16_LATENCY": "—" if latency is None else f"{latency:.6f}",
        "DATASET_ROWS": dataset_rows(),
        "CHECK_DATASET_ROWS": dataset_rows(check=True),
        "PREFIX_TOKENS": str(prefix),
        "CHAT_PREFIX_TOKENS": str(chat),
        "PPL_PROTOCOL": protocol,
        "SCORED_TOKENS": str(256 * (2048 - prefix)),
        "CONTINUATION_TOKENS": str(2048 - prefix),
        "SCORED_TOKENS_COMMA": f"{256 * (2048 - prefix):,}",
        "CHECK_TIMEOUT_MINUTES": f"{check_timeout // 60}",
        "VLLM_VERSION": _vllm_version(),
        "MODEL_FACTS": model_facts.markdown(json.loads((task_dir / "assets" / "model_facts.json").read_text())),
    }
    if gate.get("split") != "hidden" or (task_dir / "assets" / "public" / "accuracy_config.json").exists():
        raise ValueError("the correctness gate has hidden inputs only")
    return flags, values


def gate_facts(task_dir: Path, contract: dict) -> dict[str, str]:
    """The hidden gate's details, for the repository-side README only."""
    gate = json.loads((task_dir / "assets" / "hidden" / "accuracy_config.json").read_text())
    drop = contract["evaluation"]["quality"]["max_accuracy_drop"]
    bf16 = (gate.get("bf16") or {}).get("accuracy")
    return {"floor": f"above-chance accuracy ≥ {1 - drop:g} × BF16's",
            "questions": ", ".join(f"{b['name']} {b['items']} ({b['dataset']})" for b in gate["benchmarks"]),
            "bf16": "—" if bf16 is None else f"{bf16:.4f} (chance {gate['chance']:.4f})",
            "protocol": f"`{gate['protocol']}`"}


def render(task_dir: Path, contract: dict) -> str:
    flags, values = facts(task_dir, contract)
    body = expand((TEMPLATE / "instruction.md.in").read_text(), flags, values)
    framework = expand((TEMPLATE / "framework.md.in").read_text(), flags, values)
    return body.rstrip("\n") + "\n\n" + framework


def summary(task_dir: Path, contract: dict) -> str:
    """The repository-side README for one task."""
    _, values = facts(task_dir, contract)
    gate = gate_facts(task_dir, contract)
    evaluation = contract["evaluation"]
    debug = contract.get("debug_model")
    rows = [
        ("task", f"`{contract['task_id']}` (hash `{contract['task_hash']}`)"),
        ("model", f"`{contract['model']['id']}` @ `{contract['model']['revision'][:12]}`"),
        ("capacity", f"{values['CAPACITY_GIB']} GiB package and GPU-resident model storage"),
        ("reward", "10 × PPL_bf16 / PPL on a hidden corpus of mixed text continued by the BF16 checkpoint, no maximum"),
        ("correctness gate (hidden, verdict only)", gate["floor"]),
        ("gate questions (never shown to the agent)", gate["questions"]),
        ("treatment / hardware", f"{contract['treatment']} / {contract['hardware']}"),
        ("public BF16 PPL", values["BF16_PPL"]),
        ("hidden BF16 gate accuracy", gate["bf16"]),
        ("PPL protocol", f"`{evaluation['ppl_protocol']}`"),
        ("gate protocol", gate["protocol"]),
        ("scoring", f"`{evaluation['scoring_protocol']}`"),
        ("platform", f"`{contract['platform']['image']}`"
                     + (f" + patches {', '.join(contract['platform']['patches'])}"
                        if contract['platform']['patches'] else "")),
        ("debug model", f"`{debug['id']}`" if debug else "none"),
        ("agent budget", f"{contract['budget']['agent_hours']} h; verifier "
                         f"{contract['budget']['verifier_timeout_s']} s"),
    ]
    table = "\n".join(f"| {k} | {v} |" for k, v in rows)
    return f"""<!-- GENERATED by compression_tasks/_manage/taskctl.py render; do not edit. -->
# {contract['task_id']}

{contract['description']}

| | |
|---|---|
{table}

The agent receives [`instruction.md`](instruction.md) up to its `## Continuation`
heading; the session sections after it are what a resumed session is told.
Sources: `contract.yaml` and `assets/` (public, hidden and debug inputs, official
tokenizer). Everything else here is generated by
`taskctl render`; see [`MODE.md`](MODE.md) for where each container part comes from.

```bash
bash compression_bench/scripts/compress_evolve/evolve-task.sh {contract['task_id']} --hours {contract['budget']['agent_hours']}
```
"""
