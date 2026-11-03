#!/usr/bin/env python3
"""The only way to create or change a compression-bench task.

    taskctl.py show   <task>                     # the task as the tooling reads it
    taskctl.py check  [task]                     # sources valid, generated files in step
    taskctl.py render [task]                     # regenerate every generated file
    taskctl.py set    <task> --capacity-gib N | --agent-hours H | ... [--reason WHY]
    taskctl.py new    <task> --from <task> --capacity-gib N [...]
    taskctl.py rename <task> --to <new id>
    taskctl.py model-facts <task> [--model-dir DIR]   # freeze the checkpoint's structure into assets/

A task's sources are `contract.yaml` and `assets/`; `render`
writes `task.toml`, `instruction.md`, `README.md`, `MODE.md`, `environment/`
and `tests/` from them, and `check` refuses any task whose generated files
drift. `<task>` is a task id or its 8-hex hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import blocks  # noqa: E402
import model_facts  # noqa: E402
import render  # noqa: E402
import sites  # noqa: E402

FINAL_VALIDATE = sites.REPO / "compression_bench" / "harness" / "final_validate"
REQUIRED_KEYS = ("kind", "schema", "task_id", "task_hash", "version", "description", "keywords",
                 "model", "capacity_gib", "treatment", "hardware", "platform", "starter", "evaluation",
                 "reference", "debug_model", "budget", "metadata")
#: A task id, as ServerBench names its tasks: model, capacity tier, treatment
#: and hardware (`qwen3_8_27b_8g_standard_1xH200NVL`).
TASK_ID = re.compile(r"(?P<slug>[a-z0-9]+(?:_[a-z0-9]+)*?)_(?P<capacity>\d+(?:\.\d+)?)g_"
                     r"(?P<treatment>standard)_(?P<hardware>\d+x[A-Za-z0-9-]+)")
#: `standard`: the specification and nothing else.
TREATMENTS = ("standard",)


def task_id_for(contract: dict) -> str:
    capacity = sites.formatted(contract["capacity_gib"], float)
    return f"{contract['model']['slug']}_{capacity}g_{contract['treatment']}_{contract['hardware']}"


EVALUATOR_DATA = ("ppl_config.json", "ppl_inputs.json", "decode_config.json", "decode_inputs.json")
MEASUREMENT_PROTOCOL = "submitted-decode-single-token-3x3-v3"
#: The reward a bf16-fidelity task scores with, and its hidden correctness gate.
SCORING_PROTOCOL = "bf16-sample-ppl-gated-v5"
GATE_PROTOCOL = "hidden-gen-gate-v2"


def _policy_module():
    """final_validate/task_policy.py, imported without reading any policy file."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("compression_bench_task_policy", FINAL_VALIDATE / "task_policy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def task_hash(task_id: str) -> str:
    return hashlib.sha256(task_id.encode()).hexdigest()[:8]


def source_problems(task_dir: Path) -> list[str]:
    """Everything wrong with a task's sources, before anything is rendered."""
    problems: list[str] = []
    try:
        contract = render.load_contract(task_dir)
    except Exception as exc:  # noqa: BLE001
        return [f"contract.yaml does not parse: {exc}"]
    missing = [k for k in REQUIRED_KEYS if k not in contract]
    if missing:
        return [f"contract.yaml lacks {missing}"]
    if contract["kind"] != "compression_task_contract" or contract["schema"] != "compression_bench.task_contract/1":
        problems.append("contract.yaml has the wrong kind/schema")
    if contract["task_id"] != task_dir.name:
        problems.append(f"task_id {contract['task_id']!r} does not name its directory {task_dir.name!r}")
    if contract["treatment"] not in TREATMENTS:
        problems.append(f"treatment must be one of {TREATMENTS}")
    if not contract["model"].get("slug") or contract["task_id"] != task_id_for(contract):
        problems.append(f"task_id must be <model.slug>_<capacity>g_<treatment>_<hardware>: "
                        f"{task_id_for(contract) if contract['model'].get('slug') else '?'}")
    if str(contract["task_hash"]) != task_hash(task_dir.name):
        problems.append(f"task_hash must be {task_hash(task_dir.name)}")
    try:
        _policy_module().validate(render.task_policy(contract))
    except (ValueError, KeyError) as exc:
        problems.append(f"evaluation policy: {exc}")
    if contract["evaluation"]["scoring_protocol"] != contract["metadata"].get(
            "scoring_protocol", contract["evaluation"]["scoring_protocol"]):
        problems.append("metadata repeats a different scoring_protocol")
    if contract["platform"]["image"] != "gguf":
        problems.append("platform.image must be gguf")
    for patch in contract["platform"]["patches"]:
        if patch not in render.PLATFORM_PATCHES:
            problems.append(f"unknown platform patch {patch!r}")
    if "platform-gguf" not in render.images():
        problems.append("compression_bench/environments/images.json lacks platform-gguf")
    if not (render.TEMPLATE / "starters" / f"inference.{contract['starter']}.py").is_file():
        problems.append(f"no starter inference.{contract['starter']}.py in _template/starters")
    budget = contract["budget"]
    if not (isinstance(budget.get("agent_hours"), int) and 0 < budget["agent_hours"] <= 12):
        problems.append("budget.agent_hours must be a whole number of hours, 1..12")

    assets = task_dir / "assets"
    splits = {"public": "public", "hidden": "hidden"}
    if contract["debug_model"] is not None:
        splits["debug-9b"] = "debug-9b"
    for directory, split in splits.items():
        for name in EVALUATOR_DATA:
            if not (assets / directory / name).is_file():
                problems.append(f"assets/{directory}/{name} is missing")
        if problems:
            continue
        ppl = json.loads((assets / directory / "ppl_config.json").read_text())
        decode = json.loads((assets / directory / "decode_config.json").read_text())
        want_protocol = (contract["evaluation"]["ppl_protocol"] if split != "debug-9b"
                         else _policy_module().DEBUG_PPL_PROTOCOL)
        if ppl.get("split") != split:
            problems.append(f"assets/{directory}/ppl_config.json is split {ppl.get('split')!r}")
        if ppl.get("protocol") != want_protocol:
            problems.append(f"assets/{directory} PPL protocol {ppl.get('protocol')!r} != {want_protocol!r}")
        if decode.get("measurement_protocol") != MEASUREMENT_PROTOCOL:
            problems.append(f"assets/{directory} decode protocol is not {MEASUREMENT_PROTOCOL}")
        if split == "debug-9b":
            continue
        # decode_config.json's own `scoring_protocol` is a label no evaluator reads;
        # the scored version is the policy's.
        if ppl.get("baseline_status") != "measured_new_protocol" or not ppl.get("bf16_ppl"):
            problems.append(f"assets/{directory}: BF16 baseline not measured")
    problems += accuracy_problems(assets)
    problems += model_facts.problems(task_dir, contract)
    if contract["evaluation"]["scoring_protocol"] != SCORING_PROTOCOL:
        problems.append(f"scoring_protocol must be {SCORING_PROTOCOL}, the evaluator's reward")
    # Fidelity is scored on hidden text continued by the task's own BF16 checkpoint
    # (build_fidelity_assets.py).
    if contract["evaluation"]["ppl_protocol"] not in _policy_module().LIVE_PROTOCOLS:
        problems.append(f"ppl_protocol {contract['evaluation']['ppl_protocol']!r} is not a live BF16-continued "
                        "mixed-text protocol; build the blocks with build_fidelity_assets.py")
    if not problems:
        public = (assets / "public" / "ppl_inputs.json").read_bytes()
        hidden = (assets / "hidden" / "ppl_inputs.json").read_bytes()
        if public == hidden:
            problems.append("public and hidden perplexity inputs are identical")
        if (assets / "public" / "decode_inputs.json").read_bytes() == (assets / "hidden" / "decode_inputs.json").read_bytes():
            problems.append("public and hidden decode prompts are identical")
    tokenizer = assets / "tokenizer"
    for name in contract["evaluation"]["official_tokenizer_files"]:
        path = tokenizer / name
        if path.is_symlink() or not path.is_file():
            problems.append(f"assets/tokenizer/{name} is missing")
    return problems


def accuracy_problems(assets: Path) -> list[str]:
    """The hidden correctness gate's inputs every task needs: present only
    in assets/hidden, pinned, and with a measured BF16 baseline above chance."""
    problems: list[str] = []
    for name in render.ACCURACY_DATA:
        if (assets / "public" / name).exists():
            problems.append(f"assets/public/{name}: the correctness gate has no public split")
    paths = [assets / "hidden" / name for name in render.ACCURACY_DATA]
    if not all(p.is_file() for p in paths):
        return problems + [f"assets/hidden lacks {list(render.ACCURACY_DATA)}"]
    cfg = json.loads(paths[0].read_text())
    if cfg.get("split") != "hidden":
        problems.append(f"assets/hidden/accuracy_config.json is split {cfg.get('split')!r}")
    if cfg.get("protocol") != GATE_PROTOCOL:
        problems.append(f"assets/hidden gate protocol {cfg.get('protocol')!r}")
    if cfg.get("inputs_sha256") != hashlib.sha256(paths[1].read_bytes()).hexdigest():
        problems.append("assets/hidden/accuracy_inputs.json differs from its recorded digest")
    bf16 = cfg.get("bf16") or {}
    if cfg.get("baseline_status") != "measured" or not isinstance(bf16.get("accuracy"), float):
        problems.append("assets/hidden: BF16 accuracy not measured")
    elif bf16["accuracy"] <= cfg.get("chance", 1.0):
        problems.append("assets/hidden: BF16 accuracy does not exceed chance")
    return problems


def check(tasks: list[str]) -> int:
    failed = 0
    for name in tasks:
        task_dir = sites.task_dir(name)
        problems = source_problems(task_dir)
        if not problems:
            problems = render.drift(task_dir)
        if problems:
            failed += 1
            print(f"FAIL {task_dir.name}")
            for problem in problems:
                print(f"  - {problem}")
        else:
            print(f"ok   {task_dir.name}")
    return 1 if failed else 0


def cmd_render(tasks: list[str]) -> int:
    for name in tasks:
        task_dir = sites.task_dir(name)
        problems = source_problems(task_dir)
        if problems:
            print(f"refusing to render {task_dir.name}:")
            for problem in problems:
                print(f"  - {problem}")
            return 1
        changed = render.write(task_dir)
        print(f"rendered {task_dir.name}: {len(changed)} file(s) changed")
    return check(tasks)


def cmd_show(task: str) -> int:
    task_dir = sites.task_dir(task)
    contract = render.load_contract(task_dir)
    print(json.dumps({"task_dir": str(task_dir), "contract": contract,
                      "task_policy": render.task_policy(contract),
                      "evaluation_profiles": render.evaluation_profiles(contract),
                      "agent_image": render.agent_image(contract),
                      "runtime_image": render.runtime_image(contract)}, indent=2, default=str))
    return 0


def cmd_set(args) -> int:
    task_dir = sites.task_dir(args.task)
    path = task_dir / "contract.yaml"
    text = path.read_text()
    contract = render.load_contract(task_dir)
    changes = {name: getattr(args, name) for name in sites.SCALARS if getattr(args, name, None) is not None}
    if not changes:
        raise SystemExit("nothing to set")
    for name, value in changes.items():
        site = sites.SCALARS[name]
        old = sites.contract_value(contract, name)
        if site.loosens_when and not args.reason:
            loosens = (site.loosens_when == "up" and float(value) > float(old)) or \
                      (site.loosens_when == "change" and site.kind(value) != old)
            if loosens:
                raise SystemExit(f"--{name.replace('_', '-')} {value} loosens the task (was {old}); "
                                 "pass --reason, and put the reason in the commit message")
        text, changed = blocks.set_scalar(text, site.key, sites.formatted(value, site.kind),
                                          indent=site.indent, under=site.under)
        if not changed and sites.contract_value(contract, name) != site.kind(value):
            raise SystemExit(f"could not find {name} in {path}")
    blocks.write_checked(path, text)
    back = render.load_contract(task_dir)
    for name, value in changes.items():
        if sites.contract_value(back, name) != sites.SCALARS[name].kind(value):
            raise SystemExit(f"read-back of {name} disagrees: {sites.contract_value(back, name)!r}")
    if "capacity_gib" in changes:
        print("note: capacity changed; assets/*/ppl_config.json baselines are capacity-independent, "
              "but re-check the reference row and the task description")
    return cmd_render([task_dir.name])


def _top_level(text: str, key: str, value: str) -> str:
    """Set a top-level `key: value`, adding it after `capacity_gib` when absent."""
    text, changed = blocks.set_scalar(text, key, value, indent=0)
    if changed or f"\n{key}: {value}\n" in f"\n{text}":
        return text
    lines = text.splitlines(keepends=True)
    at = next(i for i, line in enumerate(lines) if line.startswith("capacity_gib:")) + 1
    return "".join(lines[:at] + [f"{key}: {value}\n"] + lines[at:])


def write_naming(text: str, new_id: str, capacity: float) -> str:
    """The contract fields a ServerBench-style id is made of, written from the id."""
    found = TASK_ID.fullmatch(new_id)
    if not found:
        raise SystemExit("a task id is <model slug>_<capacity>g_standard_<N>x<GPU>, "
                         "e.g. qwen3_8_27b_8g_standard_1xH200NVL")
    if float(found["capacity"]) != float(capacity):
        raise SystemExit(f"{new_id} names {found['capacity']} GiB; the contract says {capacity:g}")
    slug = sites.formatted(found["slug"], str)
    text, changed = blocks.set_scalar(text, "slug", slug, indent=2, under="model")
    if not changed and f"  slug: {slug}\n" not in text:
        text, _ = blocks.insert_scalar(text, "slug", slug, indent=2, under="model")
    text = _top_level(text, "treatment", sites.formatted(found["treatment"], str))
    text = _top_level(text, "hardware", sites.formatted(found["hardware"], str))
    text, _ = blocks.set_scalar(text, "task_id", new_id, indent=0)
    text, _ = blocks.set_scalar(text, "task_hash", f'"{task_hash(new_id)}"', indent=0)
    return text


def cmd_new(args) -> int:
    """A new task cloned from an existing one's sources (same model, tokenizer and inputs)."""
    source = sites.task_dir(args.from_task)
    new_id = args.task
    if not TASK_ID.fullmatch(new_id):
        raise SystemExit("a task id is <model slug>_<capacity>g_standard_<N>x<GPU>")
    target = sites.TASKS / new_id
    if target.exists():
        raise SystemExit(f"{target} exists")
    target.mkdir()
    shutil.copytree(source / "assets", target / "assets")
    text = (source / "contract.yaml").read_text()
    text, _ = blocks.set_scalar(text, "capacity_gib", sites.formatted(args.capacity_gib, float), indent=0)
    text = write_naming(text, new_id, args.capacity_gib)
    for name in ("version", "description"):
        value = getattr(args, name)
        if value is not None:
            text, _ = blocks.set_scalar(text, name, sites.formatted(value, str), indent=0)
    # The diagnostic reference row belongs to the source's capacity tier.
    text, _ = blocks.set_scalar(text, "label", "null", indent=2, under="reference")
    blocks.write_checked(target / "contract.yaml", text)
    print(f"created {target} from {source.name}")
    return cmd_render([new_id])


def cmd_model_facts(args) -> int:
    """Write assets/model_facts.json from the pinned checkpoint (config + shard headers only)."""
    task_dir = sites.task_dir(args.task)
    contract = render.load_contract(task_dir)
    model_dir = args.model_dir
    if model_dir is None:
        sys.path.insert(0, str(sites.REPO / "compression_bench" / "scripts"))
        import model_for_task
        model_dir = model_for_task.resolve(task_dir)
        if model_dir is None:
            raise SystemExit(f"{task_dir.name}: the pinned checkpoint is not on this host; pass --model-dir")
    facts = model_facts.build(Path(model_dir), contract)
    blocks.write_checked(task_dir / "assets" / "model_facts.json", json.dumps(facts, indent=1) + "\n")
    print(f"{task_dir.name}: model facts from {model_dir} ({facts['tensor_count']} tensors, "
          f"{len(facts['groups'])} groups)")
    return cmd_render([task_dir.name])


def cmd_rename(args) -> int:
    task_dir = sites.task_dir(args.task)
    new_id = args.to
    target = sites.TASKS / new_id
    if target.exists():
        raise SystemExit(f"{target} exists")
    tracked = subprocess.run(["git", "-C", str(sites.REPO), "ls-files", "--error-unmatch",
                              str(task_dir / "contract.yaml")], capture_output=True).returncode == 0
    if tracked:
        subprocess.run(["git", "-C", str(sites.REPO), "mv", str(task_dir), str(target)], check=True)
    else:
        task_dir.rename(target)
    path = target / "contract.yaml"
    text = path.read_text()
    contract = render.load_contract(target)
    text = write_naming(text, new_id, contract["capacity_gib"])
    blocks.write_checked(path, text)
    print(f"renamed {task_dir.name} -> {new_id}; evidence under {[str(p) for p in sites.EVIDENCE]} "
          "keeps the id it ran under")
    return cmd_render([new_id])


#: How `set` parses a value of each kind from the command line.
ARG_TYPES = {list: lambda text: [part for part in text.split(",") if part]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("show").add_argument("task")
    sub.add_parser("check").add_argument("task", nargs="?")
    sub.add_parser("render").add_argument("task", nargs="?")
    s = sub.add_parser("set")
    s.add_argument("task")
    for name, site in sites.SCALARS.items():
        s.add_argument(f"--{name.replace('_', '-')}", dest=name, type=ARG_TYPES.get(site.kind, site.kind))
    s.add_argument("--reason")
    n = sub.add_parser("new")
    n.add_argument("task")
    n.add_argument("--from", dest="from_task", required=True)
    n.add_argument("--capacity-gib", type=float, required=True)
    n.add_argument("--version")
    n.add_argument("--description")
    r = sub.add_parser("rename")
    r.add_argument("task")
    r.add_argument("--to", required=True)
    mf = sub.add_parser("model-facts")
    mf.add_argument("task")
    mf.add_argument("--model-dir", type=Path)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    every = sites.live_tasks()
    if args.command == "show":
        return cmd_show(args.task)
    if args.command == "check":
        return check([args.task] if args.task else every)
    if args.command == "render":
        return cmd_render([args.task] if args.task else every)
    if args.command == "set":
        return cmd_set(args)
    if args.command == "new":
        return cmd_new(args)
    if args.command == "rename":
        return cmd_rename(args)
    if args.command == "model-facts":
        return cmd_model_facts(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())
