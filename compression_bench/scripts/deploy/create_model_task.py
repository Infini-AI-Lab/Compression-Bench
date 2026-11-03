#!/usr/bin/env python3
"""Create a task for a model no live task uses yet: the contract and every input.

    python3 compression_bench/scripts/deploy/create_model_task.py \\
        --task-id qwen3_5_122b_a10b_48g_standard_1xH200NVL --model-dir <snapshot> \\
        --model-id Qwen/Qwen3.5-122B-A10B --revision <sha> --title Qwen3.5-122B-A10B \\
        --local-name Qwen3.5-122B-A10B [--mtp-note]

`taskctl new` clones a capacity tier of a model that already has a task; this
is the step before it, for a new model. From the pinned checkpoint's own
official tokenizer it builds:

  assets/tokenizer/           the official tokenizer files (and chat template)
  assets/{public,hidden}/ppl_*   protocol `c4-raw-256x2048-v1`: 256 independent
                              2048-token C4 validation blocks, seed 0, a uniform
                              document then a uniform window; public from shard 0,
                              hidden from shard 2 (allenai/c4 @ C4_REVISION). They
                              are the starting blocks; `build_fidelity_assets.py
                              --install` replaces them with BF16-continued blocks
  assets/{public,hidden}/decode_*   one warmup and three measured 2048-token
                              prompts, drawn the same way with seed "decode"
                              from documents the PPL blocks do not use
  assets/hidden/accuracy_*    the hidden correctness gate (build_accuracy_assets.py)

and writes the contract (`CONTRACT_TEMPLATE`) with the live tasks' policy (gguf
platform and repairs, bf16-fidelity, 80% gate). Every BF16 baseline is left
pending. Then, on a GPU: `build_fidelity_assets.py --install` (the perplexity
blocks and their BF16 baselines), `taskctl set <task> --ppl-protocol ...`,
`measure_bf16_accuracy.py` or `measure_bf16_baselines.py` (the gate's BF16
accuracy), and `taskctl render`. Deploy-time tool: needs network once (the
Hub), transformers and huggingface_hub.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TASKS = REPO / "compression_tasks"
sys.path.insert(0, str(TASKS / "_manage"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

C4_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
PPL_PROTOCOL = "c4-raw-256x2048-v1"
SHARDS = {"public": 0, "hidden": 2}
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
import render  # noqa: E402

#: The contract skeleton; `contract_text` fills every {{PLACEHOLDER}}.
CONTRACT_TEMPLATE = """\
# A task source, written by compression_bench/scripts/deploy/create_model_task.py; change it with taskctl (`set`, `rename`).
# Every generated file of this task (task.toml, instruction.md, README.md,
# MODE.md, environment/, tests/) is rendered from this file and assets/.
kind: compression_task_contract
schema: compression_bench.task_contract/1
task_id: {{TASK_ID}}
task_hash: "{{TASK_HASH}}"   # quoted: an all-digit hash would otherwise read as a number
version: "{{VERSION}}"
description: "{{DESCRIPTION}}"
keywords: {{KEYWORDS}}

# The source checkpoint the agent compresses, mounted read-only at /input/model.
model:
  id: {{MODEL_ID}}
  revision: {{MODEL_REVISION}}
  title: {{MODEL_TITLE}}       # as the instruction's title names it
  display: {{MODEL_DISPLAY}}   # as the instruction's prose names it
  local_name: {{MODEL_LOCAL_NAME}}   # <models root>/<local_name>-bf16 on the host that runs the task
  mtp_note: {{MTP_NOTE}}       # the model carries an MTP head the instruction must address

# One limit, applied independently to the submission package and to the
# GPU-resident model storage (1 GiB = 2**30 bytes).
capacity_gib: {{CAPACITY_GIB}}

# Where the agent, the public Checker and the Verifier run.
#   image: gguf  -> compression_bench/platform-gguf (agent-base + pinned vllm-gguf-plugin)
#   patches: pinned platform repairs from compression_bench/environments/platform-gguf/patches/
platform:
  image: {{PLATFORM_IMAGE}}
  patches: {{PLATFORM_PATCHES}}
starter: {{STARTER}}   # compression_tasks/_template/starters/inference.<starter>.py

# The evaluation policy, rendered into every evaluator copy as
# evaluator-assets/task_policy.json (see compression_bench/harness/final_validate/task_policy.py).
evaluation:
  payload_rule: {{PAYLOAD_RULE}}
  link_rule: {{LINK_RULE}}
  official_tokenizer_files: {{TOKENIZER_FILES}}
  ppl_protocol: {{PPL_PROTOCOL}}
  scoring_protocol: {{SCORING_PROTOCOL}}
  quality_rule: {{QUALITY_RULE}}
  quality: {{QUALITY}}
  smoke_prompt: {{SMOKE_PROMPT}}
  hidden_reference_fields: {{HIDDEN_REFERENCE_FIELDS}}

# A published quantized checkpoint of the model, recorded as provenance in task.toml
# [metadata]; never a gate. Measured values live in assets/{public,hidden}/*_config.json.
reference:
  label: {{REFERENCE_LABEL}}
  model: {{REFERENCE_MODEL}}
  revision: {{REFERENCE_REVISION}}
  file: {{REFERENCE_FILE}}

# Optional debug checkpoint mounted at /input/debug-model (public Checker profile debug-9b).
debug_model: {{DEBUG_MODEL}}

budget:
  agent_hours: {{AGENT_HOURS}}
  public_check_timeout_s: {{PUBLIC_CHECK_TIMEOUT_S}}
  verifier_timeout_s: {{VERIFIER_TIMEOUT_S}}
  build_timeout_s: {{BUILD_TIMEOUT_S}}

# Provenance carried verbatim into task.toml [metadata].
metadata:
{{METADATA}}
"""

DECODE = {"measurement_protocol": "submitted-decode-single-token-3x3-v3", "vllm_version": render.VLLM_VERSION,
          "input_tokens": 2048, "output_tokens": 128, "requests_per_round": 3, "rounds": 3,
          "max_model_len": 2176, "max_num_seqs": 1, "gpu_memory_utilization": 0.9}


def c4_texts(shard: int, cache: Path) -> tuple[list[str], str]:
    from huggingface_hub import hf_hub_download
    file = f"en/c4-validation.{shard:05d}-of-00008.json.gz"
    path = hf_hub_download("allenai/c4", file, repo_type="dataset", revision=C4_REVISION, cache_dir=cache)
    with gzip.open(path, "rt") as handle:
        return [json.loads(line)["text"] for line in handle], file


def draw(texts, tokenizer, count, rng, skip=frozenset()):
    """A uniformly drawn document longer than 2048 tokens, then a uniform 2048-token window of it."""
    blocks, selected = [], []
    while len(blocks) < count:
        index = rng.randint(0, len(texts) - 1)
        if index in skip:
            continue
        ids = tokenizer(texts[index])["input_ids"]
        if len(ids) <= 2048:          # strictly longer than the window
            continue
        start = rng.randint(0, len(ids) - 2048 - 1)
        blocks.append(ids[start:start + 2048])
        selected.append({"document": index, "start": start})
    return blocks, selected


def build_inputs(task_dir: Path, cache: Path) -> dict:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(task_dir / "assets" / "tokenizer"),
                                              local_files_only=True, trust_remote_code=False)
    report = {}
    for split, shard in SHARDS.items():
        out = task_dir / "assets" / split
        out.mkdir(parents=True, exist_ok=True)
        texts, file = c4_texts(shard, cache)
        source = {"repository": "allenai/c4", "revision": C4_REVISION, "file": file, "documents": len(texts)}
        blocks, selected = draw(texts, tokenizer, 256, random.Random(0))
        data = (json.dumps(blocks, separators=(",", ":")) + "\n").encode()
        (out / "ppl_inputs.json").write_bytes(data)
        (out / "ppl_config.json").write_text(json.dumps({
            "protocol": PPL_PROTOCOL, "split": split, "blocks": 256, "context_length": 2048,
            "input_tokens": 524288, "scored_tokens": 524032, "vocab_size": len(tokenizer),
            "tokenizer_class": type(tokenizer).__name__, "source": source, "seed": 0,
            "sampling": "uniform document/start draws, seed 0, tokenizer defaults, 256 independent 2048-token blocks",
            "selected": selected, "inputs_sha256": hashlib.sha256(data).hexdigest(),
            "baseline_status": "pending_new_protocol", "bf16_ppl": None}, indent=1) + "\n")
        used = frozenset(s["document"] for s in selected)
        prompts, chosen = draw(texts, tokenizer, 4, random.Random("decode"), skip=used)
        (out / "decode_inputs.json").write_text(json.dumps({"warmup": prompts[0], "measured": prompts[1:]},
                                                           separators=(",", ":")) + "\n")
        (out / "decode_config.json").write_text(json.dumps({
            **DECODE, "split": split, "source": source, "selected": chosen,
            "sampling": "the same draws with seed 'decode' from documents the PPL blocks do not use",
            "score_status": "provisional", "bf16_latency_seconds": None}, indent=1) + "\n")
        report[split] = {"ppl_blocks": len(blocks), "decode_prompts": len(prompts), "shard": file}
    import build_accuracy_assets
    report["gate"] = build_accuracy_assets.build(task_dir)
    return report


def contract_text(args) -> str:
    import taskctl
    template = CONTRACT_TEMPLATE
    values = {
        "TASK_ID": args.task_id, "TASK_HASH": taskctl.task_hash(args.task_id), "VERSION": "0.10.0-mixed-text-fidelity",
        "DESCRIPTION": f"Compress {args.display or args.title} to {args.capacity_gib:g} GiB; reward is fidelity to "
                       "BF16 (perplexity on a hidden corpus of mixed text continued by the BF16 checkpoint), "
                       "behind a hidden correctness gate.",
        "KEYWORDS": json.dumps(["compression", "llm", args.keyword, "inference"]),
        "MODEL_ID": args.model_id, "MODEL_REVISION": args.revision, "MODEL_TITLE": args.title,
        "MODEL_DISPLAY": args.display or args.title, "MODEL_LOCAL_NAME": args.local_name,
        "MTP_NOTE": "true" if args.mtp_note else "false", "CAPACITY_GIB": f"{args.capacity_gib:g}",
        "PLATFORM_IMAGE": '"gguf"', "PLATFORM_PATCHES": '["text-architecture", "q8-tail", "operator-ops"]',
        "STARTER": "vllm", "PAYLOAD_RULE": '"safetensors-or-gguf"', "LINK_RULE": '"safetensors-or-gguf"',
        "TOKENIZER_FILES": json.dumps(args.tokenizer_files), "PPL_PROTOCOL": PPL_PROTOCOL,
        "SCORING_PROTOCOL": '"bf16-sample-ppl-gated-v5"', "QUALITY_RULE": '"bf16-fidelity"',
        "QUALITY": '{"max_accuracy_drop": 0.2}', "SMOKE_PROMPT": "plain",
        "HIDDEN_REFERENCE_FIELDS": "true",
        "REFERENCE_LABEL": "null", "REFERENCE_MODEL": "null", "REFERENCE_REVISION": "null",
        "REFERENCE_FILE": "null", "DEBUG_MODEL": "null", "AGENT_HOURS": "8",
        "PUBLIC_CHECK_TIMEOUT_S": str(args.timeout_s), "VERIFIER_TIMEOUT_S": str(args.timeout_s),
        "BUILD_TIMEOUT_S": "3600",
        "METADATA": "\n".join(f"  {k}: {v}" for k, v in {
            "accuracy_protocol": '"hidden-gen-gate-v2"',
            "model_memory_protocol": '"resident-model-storage-v1"',
            "objective": '"maximize_bf16_fidelity_ppl_ratio_on_bf16_continued_mixed_text_after_capacity_and_hidden_correctness_gates"',
            "ppl_baseline_status": '"pending_new_protocol"',
            "decode_protocol": '"submitted-decode-single-token-3x3-v3"',
            "decode_workload_split": '"public-hidden-v1"',
            "inference_protocol": '"submitted-inference-audited-v1"',
            "modality_policy": '"text-only-nontext-removal-allowed-v1"',
            "storage_protocol": '"single-copy-locked-v1"',
            "review_required": "true", "score_status": '"provisional"'}.items()),
    }
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", value)
    if "{{" in template:
        raise SystemExit("contract template has unfilled fields")
    return template


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--model-dir", type=Path, required=True, help="the pinned checkpoint snapshot")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--display")
    parser.add_argument("--local-name", required=True)
    parser.add_argument("--keyword", required=True, help="the model family keyword, e.g. qwen")
    parser.add_argument("--mtp-note", action="store_true")
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "compression-bench-c4")
    args = parser.parse_args(argv)
    import taskctl
    found = taskctl.TASK_ID.fullmatch(args.task_id)
    if not found:
        raise SystemExit("a task id is <model slug>_<capacity>g_<treatment>_<N>x<GPU>")
    args.capacity_gib = float(found["capacity"])
    task_dir = TASKS / args.task_id
    if task_dir.exists():
        raise SystemExit(f"{task_dir} exists")
    files = [name for name in TOKENIZER_FILES if (args.model_dir / name).is_file()]
    if "tokenizer.json" not in files:
        raise SystemExit("the checkpoint has no tokenizer.json")
    args.tokenizer_files = files
    (task_dir / "assets" / "tokenizer").mkdir(parents=True)
    for name in files:
        shutil.copy2(args.model_dir / name, task_dir / "assets" / "tokenizer" / name)
    text = contract_text(args)
    text = taskctl.write_naming(text, args.task_id, args.capacity_gib)
    (task_dir / "contract.yaml").write_text(text)
    print(args.task_id, json.dumps(build_inputs(task_dir, args.cache)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
