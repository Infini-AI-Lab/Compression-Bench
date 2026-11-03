"""The task-owned evaluation policy: the one input that differs between tasks.

Every other evaluator module is byte-identical across tasks. Each task carries
`evaluator-assets/task_policy.json`, written by `taskctl render` from the task's
`contract.yaml`; this module reads and validates it. Stdlib only, no GPU.

Fields (all required):

  task_id            the task this copy belongs to
  size_limit_gib     package and resident-storage limit (1 GiB = 2**30 bytes)
  payload_rule       what counts as model weights in submission/model/:
                     "safetensors-or-gguf" | "any-payload"
  link_rule          which package files the private evaluation view links
                     instead of copying: "safetensors-or-gguf" | "model-payload"
  official_tokenizer_files
                     task-owned tokenizer files substituted into every package
  ppl_protocol       a key of PPL_PROTOCOLS (a task: one of LIVE_PROTOCOLS)
  scoring_protocol   the scoring version string reported with every score
  quality_rule       "bf16-fidelity": PPL scored against the BF16 baseline; in
                     the hidden verdict 0 when the correctness gate keeps less
                     than 1 - max_accuracy_drop of BF16's above-chance accuracy
                     (fidelity.py, accuracy_benchmark.py)
  quality            the rule's parameters, {"max_accuracy_drop": fraction}
  smoke_prompt       "plain" | "chat"  how the smoke prompt is tokenized
  hidden_reference_fields
                     report the reference PPL alongside hidden PPL results
"""
from __future__ import annotations

import json
import math
from pathlib import Path

SCHEMA = "compression-bench-task-policy-v1"
ASSETS_DIR = Path(__file__).resolve().parent / "evaluator-assets"

# Fixed PPL protocols, 256 blocks of 2048 tokens each. prefix_tokens is the count
# of unscored leading tokens per block; chat_prefix is the length of the fixed chat
# prefix every block must start with (0: none), named by prefix_label.
PPL_PROTOCOLS = {
    # The debug-9b Checker profile: raw C4 blocks (the first token has no context).
    "c4-raw-256x2048-v1": {"prefix_tokens": 1, "chat_prefix": 0, "prefix_label": None},
    # Every task scores with one of these. Each block's first 512 tokens are text of
    # a hidden corpus of mixed text (web pages, code, mathematics, question answering
    # and reasoning traces; its chat prefix included); the BF16 checkpoint continues
    # them for 1536 tokens at temperature 1 and only those are scored, so a
    # submission's excess cross-entropy over BF16 there is its KL divergence from
    # BF16 (compression_bench/scripts/deploy/build_fidelity_assets.py). The agent is
    # told only that a hidden corpus of mixed text supplies the blocks.
    "mixed-bf16-continued-256x2048-v1": {"prefix_tokens": 512, "chat_prefix": 0, "prefix_label": None},
    "mixed-gemma-chat-bf16-continued-256x2048-v1": {"prefix_tokens": 512, "chat_prefix": 19,
                                                    "prefix_label": "Gemma"},
}
#: The protocols a task may score with: BF16-continued blocks of hidden mixed text.
LIVE_PROTOCOLS = frozenset(("mixed-bf16-continued-256x2048-v1",
                            "mixed-gemma-chat-bf16-continued-256x2048-v1"))
# The debug-9b profile always uses the raw C4 protocol, whatever the task's.
DEBUG_PPL_PROTOCOL = "c4-raw-256x2048-v1"


def chat_prefix_tokens(protocol: str) -> int:
    """How many leading tokens of every block are the protocol's fixed chat prefix."""
    return PPL_PROTOCOLS[protocol]["chat_prefix"]


QUALITY_RULE = "bf16-fidelity"
PAYLOAD_RULES = ("safetensors-or-gguf", "any-payload")
LINK_RULES = ("safetensors-or-gguf", "model-payload")
SMOKE_PROMPTS = ("plain", "chat")


def validate(policy: dict) -> dict:
    if not isinstance(policy, dict) or policy.get("schema") != SCHEMA:
        raise ValueError("task policy has the wrong schema")
    limit = policy.get("size_limit_gib")
    if type(limit) not in (int, float) or not math.isfinite(limit) or limit <= 0:
        raise ValueError("task policy size_limit_gib must be positive")
    if policy.get("payload_rule") not in PAYLOAD_RULES:
        raise ValueError("unknown payload_rule")
    if policy.get("link_rule") not in LINK_RULES:
        raise ValueError("unknown link_rule")
    files = policy.get("official_tokenizer_files")
    if (not isinstance(files, list) or not files
            or any(not isinstance(f, str) or "/" in f for f in files)):
        raise ValueError("official_tokenizer_files must be plain file names")
    if policy.get("ppl_protocol") not in PPL_PROTOCOLS:
        raise ValueError("unknown ppl_protocol")
    if not isinstance(policy.get("scoring_protocol"), str):
        raise ValueError("scoring_protocol must be a string")
    if policy.get("quality_rule") != QUALITY_RULE:
        raise ValueError(f"quality_rule must be {QUALITY_RULE}")
    quality = policy.get("quality")
    if not isinstance(quality, dict) or set(quality) != {"max_accuracy_drop"}:
        raise ValueError("quality parameters must be exactly ('max_accuracy_drop',)")
    drop = quality["max_accuracy_drop"]
    if type(drop) not in (int, float) or not math.isfinite(drop) or not 0 < drop < 1:
        raise ValueError("max_accuracy_drop must be a fraction in (0, 1)")
    if policy.get("smoke_prompt") not in SMOKE_PROMPTS:
        raise ValueError("unknown smoke_prompt")
    if type(policy.get("hidden_reference_fields")) is not bool:
        raise ValueError("hidden_reference_fields must be a boolean")
    return policy


def load(directory=None) -> dict:
    """Read the policy next to the calling evaluator copy (or in `directory`)."""
    path = Path(directory or ASSETS_DIR) / "task_policy.json"
    return validate(json.loads(path.read_text(encoding="utf-8")))


_CURRENT = None


def current() -> dict:
    """This evaluator copy's policy, read once on first use."""
    global _CURRENT
    if _CURRENT is None:
        _CURRENT = load()
    return _CURRENT


def ppl_protocol_for_split(split) -> str:
    return DEBUG_PPL_PROTOCOL if split == "debug-9b" else current()["ppl_protocol"]
