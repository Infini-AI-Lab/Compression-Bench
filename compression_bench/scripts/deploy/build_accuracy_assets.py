#!/usr/bin/env python3
"""Build a task's hidden correctness-gate inputs.

    python3 compression_bench/scripts/deploy/build_accuracy_assets.py compression_tasks/<task> [...]

The correctness gate runs once, in the final verdict, on items only the hidden
Verifier holds: a small sample, ITEMS per benchmark, drawn once with a fixed seed
from pinned dataset revisions, answered by greedy generation
(`accuracy_benchmark.py`):

  math       GSM8K test (openai/gsm8k): 4-shot `Question: ...\\nAnswer: ...` with
             worked solutions (train problems, calculator annotations removed);
             judged by the final number. Test problems and shots are drawn
             past the first HELD_OUT positions of their split shuffled with
             seed `gsm8k-<split>-0` (`_order`).
  code       HumanEval (openai/openai_humaneval): the function prompt, completed;
             judged by running the problem's unit tests.
  knowledge  TriviaQA closed-book (mandarjoshi/trivia_qa rc.nocontext validation):
             5-shot `Question: ...\\nAnswer: ...` (train questions); judged by
             exact match of the first line against the normalised aliases.
             Only questions of at least ten words are drawn, so the guard's
             10-word signature can keep each one out of the agent's container.

No item of these datasets, of any split (`GUARD_FILES`), is placed anywhere the
agent's container can read (`benchmark_guard.py`), and the agent is told only
that a hidden multi-task benchmark supplies the gate, never which.

Every prompt starts with the task's chat prefix when its perplexity blocks have one
(Gemma's non-thinking turn opening, BOS included: an instruction-tuned model given
bare few-shot text answers in chat style or degenerates), else the tokenizer's BOS token when it has one (Qwen has none), and is
tokenized with the task's official tokenizer.
Writes `assets/hidden/accuracy_inputs.json` and `accuracy_config.json`; the BF16
baseline is measured afterwards (`measure_bf16_accuracy.py`).

Needs network once (the Hugging Face Hub), pyarrow, huggingface_hub and
transformers; it is a deploy-time tool, not part of the harness core.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

PROTOCOL = "hidden-gen-gate-v2"
ITEMS = 50
SPLIT = "hidden"
MAX_MODEL_LEN = 2049
#: The first positions of each GSM8K split's seeded order (`_order`), never drawn.
HELD_OUT = 1000

#: name -> (repo, revision, gate files, kind, max_new_tokens).
BENCHMARKS = {
    "math": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866",
             ["main/test-00000-of-00001.parquet"], "final-number", 320),
    "code": ("openai/openai_humaneval", "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
             ["openai_humaneval/test-00000-of-00001.parquet"], "unit-tests", 384),
    "knowledge": ("mandarjoshi/trivia_qa", "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f",
                  ["rc.nocontext/validation-00000-of-00001.parquet"], "short-answer", 24),
}
#: The split the few-shot examples come from, and how many.
SHOTS = {"math": (["main/train-00000-of-00001.parquet"], 4),
         "knowledge": (["rc.nocontext/train-00000-of-00001.parquet"], 5)}
#: Every split of each dataset, whose items must not reach the agent's container.
GUARD_FILES = {
    "math": ["main/train-00000-of-00001.parquet", "main/test-00000-of-00001.parquet"],
    "code": ["openai_humaneval/test-00000-of-00001.parquet"],
    "knowledge": ["rc.nocontext/train-00000-of-00001.parquet",
                  "rc.nocontext/validation-00000-of-00001.parquet",
                  "rc.nocontext/test-00000-of-00001.parquet"],
}
#: The field holding each benchmark's question, as `benchmark_guard.py` looks for it
#: (HumanEval's is its docstring, added by `rows_of`).
QUESTION_FIELD = {"math": "question", "code": "question", "knowledge": "question"}
ANNOTATION = re.compile(r"<<[^>]*>>")
_DOCSTRING = re.compile(r'"""(.*?)"""|\'\'\'(.*?)\'\'\'', re.S)
_WORD = re.compile(r"[a-z0-9]+")


def rows_of(name: str, files: list[str] | None = None) -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    repo, revision, gate_files, _, _ = BENCHMARKS[name]
    out = []
    for file in files or gate_files:
        path = hf_hub_download(repo, file, repo_type="dataset", revision=revision)
        out += pq.read_table(path).to_pylist()
    if name == "code":
        for row in out:
            found = _DOCSTRING.search(row["prompt"])
            row["question"] = (found.group(1) or found.group(2)).strip() if found else row["prompt"]
    return out


def solution(row: dict) -> str:
    return ANNOTATION.sub("", row["answer"])


def final(row: dict) -> str:
    return row["answer"].rsplit("####", 1)[1].strip().replace(",", "")


def usable(name: str, row: dict) -> bool:
    if name == "knowledge":
        answer = row["answer"]
        return (len(_WORD.findall(row["question"].lower())) >= 10 and bool(answer["normalized_aliases"])
                and 1 <= len(answer["value"].split()) <= 5)
    return True


def _order(split: str, count: int) -> list[int]:
    """A GSM8K split's indices shuffled with seed `gsm8k-<split>-0`."""
    order = list(range(count))
    random.Random(f"gsm8k-{split}-0").shuffle(order)
    return order


def gate_rows(name: str) -> list[dict]:
    """The rows the gate holds, in item order."""
    rows = rows_of(name)
    if name == "math":
        candidates = _order("test", len(rows))[HELD_OUT:]
    else:
        candidates = [i for i, row in enumerate(rows) if usable(name, row)]
    chosen = sorted(random.Random(f"{PROTOCOL}/{name}").sample(candidates, ITEMS))
    return [rows[i] for i in chosen]


def shot_rows(name: str) -> list[dict]:
    if name not in SHOTS:
        return []
    files, count = SHOTS[name]
    rows = rows_of(name, files)
    if name == "math":
        candidates = _order("train", len(rows))[HELD_OUT:]
    else:
        candidates = [i for i, row in enumerate(rows) if usable(name, row)]
    return [rows[i] for i in random.Random(f"{PROTOCOL}/{name}-shots").sample(candidates, count)]


def prompt_and_answer(name: str, row: dict, shots: list[dict]) -> tuple[str, dict]:
    if name == "math":
        head = "".join(f"Question: {s['question']}\nAnswer: {solution(s)}\n\n" for s in shots)
        return head + f"Question: {row['question']}\nAnswer:", {"number": final(row)}
    if name == "knowledge":
        head = "".join(f"Question: {s['question']}\nAnswer: {s['answer']['value']}\n\n" for s in shots)
        aliases = sorted(set(row["answer"]["normalized_aliases"]) | {row["answer"]["normalized_value"]})
        return head + f"Question: {row['question']}\nAnswer:", {"aliases": aliases}
    return row["prompt"], {"prompt": row["prompt"], "test": row["test"], "entry_point": row["entry_point"]}


def tokens(tokenizer, text: str, chat: list[int] | None = None) -> list[int]:
    """`text`'s IDs, after the task's chat prefix (an instruction-tuned model's non-thinking
    turn opening, the one its perplexity blocks start with; it carries the BOS token) or,
    without one, the tokenizer's BOS token when it has one."""
    if chat:
        return list(chat) + tokenizer(text, add_special_tokens=False).input_ids
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    return bos + tokenizer(text, add_special_tokens=False).input_ids


def build(task_dir: Path) -> dict:
    from transformers import AutoTokenizer
    ppl = json.loads((task_dir / "assets" / "public" / "ppl_config.json").read_text())
    chat = list(ppl["prefix_tokens"]) if isinstance(ppl.get("prefix_tokens"), list) else []
    tokenizer = AutoTokenizer.from_pretrained(str(task_dir / "assets" / "tokenizer"),
                                              local_files_only=True, trust_remote_code=False)
    out_items, benchmarks, token_count, longest = [], [], 0, 0
    for index, name in enumerate(sorted(BENCHMARKS)):
        repo, revision, files, kind, max_new = BENCHMARKS[name]
        shots = shot_rows(name)
        rows = gate_rows(name)
        for row in rows:
            text, answer = prompt_and_answer(name, row, shots)
            ids = tokens(tokenizer, text, chat)
            if len(ids) + max_new > MAX_MODEL_LEN:
                raise AssertionError(f"a {name} prompt of {len(ids)} tokens does not fit max_model_len")
            token_count += len(ids)
            longest = max(longest, len(ids))
            out_items.append({"b": index, "prompt": ids, "answer": answer})
        benchmarks.append({"name": name, "kind": kind, "dataset": repo, "revision": revision,
                           "files": files, "items": len(rows), "max_new_tokens": max_new,
                           "shots": len(shots), "chance": 0.0})
    data = (json.dumps({"protocol": PROTOCOL, "split": SPLIT, "items": out_items},
                       separators=(",", ":")) + "\n").encode()
    config = {"protocol": PROTOCOL, "split": SPLIT, "benchmarks": benchmarks, "items": len(out_items),
              "input_tokens": token_count, "longest_prompt_tokens": longest,
              "vocab_size": len(tokenizer), "tokenizer_class": type(tokenizer).__name__,
              "inputs_sha256": hashlib.sha256(data).hexdigest(),
              "format": "greedy generation; final-number, short-answer exact match, unit tests run",
              "bos_token_id": tokenizer.bos_token_id, "chat_prefix_tokens": len(chat), "chance": 0.0,
              "baseline_status": "pending", "bf16": None}
    target = task_dir / "assets" / SPLIT
    previous = target / "accuracy_config.json"
    if previous.is_file():
        old = json.loads(previous.read_text())
        if old.get("inputs_sha256") == config["inputs_sha256"] and old.get("bf16"):
            config["baseline_status"], config["bf16"] = old["baseline_status"], old["bf16"]
    (target / "accuracy_inputs.json").write_bytes(data)
    (target / "accuracy_config.json").write_text(json.dumps(config, indent=2) + "\n")
    for stale in ("accuracy_inputs.json", "accuracy_config.json"):     # the gate has no public split
        (task_dir / "assets" / "public" / stale).unlink(missing_ok=True)
    return {"items": config["items"], "input_tokens": token_count, "longest_prompt_tokens": longest,
            "inputs_sha256": config["inputs_sha256"][:12]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tasks", nargs="+", type=Path)
    args = parser.parse_args(argv)
    for task in args.tasks:
        print(task.name, json.dumps(build(task.resolve())), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
