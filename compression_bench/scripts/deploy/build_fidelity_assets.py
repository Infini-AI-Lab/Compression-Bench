#!/usr/bin/env python3
"""Build a model's fidelity perplexity blocks: mixed text continued by its own BF16 checkpoint.

    python3 compression_bench/scripts/deploy/build_fidelity_assets.py --gpu N[,M...] \\
        compression_tasks/<task> [compression_tasks/<task> ...] [--install]

The tasks must share one model and one tokenizer (the capacity tiers of one
model). For each split, 256 contexts of 512 tokens are drawn
from the Dolma 3 Dolmino 10B mix (`allenai/dolma3_dolmino_mix-10B-1025`, pinned;
see `dolmino_contexts`), packed after the protocol's chat prefix (gemma's 19
tokens, from the task's current blocks) and cut at token 512, each document used
once: the mix's shards are split by hash into a public half and a hidden half (so
the two splits share no shard), 64 shards per split (stratified by source family,
in proportion to the mix), 24 seeded random documents from each.

Each context is continued for 1536 tokens by the BF16 checkpoint, run through the
task's own starter `inference.py` engine in the task's runtime image, sampled at
temperature 1 with every other sampling option off (a fixed seed per block,
end-of-sequence ignored, so the continuation is a draw from BF16's own next-token
distribution). BF16's perplexity on the continuation tokens is then measured with
the evaluator's `ppl_benchmark.measure` -- the code the Checker and the Verifier
score with -- and recorded as the split's `bf16_ppl`. On these blocks a
submission's excess cross-entropy over BF16 is its KL divergence from BF16
(`final_validate/fidelity.py`).

A model too large for one card runs tensor-parallel over the cards named
(`--gpu 2,7`): the starter's engine is used with `tensor_parallel_size` set to
their number, the only change from the starter, and the census is left out.

Everything is written under `compression_bench/evolve_space/baselines/fidelity-dolmino-<model>-<stamp>/`
(the contexts and which documents each holds, the documents' text, which
`prepare_datasets.py` keeps out of the agent's training data, every block's BF16
negative log-likelihood and the sampling record). `--install` copies
the new `ppl_inputs.json` / `ppl_config.json` into every task's `assets/<split>/`;
`--install-from <that dir>` does so later, for a task whose live run must finish
first. Run `taskctl set <task> --ppl-protocol <protocol>` afterwards.

Inspect the card first (`nvidia-smi --query-compute-apps=...`); the GPU is held
with the harness's own card claim while this runs, so no evolve picks it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[3] if len(HERE.parents) > 3 else HERE.parent   # in the container: unused
FINAL_VALIDATE = REPO / "compression_bench" / "harness" / "final_validate"
EVALUATOR = ("ppl_benchmark.py", "inference_runtime.py", "model_memory.py", "task_policy.py")
SPLITS = ("public", "hidden")
BLOCKS, CONTEXT = 256, 2048
SEED_BASE = {"public": 0, "hidden": 1_000_000}
DOLMINO_REPOSITORY = "allenai/dolma3_dolmino_mix-10B-1025"
DOLMINO_REVISION = "275d00e588098d2658e36be74ffd010dfb54d8a6"
DOLMINO_SEED = "dolmino-10B-1025"
DOLMINO_SHARDS, DOLMINO_DOCS_PER_SHARD, DOLMINO_MIN_CHARS = 64, 24, 300
#: The protocol the blocks are built for, by the chat prefix of the task's current protocol.
TARGETS = {None: "mixed-bf16-continued-256x2048-v1", "Gemma": "mixed-gemma-chat-bf16-continued-256x2048-v1"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- in the container ---------------------------------------------------------------

def dolmino_family(path: str) -> str:
    """The source family of a shard: `data/<family>/shard_N.jsonl.zst`."""
    return path.split("/")[1]


def dolmino_shards(split: str, files: list[tuple[str, int]]) -> list[str]:
    """DOLMINO_SHARDS shards for this split, STRATIFIED by source family: a family gets the same number of
    shards in both splits, in proportion to its bytes in the mix (largest remainder), so the two splits
    have the same composition in expectation; within a family the shards are this split's half (a hash of
    the path, alternating), sampled by size without replacement (seeded). The two splits share no shard."""
    half = {"public": 0, "hidden": 1}[split]
    families: dict[str, int] = {}
    for path, size in files:
        families[dolmino_family(path)] = families.get(dolmino_family(path), 0) + size
    total = sum(families.values())
    exact = {f: DOLMINO_SHARDS * size / total for f, size in families.items()}
    quota = {f: int(x) for f, x in exact.items()}
    for f in sorted(exact, key=lambda f: (exact[f] - quota[f], f), reverse=True)[:DOLMINO_SHARDS - sum(quota.values())]:
        quota[f] += 1
    chosen = []
    for family, want in sorted(quota.items()):
        # Within a family the shards alternate by the rank of a hash of the path: each split gets half.
        ranked = sorted((path, size) for path, size in files if dolmino_family(path) == family)
        ranked.sort(key=lambda item: _sha(f"{DOLMINO_SEED}/{item[0]}".encode()))
        mine = [item for rank, item in enumerate(ranked) if rank % 2 == half]
        rng = random.Random(f"{DOLMINO_SEED}-{split}-{family}-shards")
        keyed = sorted(((rng.random() ** (1.0 / max(size, 1)), path) for path, size in mine), reverse=True)
        chosen += [path for _, path in keyed[:want]]
    return sorted(chosen)


def dolmino_contexts(tokenizer_dir: Path, split: str, prompt_tokens: int, chat: list[int],
                     cache: Path) -> tuple[list[list[int]], dict]:
    """256 contexts of `prompt_tokens` tokens from the Dolma 3 Dolmino 10B mix: the chat prefix, then
    documents in a seeded order, each used once, the last one cut at the boundary. A document that holds
    any special-token string of the tokenizer is skipped (it would tokenize to a control token)."""
    import io
    import zstandard
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    specials = [t for t in tokenizer.all_special_tokens if t]
    info = HfApi().dataset_info(DOLMINO_REPOSITORY, revision=DOLMINO_REVISION, files_metadata=True)
    files = [(f.rfilename, f.size or 0) for f in info.siblings if f.rfilename.startswith("data/") and f.rfilename.endswith(".jsonl.zst")]
    shards = dolmino_shards(split, files)
    documents, skipped = [], 0
    for path in shards:
        local = hf_hub_download(DOLMINO_REPOSITORY, path, repo_type="dataset", revision=DOLMINO_REVISION, cache_dir=cache)
        texts = []
        with open(local, "rb") as handle, zstandard.ZstdDecompressor().stream_reader(handle) as reader:
            for line in io.TextIOWrapper(reader, encoding="utf-8"):
                row = json.loads(line)
                text = (row.get("text") or "").strip()
                if len(text) < DOLMINO_MIN_CHARS or any(t in text for t in specials):
                    skipped += 1
                    continue
                texts.append((row.get("id"), text))
        rng = random.Random(f"{DOLMINO_SEED}-{split}-{path}")
        for index in sorted(rng.sample(range(len(texts)), min(DOLMINO_DOCS_PER_SHARD, len(texts)))):
            documents.append({"shard": path, "id": texts[index][0], "text": texts[index][1]})
    random.Random(f"{DOLMINO_SEED}-{split}-order").shuffle(documents)
    contexts, per_block, cursor = [], [], 0
    for _ in range(BLOCKS):
        ids, picked = list(chat), []
        while len(ids) < prompt_tokens:
            if cursor == len(documents):
                raise SystemExit(f"the sampled Dolmino {split} documents are too few for {BLOCKS} contexts")
            doc = documents[cursor]
            cursor += 1
            ids += tokenizer(doc["text"] + "\n\n", add_special_tokens=False)["input_ids"]
            picked.append({"shard": doc["shard"], "id": doc["id"]})
        contexts.append(ids[:prompt_tokens])
        per_block.append(picked)
    used = [{"shard": d["shard"], "id": d["id"], "text": d["text"]} for d in documents[:cursor]]
    return contexts, {"documents_text": used, "repository": DOLMINO_REPOSITORY, "revision": DOLMINO_REVISION, "split": split,
                      "shards": shards, "documents_per_shard": DOLMINO_DOCS_PER_SHARD, "problems": len(documents),
                      "problems_used": cursor, "documents_skipped_unusable": skipped, "order_seed": DOLMINO_SEED,
                      "format": "document text, then a blank line; documents packed to the boundary",
                      "tokenizer_class": type(tokenizer).__name__, "problems_per_block": per_block}


def inside_generate(work: Path) -> None:
    sys.path.insert(0, str(work / "submission"))
    import inference
    from vllm import SamplingParams
    lengths = json.loads((work / "build.json").read_text())
    continuation_tokens = lengths["continuation_tokens"]
    engine = inference.load(str(work / "submission" / "model"), max_model_len=2049,
                            max_num_seqs=64, gpu_memory_utilization=0.9)
    for split in SPLITS:
        prompts = json.loads((work / "source" / split / "contexts.json").read_text())
        if any(len(p) != lengths["prompt_tokens"] for p in prompts):
            raise SystemExit("a context has the wrong length")
        params = [SamplingParams(temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, max_tokens=continuation_tokens,
                                 ignore_eos=True, detokenize=False, seed=SEED_BASE[split] + i)
                  for i in range(len(prompts))]
        started = time.monotonic()
        outputs = engine.model.generate([{"prompt_token_ids": p} for p in prompts], params, use_tqdm=False)
        blocks = []
        for prompt, output in zip(prompts, outputs, strict=True):
            continuation = list(output.outputs[0].token_ids)
            if len(continuation) != continuation_tokens:
                raise SystemExit(f"a continuation has {len(continuation)} tokens, not {continuation_tokens}")
            blocks.append(prompt + continuation)
        out = work / "new" / split
        out.mkdir(parents=True, exist_ok=True)
        (out / "ppl_inputs.json").write_text(json.dumps(blocks))
        print(split, "generated", len(blocks), "blocks in", round(time.monotonic() - started, 1), "s", flush=True)


def inside_score(work: Path, timeout: float) -> None:
    sys.path.insert(0, str(work / "evaluator"))
    import ppl_benchmark
    from inference_runtime import SubmittedRuntime
    for split in SPLITS:
        started = time.monotonic()
        result = ppl_benchmark.measure(work / "submission", work / "new" / split, work / "submission" / "model",
                                       SubmittedRuntime, timeout=timeout, expected_split=split)
        result["seconds"] = time.monotonic() - started
        (work / "results" / f"{split}.json").write_text(json.dumps(result, indent=1) + "\n")
        print(split, "BF16 PPL on its own continuations", f"{result['ppl']:.6f}", flush=True)


# --- on the host --------------------------------------------------------------------

def run_container(work: Path, model: Path, image: str, gpu: str, phase: str, timeout: float) -> None:
    name = f"compression-bench-fidelity-{phase}-{work.name.lower()}".replace(".", "-")[:120]
    command = ["docker", "run", "--rm", "--name", name,
               "--gpus", f'"device={gpu}"', "--ipc=host", "--network=none",
               "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
               "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "VLLM_NO_USAGE_STATS=1",
               "-v", f"{work}:/work", "-v", f"{model_mount(model)[0]}:/model:ro", "--entrypoint", "python3", image,
               "/work/evaluator/build_fidelity_assets.py", "--inside", "/work", "--phase", phase,
               "--timeout", str(timeout)]
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def model_mount(model: Path) -> tuple[Path, str]:
    """(host directory to mount at /model, the checkpoint's path inside the container). A
    Hugging Face cache snapshot's files link into ../../blobs: the whole model repository is
    mounted so they resolve."""
    if model.parent.name == "snapshots":
        return model.parent.parent, f"/model/snapshots/{model.name}"
    return model, "/model"


def install(work: Path, tasks: list[Path]) -> None:
    for split in SPLITS:
        for task in tasks:
            target = task / "assets" / split
            for name in ("ppl_inputs.json", "ppl_config.json"):
                shutil.copy2(work / "new" / split / name, target / name)
    print(f"installed {work.name} into {len(tasks)} task(s); next: taskctl set --ppl-protocol ... and render")


def host(args) -> int:
    sys.path.insert(0, str(REPO / "compression_tasks" / "_manage"))
    sys.path.insert(0, str(REPO / "compression_bench" / "harness"))
    sys.path.insert(0, str(FINAL_VALIDATE))
    import render
    import task_policy
    from lifecycle import gpus
    tasks = [Path(t).resolve() for t in args.tasks]
    if args.install_from:
        install(Path(args.install_from).resolve(), tasks)
        return 0
    contracts = [render.load_contract(t) for t in tasks]
    if len({c["model"]["id"] for c in contracts}) != 1:
        raise SystemExit("the tasks do not share one model")
    if len({_sha((t / "assets" / "tokenizer" / "tokenizer.json").read_bytes()) for t in tasks}) != 1:
        raise SystemExit("the tasks do not share one tokenizer")
    contract, first = contracts[0], tasks[0]
    current = {split: json.loads((first / "assets" / split / "ppl_config.json").read_text()) for split in SPLITS}
    current_protocol = current["public"]["protocol"]
    if len({json.loads((t / "assets" / "public" / "ppl_config.json").read_text())["protocol"] for t in tasks}) != 1:
        raise SystemExit("the tasks' perplexity protocols differ")
    chat_length = task_policy.chat_prefix_tokens(current_protocol)
    label = task_policy.PPL_PROTOCOLS[current_protocol]["prefix_label"] if chat_length else None
    if label not in TARGETS:
        raise SystemExit(f"no protocol for the {label} chat prefix")
    target = TARGETS[label]
    chat = list(current["public"]["prefix_tokens"]) if chat_length else []
    if len(chat) != task_policy.chat_prefix_tokens(target):
        raise SystemExit("the chat prefix does not have the target protocol's length")
    prompt_tokens = task_policy.PPL_PROTOCOLS[target]["prefix_tokens"]
    continuation_tokens = CONTEXT - prompt_tokens
    model = Path(subprocess.check_output([sys.executable, str(REPO / "compression_bench" / "scripts" / "model_for_task.py"),
                                          str(first)], text=True).strip()).resolve()
    image = render.runtime_image(contract)
    if args.resume:
        # An earlier build whose continuations exist: the evaluator is copied afresh and
        # only the score phase runs again.
        work = Path(args.resume).resolve()
        stamp = work.name.rsplit("-", 1)[-1]
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        work = REPO / "compression_bench" / "evolve_space" / "baselines" / f"fidelity-dolmino-{contract['model']['local_name']}-{stamp}"
    generated = all((work / "new" / split / name).is_file() for split in SPLITS
                    for name in ("ppl_inputs.json", "ppl_config.json"))
    if args.resume and not generated:
        raise SystemExit(f"{work} holds no generated continuations to score")
    for sub in ("evaluator", "submission", "results", *(f"source/{s}" for s in SPLITS)):
        (work / sub).mkdir(parents=True, exist_ok=True)
    for name in EVALUATOR:
        shutil.copy2(FINAL_VALIDATE / name, work / "evaluator" / name)
    shutil.copy2(HERE, work / "evaluator" / "build_fidelity_assets.py")
    (work / "evaluator" / "evaluator-assets").mkdir(exist_ok=True)
    (work / "evaluator" / "evaluator-assets" / "model_memory_policy.json").write_text(
        json.dumps(dict(render.model_memory_policy(contract), limit_gib=1024)) + "\n")
    if "," in str(args.gpu):
        # The census reads exactly one GPU worker; a BF16 checkpoint spread over several
        # cards is scored without it, as its baseline was (measure_bf16_baselines.py).
        runtime_py = work / "evaluator" / "inference_runtime.py"
        old = "        self.memory_policy = model_memory.read_policy(Path(__file__).resolve().parent)\n"
        text = runtime_py.read_text()
        if text.count(old) != 1:
            raise SystemExit("inference_runtime.py does not read its census policy where expected")
        runtime_py.write_text(text.replace(old, "        self.memory_policy = None\n"))
    policy = dict(render.task_policy(contract), size_limit_gib=1024, ppl_protocol=target)
    (work / "evaluator" / "evaluator-assets" / "task_policy.json").write_text(json.dumps(policy) + "\n")
    cards = [int(g) for g in str(args.gpu).split(",")] if args.gpu is not None else []
    starter = (render.TEMPLATE / "starters" / f"inference.{contract['starter']}.py").read_text()
    if len(cards) > 1:
        if starter.count("tensor_parallel_size=1") != 1:
            raise SystemExit("the starter does not set tensor_parallel_size=1 exactly once")
        starter = starter.replace("tensor_parallel_size=1", f"tensor_parallel_size={len(cards)}")
    (work / "submission" / "inference.py").write_text(starter)
    if not (work / "submission" / "model").is_symlink():
        (work / "submission" / "model").symlink_to(model_mount(model)[1])
    (work / "build.json").write_text(json.dumps({"prompt_tokens": prompt_tokens,
                                                 "continuation_tokens": continuation_tokens}) + "\n")
    sources = {}
    for split in SPLITS:
        if args.resume:
            sources[split] = json.loads((work / "source" / split / "source.json").read_text())
            continue
        contexts, sources[split] = dolmino_contexts(first / "assets" / "tokenizer", split, prompt_tokens, chat,
                                                    Path.home() / ".cache" / "compression-bench-dolmino")
        (work / "source" / split / "contexts.json").write_text(json.dumps(contexts, separators=(",", ":")) + "\n")
        # The documents the contexts hold stay host-side (gitignored): prepare_datasets.py keeps every
        # overlapping row out of the agent's training data.
        (work / "source" / split / "documents.jsonl").write_text(
            "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in sources[split].pop("documents_text")))
        (work / "source" / split / "source.json").write_text(json.dumps(sources[split]) + "\n")
        print(split, f"{len(contexts)} contexts from {sources[split]['problems_used']} of "
              f"{sources[split]['problems']} Dolmino {sources[split]['split']} documents", flush=True)
    if args.contexts_only:
        print(f"contexts written under {work}/source (no GPU used)")
        return 0
    claims = [gpus.Claim(card, run_id=work.name) for card in cards]
    for claim in claims:
        if not claim.acquire():
            for held in claims:
                held.release()
            raise SystemExit(f"GPU {claim.index} is claimed by a compression-bench run")
    try:
        if not generated:
            run_container(work, model, image, ",".join(map(str, cards)), "generate", args.timeout)
        for split in SPLITS if not generated else ():
            rows = json.loads((work / "new" / split / "ppl_inputs.json").read_text())
            cfg = {"split": split, "blocks": BLOCKS, "context_length": CONTEXT, "input_tokens": BLOCKS * CONTEXT,
                   "vocab_size": current[split]["vocab_size"], "tokenizer_class": sources[split]["tokenizer_class"]}
            if chat:
                cfg["prefix_tokens"] = chat
            cfg.update(protocol=target, prompt_tokens=prompt_tokens, continuation_tokens=continuation_tokens,
                       scored_tokens=len(rows) * continuation_tokens,
                       source={"contexts": {k: v for k, v in sources[split].items() if k != "problems_per_block"},
                               "contexts_sha256": _sha((work / "source" / split / "contexts.json").read_bytes()),
                               "problems_per_block": f"{work.relative_to(REPO)}/source/{split}/source.json",
                               "continued_by": {"model": contract["model"]["id"],
                                                "revision": contract["model"]["revision"],
                                                "runtime": f"starter inference.{contract['starter']}.py, {image}",
                                                "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0,
                                                             "min_p": 0.0, "ignore_eos": True,
                                                             "seed": f"{SEED_BASE[split]} + block index"}}},
                       inputs_sha256=_sha((work / "new" / split / "ppl_inputs.json").read_bytes()))
            (work / "new" / split / "ppl_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        run_container(work, model, image, ",".join(map(str, cards)), "score", args.timeout)
    finally:
        for claim in claims:
            claim.release()
    for split in SPLITS:
        result = json.loads((work / "results" / f"{split}.json").read_text())
        path = work / "new" / split / "ppl_config.json"
        cfg = json.loads(path.read_text())
        cfg.update(bf16_ppl=result["ppl"], baseline_status="measured_new_protocol",
                   baseline_evidence={"measured_utc": stamp, "seconds": round(result["seconds"], 1),
                                      "result": str((work / "results" / f"{split}.json").relative_to(REPO))})
        path.write_text(json.dumps(cfg, indent=2) + "\n")
        print(f"{split}: BF16 PPL on its own continuations {result['ppl']:.6f}")
    if args.install:
        install(work, tasks)
    print(f"built {work}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tasks", nargs="*")
    parser.add_argument("--gpu", help="a card, or cards for tensor parallelism (2,7)")
    parser.add_argument("--contexts-only", action="store_true",
                        help="draw and tokenize the contexts, write them, and stop (no GPU)")
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--install-from", help="copy an earlier build's blocks into the tasks")
    parser.add_argument("--resume", help="score an earlier build's continuations again (its directory)")
    parser.add_argument("--inside", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--phase", choices=("generate", "score"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.inside:
        if args.phase == "generate":
            inside_generate(args.inside)
        else:
            inside_score(args.inside, args.timeout)
        return 0
    if not args.tasks or (args.gpu is None and not args.install_from and not args.contexts_only):
        parser.error("tasks and --gpu (or --install-from) are required")
    return host(args)


if __name__ == "__main__":
    sys.exit(main())
