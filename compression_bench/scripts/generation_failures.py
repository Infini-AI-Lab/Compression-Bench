#!/usr/bin/env python3
r"""How the hidden accuracy gate's generations fail: a behavioral taxonomy of every wrong item, per verdict.

    "$COMPRESSION_BENCH_PY" compression_bench/scripts/generation_failures.py [--data DIR] [--runs RUN ...]
                                                     # writes DIR/generation_failures.json

Inputs: the selected runs of the version ledger (`DIR/version_ledger.json`, version_ledger.py; DIR defaults to
evaluation_progress/trajectory_analysis/) and every final verification the ledger records for them: the final verdict (`<lineage>/FINAL_VALIDATION/<run>/accuracy.json`) and every
checkpoint verdict (`<lineage>/checkpoint_verdicts.jsonl: evidence`), deduplicated by content identity (one verdict
of the same bytes measured once), plus BF16's own hidden generations (`accuracy_config.json: bf16.evidence`). The
public Checker's results ("reward feedback") and the harness probe (8 items, not a verdict) are not used.

Everything that reads a hidden item or a generation runs in ONE no-network container (`--image`, the Verifier's own
image by default; `--network=none`, 4 CPUs, 16 GiB, read-only root and repository, no GPU), which re-scores every
generation with the evaluator's own judge (`final_validate/accuracy_benchmark.py`, protocol hidden-gen-gate-v2,
imported unchanged) and executes every HumanEval program there, never on the host. The container returns per-item
categories, flags and item indices only; this script writes nothing else. No prompt, generation, answer or test
leaves the container.

Judged span (exactly what the judge reads):
  math (GSM8K, final-number)   the text up to the first `\n\s*Question:`; answer = last `#### n`, else the last number
  code (HumanEval, unit-tests) `completion_code(text)`: the text up to the first STOP_CODE marker
  knowledge (TriviaQA)         the first line of the stripped text, normalised

Flags (computed for every item, right or wrong):
  leak          a chat-template or reasoning tag in the generation: any non-special added token of the task's
                tokenizer (`<think>`, `</think>`, `<tool_call>`, `<|fim_*|>`, ...; they survive `skip_special_tokens`),
                a literal `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, `<s>`, `</s>`, `[INST]`, `[/INST]`, or a role line
                (a line that is exactly `assistant`, `user` or `system`: what a skipped `<|im_start|>` leaves behind).
                Tags and whitespace at the very start of the generation (e.g. an empty `<think></think>` block,
                which BF16 emits as well) are a `leading_tag_prefix` flag, not a leak.
  leak_in_span  a leak inside the judged span; for math, before the judge's answer number (the answer was read
                from the text after the leak) or anywhere in the span when no number was found.
  loop          a repetition loop in the judged span: a block of 1-10 identical non-blank lines repeated >= 3 times in
                a row covering >= 30 characters; or a unit of 1-50 words (containing a word of >= 2 letters)
                repeated >= 3 times in a row covering >= 15 words, or >= 5 times covering >= 5 words; or one line of
                >= 20 characters occurring >= 4 times in the span.
  at_limit      the generation reached `max_new_tokens`: its re-encoded length (task tokenizer, no special tokens)
                is >= max_new_tokens - max(2, ceil(3% of it)) (the decoded text drops skipped special tokens, so
                the re-encoded length can fall slightly short of the generated one).
  unterminated  the judged span runs to the end of the generation (no `Question:` cut / STOP_CODE / newline).
  truncated     at_limit and unterminated: the answer was still being written when generation stopped.

Category of a wrong item (first rule that matches):
  1 tag_leak     leak_in_span
  2 repetition   loop
  3 truncated    truncated
  4 empty        nothing in the judged span (code: no statement beyond comments/blank lines; knowledge: an
                 empty normalised first line; math: a blank span); flag `blank`: the whole generation is
                 whitespace (generation ended at once)
  5 content      code: by executing the judge's program in the container with a reporting wrapper:
                   syntax_error (does not compile), runtime_error (an exception raised in the prompt/completion
                   code), tests_fail (an exception, normally an AssertionError, raised in the benchmark's own test
                   lines: the completion's result was rejected), timeout (wall or CPU limit), early_exit (the
                   process ended without finishing check, e.g. exit()), other_fail (the wrapper ran clean but the
                   judge failed it: recorded as exec_mismatch)
                 math: no_number (no number in the span), unmarked_stop (no `####` line: the solution ended, not at
                   the token limit, before its final-answer line and the judge's fallback last number is wrong),
                   wrong_number (a `####` answer, wrong)
                 knowledge: answer_with_extra_text (a normalised alias is a whole-word part of the first line),
                   wrong_answer
  Every wrong code item also records its execution outcome (`exec`), whatever its category.

Diagnostic re-scoring (NOT the official accuracy; the same judge on a modified generation):
  cut_at_leak     cut each generation at its first leak (after the leading tag prefix)
  cut_loop        cut each generation with a loop in its judged span at the start of the loop's second copy
  keep_helpers    code: let the completion run through top-level `def`/`class`/`#` lines (stop only at
                  `\nif __name__`, `\nprint(`, `\n```` `, `\nassert ` and a leak)
  alias_contained knowledge: count an answer whose normalised first line contains a normalised alias as a whole word
Each lists the items that flip wrong->right and right->wrong.

Within-run transitions: consecutive distinct verdicted versions of a run (in version order), with hidden PPL, the
items that flipped per benchmark and the category counts on each side. Shared-with-BF16: per benchmark, the wrong
items BF16 also gets wrong, the compression-induced ones (BF16 right), and those BF16 gets wrong but the
submission right. A verdict whose accuracy stage never finished (a timeout) is "not measured", never zero.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))     # explicit: the container runs this file with -I
from analysis_common import DATA, PROTOTYPES, REPO, lineage_checkpoint_verdicts, task_dir  # noqa: E402

IMAGE = "compression_bench/platform-gguf:v1"
BENCH = ("math", "code", "knowledge")
BENCH_NAME = {"math": "GSM8K", "code": "HumanEval", "knowledge": "TriviaQA"}
CATEGORIES = {
    "code": ("tag_leak", "repetition", "truncated", "empty", "syntax_error", "runtime_error", "tests_fail",
             "timeout", "early_exit", "other_fail"),
    "math": ("tag_leak", "repetition", "truncated", "empty", "no_number", "unmarked_stop", "wrong_number"),
    "knowledge": ("tag_leak", "repetition", "truncated", "empty", "answer_with_extra_text", "wrong_answer"),
}
DIAGNOSTICS = ("cut_at_leak", "cut_loop", "keep_helpers", "alias_contained")


# ----------------------------------------------------------------------------------------------------------------
# Worker: runs inside the container. Reads hidden items and generations; prints categories, flags and indices only.
# ----------------------------------------------------------------------------------------------------------------

def worker(jobs_path: str, repo: str, workers: int) -> None:
    import concurrent.futures
    import hashlib
    import math
    import re
    import secrets
    sys.path.insert(0, f"{repo}/compression_bench/harness/final_validate")
    import accuracy_benchmark as ab
    from transformers import AutoTokenizer

    role_line = re.compile(r"(?m)^(assistant|user|system)[ \t]*$")
    generic_tags = ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<s>", "</s>", "[INST]", "[/INST]")
    marked = ab._MARKED
    number = ab._NUMBER
    helper_stops = ("\nif __name__", "\nprint(", "\n```", "\nassert ")

    def leading_prefix_end(text, tags):
        """End of the run of tags and whitespace at the very start of the generation (0 when it opens with no tag)."""
        pos, consumed = 0, False
        while True:
            q = len(text) - len(text[pos:].lstrip())
            hit = next((t for t in tags if text.startswith(t, q)), None)
            if hit is None:
                return q if consumed else 0
            pos, consumed = q + len(hit), True

    def find_leaks(text, tags, start):
        found = []
        for t in tags:
            i = text.find(t, start)
            while i >= 0:
                found.append((i, t))
                i = text.find(t, i + 1)
        for m in role_line.finditer(text, start):
            found.append((m.start(), "role_line"))
        return sorted(found)

    def tandem(units, max_p, qualifies):
        """Every maximal run of >= 3 consecutive copies of a unit of p items that `qualifies(start, p, copies)`:
        the earliest such run as (start, p, copies), or None."""
        found = []
        n = len(units)
        for p in range(1, max_p + 1):
            i = 0
            while i + p < n:
                if units[i] != units[i + p]:
                    i += 1
                    continue
                j = i
                while j + p < n and units[j] == units[j + p]:
                    j += 1
                copies = (j - i + p) // p
                if copies >= 3 and qualifies(i, p, copies):
                    found.append((i, p, copies))
                i = j + 1
        return min(found) if found else None

    def loop_in(span):
        """(found, char position in the span where the loop's second copy starts)."""
        hits = []
        lines, offs, pos = [], [], 0
        for raw in span.split("\n"):
            if raw.strip():
                lines.append(raw.strip())
                offs.append(pos)
            pos += len(raw) + 1
        t = tandem(lines, 10, lambda i, p, c: sum(len(x) for x in lines[i:i + p * c]) >= 30)
        if t:
            hits.append(offs[t[0] + t[1]])
        words = [(m.group(0), m.start()) for m in re.finditer(r"\S+", span)]
        toks = [w for w, _ in words]
        t = tandem(toks, 50, lambda i, p, c: any(sum(ch.isalpha() for ch in w) >= 2 for w in toks[i:i + p])
                   and (p * c >= 15 or (c >= 5 and p * c >= 5)))
        if t:
            hits.append(words[t[0] + t[1]][1])
        seen = {}
        for line, off in zip(lines, offs):
            if len(line) >= 20:
                seen.setdefault(line, []).append(off)
        hits += [where[1] for where in seen.values() if len(where) >= 4]
        return (True, min(hits)) if hits else (False, None)

    def run_source(source, token):
        """The judge's own execution (`passes`), for a program assembled here."""
        with tempfile.TemporaryDirectory(prefix="gate-exec-") as work:
            path = Path(work) / "program.py"
            path.write_text(source)
            env = {"PATH": "/usr/bin:/bin", "HOME": work, "TMPDIR": work, "PYTHONHASHSEED": "0",
                   "PYTHONDONTWRITEBYTECODE": "1"}
            proc = subprocess.Popen([sys.executable, "-I", str(path)], cwd=work, env=env, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True,
                                    preexec_fn=ab._limits)
            try:
                out, _ = proc.communicate(timeout=ab.EXEC_TIMEOUT_S + 5)
                return proc.returncode == 0 and out.decode("utf-8", "replace").split()[-1:] == [token]
            except subprocess.TimeoutExpired:
                return False
            finally:
                try:
                    os.killpg(proc.pid, 9)
                except (ProcessLookupError, PermissionError):
                    pass
                proc.wait()

    wrapper_src = r'''
import os, sys, traceback
_fd = os.dup(1)
TOKEN, PATH, BODY = sys.argv[1], sys.argv[2], int(sys.argv[3])
def report(kind, name="-", where="-"):
    os.write(_fd, ("\n%s %s %s %s\n" % (TOKEN, kind, name, where)).encode())
src = open(PATH).read()
try:
    code = compile(src, "program.py", "exec")
except (SyntaxError, ValueError) as e:
    report("syntax_error", type(e).__name__); os._exit(0)
def where_of(e):
    frames = [f for f in traceback.extract_tb(e.__traceback__) if f.filename == "program.py"]
    return "test" if frames and frames[-1].lineno > BODY else "body"
try:
    exec(code, {"__name__": "__main__", "__builtins__": __builtins__})
except SystemExit as e:
    report("early_exit", "SystemExit")
except BaseException as e:
    w = where_of(e); report("tests_fail" if w == "test" else "runtime_error", type(e).__name__, w)
else:
    report("ok")
os._exit(0)
'''

    def outcome(answer, text):
        """Execution outcome of the judge's program for this generation (wrapper; same limits and timeout)."""
        source = ab.program(answer, text)
        body = (answer["prompt"] + ab.completion_code(text)).count("\n") + 1
        token = secrets.token_hex(16)
        with tempfile.TemporaryDirectory(prefix="gate-diag-") as work:
            prog = Path(work) / "program.py"
            prog.write_text(source)
            wrap = Path(work) / "wrapper.py"
            wrap.write_text(wrapper_src)
            env = {"PATH": "/usr/bin:/bin", "HOME": work, "TMPDIR": work, "PYTHONHASHSEED": "0",
                   "PYTHONDONTWRITEBYTECODE": "1"}
            proc = subprocess.Popen([sys.executable, "-I", str(wrap), token, str(prog), str(body)], cwd=work, env=env,
                                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                    start_new_session=True, preexec_fn=ab._limits)
            try:
                out, _ = proc.communicate(timeout=ab.EXEC_TIMEOUT_S + 5)
                rows = [line.split() for line in out.decode("utf-8", "replace").splitlines() if line.startswith(token)]
                if rows:
                    return rows[-1][1], rows[-1][2]
                return ("timeout", "SIGXCPU") if proc.returncode in (-24, -9) else ("early_exit", str(proc.returncode))
            except subprocess.TimeoutExpired:
                return "timeout", "wall"
            finally:
                try:
                    os.killpg(proc.pid, 9)
                except (ProcessLookupError, PermissionError):
                    pass
                proc.wait()

    cache_pass, cache_out = {}, {}

    def key(answer, text):
        return hashlib.sha256((answer["entry_point"] + "\0" + answer["test"] + "\0" + answer["prompt"] + "\0"
                               + text).encode()).hexdigest()

    def official_pass(answer, text):
        k = key(answer, ab.completion_code(text))
        if k not in cache_pass:
            cache_pass[k] = ab.passes(answer, text)
        return cache_pass[k]

    def exec_outcome(answer, text):
        k = key(answer, ab.completion_code(text))
        if k not in cache_out:
            cache_out[k] = outcome(answer, text)
        return cache_out[k]

    def helper_pass(answer, text, leak_at):
        end = min([i for i in (text.find(s) for s in helper_stops) if i >= 0] + [leak_at if leak_at is not None
                                                                                  else len(text)])
        completion = text[:end]
        k = "helpers:" + key(answer, completion)
        if k not in cache_pass:
            token = secrets.token_hex(16)
            source = (answer["prompt"] + completion + "\n\n" + answer["test"]
                      + f"\n\ncheck({answer['entry_point']})" + f"\nprint({token!r})\n")
            cache_pass[k] = run_source(source, token)
        return cache_pass[k]

    def judge_text(kind, answer, text):
        if kind == "unit-tests":
            return official_pass(answer, text)
        return ab.judge(kind, answer, text)

    jobs = json.loads(Path(jobs_path).read_text())
    assets = {}
    results = {}
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    for job in jobs:
        task = job["task"]
        tdir = Path(repo) / "compression_tasks" / f"{task}_standard_1xH200NVL" / "assets"
        if task not in assets:
            cfg, items = ab.load_inputs(tdir / "hidden", expected_split="hidden")
            tok = AutoTokenizer.from_pretrained(str(tdir / "tokenizer"), local_files_only=True,
                                                trust_remote_code=False, use_fast=True)
            added = json.loads((tdir / "tokenizer" / "tokenizer.json").read_text())["added_tokens"]
            tags = sorted({a["content"] for a in added if not a["special"]} | set(generic_tags), key=len, reverse=True)
            assets[task] = (cfg, items, tok, tags)
        cfg, items, tok, tags = assets[task]
        acc = json.loads((Path(repo) / job["accuracy"]).read_text())
        if not isinstance(acc.get("generations"), list) or acc.get("valid") is not True:
            results[job["id"]] = {"measured": False, "error": str(acc.get("error"))[:200]}
            continue
        texts = acc["generations"]
        benches = cfg["benchmarks"]
        within = []
        counter = [0] * len(benches)
        for it in items:
            within.append(counter[it["b"]])
            counter[it["b"]] += 1
        # official executions first (parallel), then wrapper outcomes for the failing code items
        code_idx = [i for i, it in enumerate(items) if benches[it["b"]]["kind"] == "unit-tests"]
        list(pool.map(lambda i: official_pass(items[i]["answer"], texts[i]), code_idx))
        fail_idx = [i for i in code_idx if not official_pass(items[i]["answer"], texts[i])]
        list(pool.map(lambda i: exec_outcome(items[i]["answer"], texts[i]), fail_idx))

        per_item = []
        reenc_total = 0
        for i, it in enumerate(items):
            bench = benches[it["b"]]
            name, kind, limit = bench["name"], bench["kind"], bench["max_new_tokens"]
            answer, text = it["answer"], texts[i]
            official = bool(acc["correct_items"][name][within[i]])
            ok = judge_text(kind, answer, text)
            n_tok = len(tok.encode(text, add_special_tokens=False))
            reenc_total += n_tok
            at_limit = n_tok >= limit - max(2, math.ceil(0.03 * limit))
            prefix = leading_prefix_end(text, [t for t in tags if t.startswith("<")])
            leaks = find_leaks(text, tags, prefix)
            first_leak = leaks[0][0] if leaks else None
            if kind == "final-number":
                parts = re.split(r"\n\s*Question:", text, maxsplit=1)
                span, terminated = parts[0], len(parts) > 1
                ms = list(marked.finditer(span))
                if ms:
                    answer_pos = ms[-1].start(1)
                else:
                    ns = list(number.finditer(span))
                    answer_pos = ns[-1].start() if ns else None
                span_end = len(span)
                leak_in_span = first_leak is not None and first_leak < span_end and (
                    answer_pos is None or first_leak < answer_pos)
                empty = not span.strip()
            elif kind == "unit-tests":
                span = ab.completion_code(text)
                terminated = len(span) < len(text)
                leak_in_span = first_leak is not None and first_leak < len(span)
                empty = not any(line.strip() and not line.strip().startswith("#") for line in span.split("\n"))
                answer_pos = None
            else:
                stripped = text.strip()
                span = stripped.split("\n", 1)[0]
                terminated = "\n" in stripped
                off = len(text) - len(text.lstrip())
                leak_in_span = first_leak is not None and first_leak < off + len(span)
                empty = ab.short_answer(text) == ""
                answer_pos = None
            loop, loop_at = loop_in(span)
            if kind == "short-answer":
                loop_at = None if loop_at is None else off + loop_at
            truncated = at_limit and not terminated
            rec = {"b": name, "j": within[i], "official": official, "rescored": bool(ok),
                   "flags": {"leak": bool(leaks), "leak_kinds": sorted({k for _, k in leaks}),
                             "leak_in_span": bool(leak_in_span), "leading_tag_prefix": prefix > 0,
                             "loop": loop, "at_limit": at_limit, "unterminated": not terminated,
                             "truncated": truncated, "blank": not text.strip()}}
            if kind == "final-number":
                rec["flags"]["unmarked"] = not marked.search(span)
            if kind == "short-answer":
                aliases = {ab.normalise(a) for a in answer["aliases"]} - {""}
                first = ab.short_answer(text)
                contained = any(re.search(r"(?:^| )" + re.escape(a) + r"(?:$| )", first) for a in aliases)
                rec["flags"]["alias_contained"] = bool(contained) and not ok
            if not official:
                if leak_in_span:
                    cat = "tag_leak"
                elif loop:
                    cat = "repetition"
                elif truncated:
                    cat = "truncated"
                elif empty:
                    cat = "empty"
                elif kind == "unit-tests":
                    cat = None
                elif kind == "final-number":
                    cat = ("no_number" if ab.final_number(text) is None
                           else "unmarked_stop" if rec["flags"]["unmarked"] else "wrong_number")
                else:
                    cat = "answer_with_extra_text" if rec["flags"]["alias_contained"] else "wrong_answer"
                if kind == "unit-tests":
                    ex, exname = exec_outcome(answer, text)
                    rec["exec"] = ex if ex != "ok" else "other_fail"
                    rec["exec_exception"] = exname
                    rec["exec_mismatch"] = ex == "ok"
                    if cat is None:
                        cat = rec["exec"]
                rec["category"] = cat
            # diagnostic re-scoring: same judge, modified generation
            diag = {}
            if first_leak is not None:
                diag["cut_at_leak"] = bool(judge_text(kind, answer, text[:first_leak]))
            if loop:
                diag["cut_loop"] = bool(judge_text(kind, answer, text[:loop_at]))
            if kind == "unit-tests" and not official:
                diag["keep_helpers"] = bool(helper_pass(answer, text, first_leak))
            if kind == "short-answer":
                diag["alias_contained"] = bool(ok or rec["flags"]["alias_contained"])
            rec["diag"] = diag
            per_item.append(rec)
        results[job["id"]] = {"measured": True, "generated_tokens": acc.get("generated_tokens"),
                              "reencoded_tokens": reenc_total, "items": per_item}
    pool.shutdown()
    json.dump(results, sys.stdout)


# ----------------------------------------------------------------------------------------------------------------
# Host: which generation sets to read, the container call, aggregation. Never reads an item or a generation.
# ----------------------------------------------------------------------------------------------------------------

def vnum(v: str) -> int:
    return int(v.lstrip("v"))


def collect(data: Path) -> tuple[list[dict], list[dict], dict]:
    ledger = json.loads((data / "version_ledger.json").read_text())
    jobs, runs, bf16 = [], [], {}
    for r in ledger["runs"]:
        lineage = PROTOTYPES / r["lineage"]
        rows = lineage_checkpoint_verdicts(lineage, r["run"], ("SCORED", "ZERO"))
        final_evidence = lineage / "FINAL_VALIDATION" / r["run"]
        verdicts = {}
        for v in sorted((v for v in r["versions"] if v.get("verdict")), key=lambda v: vnum(v["version"])):
            row = rows.get(v["version"])
            evidence = Path(row["evidence"]) if row else final_evidence
            evidence = evidence if evidence.is_absolute() else REPO / evidence
            ident = (row or {}).get("content_identity") or v.get("manifest_identity")
            entry = verdicts.setdefault(str(evidence), {"versions": [], "content_identity": ident,
                                                        "evidence": str(evidence.relative_to(REPO)),
                                                        "verdict_of": (row or {}).get("verdict_of")})
            entry["versions"].append(v["version"])
            entry.setdefault("ledger", v["verdict"])
        if str(final_evidence) not in verdicts:
            verdicts[str(final_evidence)] = {"versions": ["final"], "content_identity": None,
                                             "evidence": str(final_evidence.relative_to(REPO)), "verdict_of": "final",
                                             "ledger": {"status": r["final"]["status"], "ppl": r["final"]["ppl"],
                                                        "counts": r["final"]["counts"],
                                                        "failed_stage": r["final"].get("failed_stage")}}
        vlist = sorted(verdicts.values(), key=lambda e: min((vnum(v) for v in e["versions"] if v != "final"),
                                                            default=10 ** 6))
        for k, e in enumerate(vlist):
            e["id"] = f"{r['run']}#{k}"
            e["is_final"] = e["evidence"] == str(final_evidence.relative_to(REPO))
            jobs.append({"id": e["id"], "task": r["task"], "accuracy": e["evidence"] + "/accuracy.json"})
        runs.append({"row": r, "verdicts": vlist})
        if r["model"] not in bf16:
            cfg = json.loads((task_dir(r["task"]) / "assets" / "hidden" / "accuracy_config.json").read_text())
            ev = cfg["bf16"]["evidence"]
            bf16[r["model"]] = {"task": r["task"], "evidence": ev, "id": f"bf16:{r['model']}"}
            jobs.append({"id": f"bf16:{r['model']}", "task": r["task"], "accuracy": ev})
    return jobs, runs, bf16


def run_container(jobs: list[dict], image: str, workers: int, scratch: Path | None) -> dict:
    """The job list (ids and evidence paths, no hidden data) goes in through a temporary read-only mount."""
    with tempfile.TemporaryDirectory(prefix="genfail-", dir=scratch) as tmp:
        (Path(tmp) / "jobs.json").write_text(json.dumps(jobs))
        cmd = ["docker", "run", "--rm", "--network=none", "--cpus", "4", "--memory", "16g", "--pids-limit", "1024",
               "--read-only", "--tmpfs", "/tmp:rw,size=4g", "--user", f"{os.getuid()}:{os.getgid()}",
               "-e", "HOME=/tmp", "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
               "-e", "NVIDIA_VISIBLE_DEVICES=void", "-e", "CUDA_VISIBLE_DEVICES=",
               "-v", f"{REPO}:/repo:ro", "-v", f"{tmp}:/jobs:ro", "-w", "/tmp", image,
               "python3", "-I", "/repo/compression_bench/scripts/generation_failures.py", "--worker", "/jobs/jobs.json",
               "--repo", "/repo", "--workers", str(workers)]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-4000:])
        raise SystemExit(f"container failed ({proc.returncode})")
    return json.loads(proc.stdout)


def summarise(res: dict, bf16_res: dict | None) -> dict:
    """Per benchmark: counts, categories, indices, flags, diagnostics, BF16 overlap. Indices are within-benchmark."""
    if not res.get("measured"):
        return {"measured": False, "note": "accuracy stage did not finish: not measured (not zero accuracy)",
                "error": res.get("error")}
    out = {"measured": True, "generated_tokens": res["generated_tokens"], "reencoded_tokens": res["reencoded_tokens"],
           "benchmarks": {}}
    bf_items = {}
    if bf16_res and bf16_res.get("measured"):
        for it in bf16_res["items"]:
            bf_items[(it["b"], it["j"])] = it
    for b in BENCH:
        items = sorted((it for it in res["items"] if it["b"] == b), key=lambda it: it["j"])
        wrong = [it for it in items if not it["official"]]
        cats = {c: [it["j"] for it in wrong if it["category"] == c] for c in CATEGORIES[b]}
        flags = {}
        for f in ("leak", "leak_in_span", "leading_tag_prefix", "loop", "at_limit", "truncated", "blank", "unmarked",
                  "alias_contained"):
            if any(f in it["flags"] for it in items):
                flags[f] = {"wrong": [it["j"] for it in wrong if it["flags"].get(f)],
                            "right": [it["j"] for it in items if it["official"] and it["flags"].get(f)]}
        kinds = {}
        for it in items:
            for k in it["flags"]["leak_kinds"]:
                kinds.setdefault(k, 0)
                kinds[k] += 1
        row = {"items": len(items), "correct": sum(it["official"] for it in items),
               "rescored_correct": sum(it["rescored"] for it in items),
               "rescore_mismatch": [it["j"] for it in items if it["official"] != it["rescored"]],
               "wrong": [it["j"] for it in wrong],
               "categories": {c: len(v) for c, v in cats.items() if v},
               "items_by_category": {c: v for c, v in cats.items() if v},
               "flags": flags, "leak_kinds_items": kinds}
        if b == "code":
            ex = {}
            for it in wrong:
                ex.setdefault(it["exec"], []).append(it["j"])
            row["exec_outcomes"] = {k: len(v) for k, v in sorted(ex.items())}
            row["exec_items"] = {k: v for k, v in sorted(ex.items())}
            exc = {}
            for it in wrong:
                if it["exec"] == "runtime_error":
                    exc[it["exec_exception"]] = exc.get(it["exec_exception"], 0) + 1
            row["runtime_exceptions"] = exc
            row["exec_mismatch"] = [it["j"] for it in wrong if it.get("exec_mismatch")]
        diag = {}
        for d in DIAGNOSTICS:
            touched = [it for it in items if d in it["diag"]]
            if not touched:
                continue
            w2c = [it["j"] for it in touched if not it["official"] and it["diag"][d]]
            c2w = [it["j"] for it in touched if it["official"] and not it["diag"][d]]
            diag[d] = {"applied_to": len(touched), "wrong_to_right": w2c, "right_to_wrong": c2w,
                       "diagnostic_correct": row["correct"] + len(w2c) - len(c2w)}
        row["diagnostic_rescoring"] = diag
        if bf_items:
            bw = {j for (bb, j), it in bf_items.items() if bb == b and not it["official"]}
            mine = set(row["wrong"])
            shared = sorted(mine & bw)
            same = [j for j in shared if bf_items[(b, j)]["category"] == next(it["category"] for it in wrong
                                                                               if it["j"] == j)]
            row["vs_bf16"] = {"bf16_wrong": len(bw), "shared_wrong": shared, "compression_induced": sorted(mine - bw),
                              "bf16_only_wrong": sorted(bw - mine), "shared_same_category": len(same),
                              "induced_categories": _count(it["category"] for it in wrong if it["j"] not in bw),
                              "shared_categories": _count(it["category"] for it in wrong if it["j"] in bw)}
        out["benchmarks"][b] = row
    return out


def _count(xs) -> dict:
    out = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def transition(a: dict, b: dict, res_a: dict, res_b: dict) -> dict:
    row = {"from": a["versions"], "to": b["versions"], "hidden_ppl_from": a["ledger"].get("ppl"),
           "hidden_ppl_to": b["ledger"].get("ppl"), "benchmarks": {}}
    if row["hidden_ppl_from"] and row["hidden_ppl_to"]:
        row["hidden_ppl_delta"] = row["hidden_ppl_to"] - row["hidden_ppl_from"]
    if not (res_a.get("measured") and res_b.get("measured")):
        row["note"] = "a side was not measured"
        return row
    for bench in BENCH:
        ia = {it["j"]: it for it in res_a["items"] if it["b"] == bench}
        ib = {it["j"]: it for it in res_b["items"] if it["b"] == bench}
        c2w = sorted(j for j in ia if ia[j]["official"] and not ib[j]["official"])
        w2c = sorted(j for j in ia if not ia[j]["official"] and ib[j]["official"])
        moves = _count(f"{ia[j].get('category') or 'correct'}->{ib[j].get('category') or 'correct'}"
                       for j in ia if ia[j]["official"] != ib[j]["official"]
                       or ia[j].get("category") != ib[j].get("category"))
        row["benchmarks"][bench] = {
            "correct_from": sum(it["official"] for it in ia.values()),
            "correct_to": sum(it["official"] for it in ib.values()),
            "right_to_wrong": c2w, "wrong_to_right": w2c,
            "categories_from": _count(it["category"] for it in ia.values() if not it["official"]),
            "categories_to": _count(it["category"] for it in ib.values() if not it["official"]),
            "item_moves": moves}
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--worker")
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--image", default=IMAGE)
    ap.add_argument("--data", type=Path, default=DATA, help="the analysis directory (version_ledger.json in)")
    ap.add_argument("--out", type=Path, help="default: DATA/generation_failures.json")
    ap.add_argument("--scratch", type=Path, default=None, help="where the job list is staged (default: $TMPDIR)")
    ap.add_argument("--runs", nargs="*", help="only these run ids (default: every run of the ledger)")
    args = ap.parse_args(argv)
    if args.worker:
        worker(args.worker, args.repo, args.workers)
        return 0
    args.out = args.out or args.data / "generation_failures.json"
    jobs, runs, bf16 = collect(args.data)
    if args.runs:
        runs = [r for r in runs if r["row"]["run"] in args.runs]
        keep = {v["id"] for r in runs for v in r["verdicts"]} | {bf16[r["row"]["model"]]["id"] for r in runs}
        jobs = [j for j in jobs if j["id"] in keep]
        bf16 = {m: b for m, b in bf16.items() if b["id"] in keep}
    res = run_container(jobs, args.image, args.workers, args.scratch)
    out = {"about": "Behavioral taxonomy of the hidden accuracy gate's wrong items for every final verification of "
                    "the selected runs and BF16 (compression_bench/scripts/generation_failures.py; rules in its docstring). "
                    "Counts, categories, flags and within-benchmark item indices (0-49) only. 'diagnostic_rescoring' "
                    "is diagnostic re-scoring with the same judge on modified generations, NOT the official accuracy.",
           "protocol": "hidden-gen-gate-v2", "benchmarks": BENCH_NAME,
           "categories": {b: list(c) for b, c in CATEGORIES.items()},
           "execution": f"every HumanEval program executed in {args.image} with --network=none, 4 CPUs, 16 GiB, "
                        "read-only, no GPU, the evaluator's limits (10 s CPU, 4 GiB address space)",
           "bf16": {}, "runs": []}
    for model, b in bf16.items():
        out["bf16"][model] = {"task": b["task"], "evidence": b["evidence"], **summarise(res[b["id"]], None)}
    for run in runs:
        r = run["row"]
        bf = res[bf16[r["model"]]["id"]]
        entry = {k: r[k] for k in ("task", "model", "cap", "agent", "run", "lineage")}
        entry["final_status"] = r["final"]["status"]
        entry["bf16_hidden_ppl"] = r["final"].get("bf16_ppl")
        entry["verdicts"] = []
        for v in run["verdicts"]:
            s = summarise(res[v["id"]], bf)
            led = v["ledger"]
            ledger_counts = led.get("counts")
            if s.get("measured") and ledger_counts:
                s["counts_match_ledger"] = all(s["benchmarks"][b]["correct"] == ledger_counts[b] for b in BENCH)
            entry["verdicts"].append({"versions": v["versions"], "is_final": v["is_final"],
                                      "content_identity": v["content_identity"], "evidence": v["evidence"],
                                      "status": led.get("status"), "failed_stage": led.get("failed_stage"),
                                      "hidden_ppl": led.get("ppl"),
                                      "reward": (10 * entry["bf16_hidden_ppl"] / led["ppl"]
                                                 if entry["bf16_hidden_ppl"] and led.get("ppl") else None),
                                      **s})
        vs = run["verdicts"]
        entry["transitions"] = [transition(a, b, res[a["id"]], res[b["id"]]) for a, b in zip(vs, vs[1:])]
        out["runs"].append(entry)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {args.out.relative_to(REPO) if args.out.is_relative_to(REPO) else args.out}: "
          f"{sum(len(r['verdicts']) for r in out['runs'])} distinct verdicts, {len(out['bf16'])} BF16 baselines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
