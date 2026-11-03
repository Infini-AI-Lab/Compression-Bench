"""The hidden correctness gate's measurement: generated answers, verdict only.

The items are task assets that exist only in the hidden Verifier's copy (built by
compression_bench/scripts/deploy/build_accuracy_assets.py); the public Checker carries this
module but no items, and never runs it. Every item is a prompt the submission
continues by greedy generation (`generate`, at most the benchmark's
`max_new_tokens`), and its answer is judged from the decoded text by the
benchmark's `kind`:

  final-number  the text up to the next `Question:`; its number after the last
                `####`, else its last number, equals the reference number
  short-answer  the first line of the text, normalised (lower case; punctuation,
                articles and extra spaces removed), equals a normalised alias
  unit-tests    the prompt, the text up to its first top-level statement
                (STOP_CODE), the benchmark's tests and `check(<entry point>)`
                run as one Python program, which exits 0 within EXEC_TIMEOUT_S
                and prints a fresh random token after `check` returned (a
                completion that ends the process early passes nothing)

The program runs in a child process of the Verifier's own interpreter, isolated
(`-I`), in an empty temporary directory, with a minimal environment, CPU time,
memory and file-size limits, and is killed with its process group at the time
limit. It is no more trusted than the submission's `inference.py`, which runs in
the same container, and it runs after the submission's engine has stopped.

The accuracy is the mean of the per-benchmark accuracies; `chance` is 0 (an
answer is generated, not chosen).
"""
import concurrent.futures
import decimal
import hashlib
import json
import math
import os
import re
import secrets
import string
import subprocess
import sys
import tempfile
from pathlib import Path

PROTOCOL = 'hidden-gen-gate-v2'
KINDS = ('final-number', 'short-answer', 'unit-tests')
# The PPL stage's runtime options: one engine configuration serves both stages.
MAX_MODEL_LEN = 2049
MAX_NUM_SEQS = 32
GPU_MEMORY_UTILIZATION = 0.9
EXEC_TIMEOUT_S = 10
EXEC_WORKERS = 8
EXEC_MEMORY_BYTES = 4 * 2**30
#: A completion ends at the first top-level statement after the function body.
STOP_CODE = ('\ndef ', '\nclass ', '\nif __name__', '\nprint(', '\n#', '\n```', '\nassert ')
_NUMBER = re.compile(r'-?\d[\d,]*(?:\.\d+)?')
_MARKED = re.compile(r'####\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)')
_ARTICLES = re.compile(r'\b(a|an|the)\b')


def load_inputs(assets, *, expected_split):
    assets = Path(assets)
    cfg = json.loads((assets / 'accuracy_config.json').read_text())
    if cfg.get('protocol') != PROTOCOL:
        raise ValueError('wrong accuracy protocol')
    if cfg.get('split') != expected_split:
        raise ValueError('wrong accuracy split')
    data = (assets / 'accuracy_inputs.json').read_bytes()
    if hashlib.sha256(data).hexdigest() != cfg.get('inputs_sha256'):
        raise ValueError('accuracy inputs differ from their recorded digest')
    payload = json.loads(data)
    if payload.get('protocol') != PROTOCOL or payload.get('split') != expected_split:
        raise ValueError('accuracy inputs carry the wrong protocol or split')
    benchmarks = cfg['benchmarks']
    if not benchmarks or any(b.get('kind') not in KINDS for b in benchmarks):
        raise ValueError('invalid accuracy benchmark list')
    items, vocab = payload['items'], cfg['vocab_size']
    counts = [0] * len(benchmarks)
    for item in items:
        b, prompt, answer = item['b'], item['prompt'], item['answer']
        if type(b) is not int or not 0 <= b < len(benchmarks):
            raise ValueError('accuracy item names an unknown benchmark')
        if not prompt or any(type(t) is not int or not 0 <= t < vocab for t in prompt):
            raise ValueError('invalid accuracy prompt token IDs')
        if len(prompt) + benchmarks[b]['max_new_tokens'] > MAX_MODEL_LEN:
            raise ValueError('accuracy prompt and answer exceed the runtime context')
        kind = benchmarks[b]['kind']
        needed = {'final-number': ('number',), 'short-answer': ('aliases',),
                  'unit-tests': ('prompt', 'test', 'entry_point')}[kind]
        if not isinstance(answer, dict) or any(not answer.get(k) for k in needed):
            raise ValueError('accuracy item lacks its reference answer')
        counts[b] += 1
    if counts != [bench['items'] for bench in benchmarks] or len(items) != cfg['items']:
        raise ValueError('accuracy inputs do not match their configuration')
    return cfg, items


def _number(text):
    try:
        return decimal.Decimal(text.replace(',', '').rstrip('.'))
    except (decimal.InvalidOperation, ValueError):
        return None


def final_number(text):
    """The answer number of a worked solution, or None."""
    cut = re.split(r'\n\s*Question:', text, maxsplit=1)[0]
    marked = _MARKED.findall(cut)
    found = marked[-1] if marked else (_NUMBER.findall(cut) or [None])[-1]
    return None if found is None else _number(found)


def normalise(text):
    text = ''.join(c for c in text.lower() if c not in set(string.punctuation))
    return ' '.join(_ARTICLES.sub(' ', text).split())


def short_answer(text):
    return normalise(text.strip().split('\n', 1)[0])


def completion_code(text):
    end = min([i for i in (text.find(stop) for stop in STOP_CODE) if i >= 0], default=len(text))
    return text[:end]


def program(answer, text, token=''):
    done = f"\nprint({token!r})\n" if token else '\n'
    return (answer['prompt'] + completion_code(text) + '\n\n' + answer['test']
            + f"\n\ncheck({answer['entry_point']})" + done)


def _limits():
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (EXEC_TIMEOUT_S, EXEC_TIMEOUT_S))
    resource.setrlimit(resource.RLIMIT_AS, (EXEC_MEMORY_BYTES, EXEC_MEMORY_BYTES))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 2**20, 16 * 2**20))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def passes(answer, text):
    """Whether the item's program runs to exit 0 within the limits and reaches its end."""
    token = secrets.token_hex(16)
    source = program(answer, text, token)
    with tempfile.TemporaryDirectory(prefix='gate-exec-') as work:
        path = Path(work) / 'program.py'
        path.write_text(source)
        env = {'PATH': '/usr/bin:/bin', 'HOME': work, 'TMPDIR': work, 'PYTHONHASHSEED': '0',
               'PYTHONDONTWRITEBYTECODE': '1'}
        process = subprocess.Popen([sys.executable, '-I', str(path)], cwd=work, env=env,
                                   stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL, start_new_session=True,
                                   preexec_fn=_limits)
        try:
            out, _ = process.communicate(timeout=EXEC_TIMEOUT_S + 5)
            return process.returncode == 0 and out.decode('utf-8', 'replace').split()[-1:] == [token]
        except subprocess.TimeoutExpired:
            return False
        finally:
            try:
                os.killpg(process.pid, 9)
            except (ProcessLookupError, PermissionError):
                pass
            process.wait()


def judge(kind, answer, text, results=None):
    """Whether one generated text answers its item (unit tests: `results` holds the run)."""
    if kind == 'final-number':
        got = final_number(text)
        return got is not None and got == _number(answer['number'])
    if kind == 'short-answer':
        return short_answer(text) in {normalise(a) for a in answer['aliases']}
    return bool(results)


def score(cfg, items, texts, executed=None):
    """Accuracy from the decoded generations ({item index: text}); no GPU.
    `executed` ({item index: bool}) holds the unit-test runs; missing ones run here."""
    benchmarks = cfg['benchmarks']
    executed = dict(executed or {})
    pending = [i for i, item in enumerate(items)
               if benchmarks[item['b']]['kind'] == 'unit-tests' and i not in executed]
    with concurrent.futures.ThreadPoolExecutor(max_workers=EXEC_WORKERS) as pool:
        for i, ok in zip(pending, pool.map(lambda i: passes(items[i]['answer'], texts[i]), pending)):
            executed[i] = ok
    correct = [0] * len(benchmarks)
    verdicts = [[] for _ in benchmarks]
    for i, item in enumerate(items):
        kind = benchmarks[item['b']]['kind']
        ok = judge(kind, item['answer'], texts[i], executed.get(i))
        correct[item['b']] += ok
        verdicts[item['b']].append(bool(ok))
    per = {bench['name']: {'items': bench['items'], 'correct': correct[k],
                           'accuracy': correct[k] / bench['items'], 'chance': 0.0, 'kind': bench['kind']}
           for k, bench in enumerate(benchmarks)}
    accuracy = sum(row['accuracy'] for row in per.values()) / len(per)
    return {'accuracy': accuracy, 'chance': 0.0, 'benchmarks': per,
            'correct_items': {bench['name']: verdicts[k] for k, bench in enumerate(benchmarks)}}


def measure(submission, assets, runtime_class, *, tokenizer_dir, timeout=600, expected_split):
    from transformers import AutoTokenizer
    cfg, items = load_inputs(assets, expected_split=expected_split)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True,
                                              trust_remote_code=False, use_fast=True)
    generated, calls = {}, 0
    with runtime_class(submission, max_model_len=MAX_MODEL_LEN, max_num_seqs=MAX_NUM_SEQS,
                       gpu_memory_utilization=GPU_MEMORY_UTILIZATION, timeout=timeout) as runtime:
        for b, bench in enumerate(cfg['benchmarks']):
            chosen = [i for i, item in enumerate(items) if item['b'] == b]
            calls += 1
            outputs = runtime.generate([items[i]['prompt'] for i in chosen],
                                       max_new_tokens=bench['max_new_tokens'], vocab_size=cfg['vocab_size'])
            if len(outputs) != len(chosen):
                raise ValueError('missing accuracy generations')
            for i, ids in zip(chosen, outputs, strict=True):
                if len(ids) > bench['max_new_tokens']:
                    raise ValueError('an accuracy generation is longer than requested')
                generated[i] = list(ids)
        memory = runtime.memory_report()
    texts = {i: tokenizer.decode(ids, skip_special_tokens=True) for i, ids in generated.items()}
    result = score(cfg, items, texts)
    return dict(valid=True, model_memory=memory, accuracy_protocol=PROTOCOL,
                accuracy_split=cfg['split'], items=len(items), calls=calls,
                generated_tokens=sum(len(ids) for ids in generated.values()),
                input_tokens=sum(len(item['prompt']) for item in items),
                max_model_len=MAX_MODEL_LEN, max_num_seqs=MAX_NUM_SEQS,
                gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
                generations=[texts[i] for i in range(len(items))], **result)


def validate_result(result, assets, *, expected_split):
    """Do not score a stale result or mix public and hidden measurements."""
    cfg = json.loads((Path(assets) / 'accuracy_config.json').read_text())
    if (result.get('valid') is not True or result.get('accuracy_protocol') != PROTOCOL
            or result.get('accuracy_split') != expected_split or cfg.get('split') != expected_split
            or result.get('items') != cfg.get('items')):
        raise ValueError('accuracy result uses the wrong protocol, split or item count')
    for name in ('accuracy', 'chance'):
        value = result.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'invalid {name} in the accuracy result')
