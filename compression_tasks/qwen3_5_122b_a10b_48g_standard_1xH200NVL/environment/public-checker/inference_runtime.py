"""Submitted inference driver with independent code review.

PPL/reward are task-owned, but candidate outputs are NOT independently verified.
Container isolation and a child process are not a complete adversarial sandbox.
"""
import importlib.util
import json
import math
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time

# Load the evaluator-owned sibling by absolute path, never candidate sys.path.
_memory_spec = importlib.util.spec_from_file_location(
    "evaluator_model_memory", Path(__file__).resolve().with_name("model_memory.py"))
model_memory = importlib.util.module_from_spec(_memory_spec)
sys.modules[_memory_spec.name] = model_memory
_memory_spec.loader.exec_module(model_memory)

RUNTIME_PROTOCOL = 'submitted-inference-audited-v1'
API_VERSION = 1
MAX_REPLY_BYTES = 4 * 1024 * 1024


def review_metadata():
    return {'inference_protocol': RUNTIME_PROTOCOL, 'audit_status': 'pending',
            'review_required': True, 'score_status': 'provisional'}

class InferenceError(RuntimeError):
    pass


def read_token_stream(stream, count, vocab_size, started, clock=time.perf_counter):
    """Timestamp receipt in the trusted parent, not candidate-supplied timestamps."""
    tokens, arrivals, chunk_sizes, chunk_arrivals = [], [], [], []
    while True:
        line = stream.readline(MAX_REPLY_BYTES + 1)
        received = clock()
        if not line or len(line) > MAX_REPLY_BYTES or not line.endswith(b'\n'):
            raise InferenceError('missing or oversized stream response')
        reply = json.loads(line)
        if not isinstance(reply, dict) or reply.get('ok') is not True:
            raise InferenceError('submitted stream failed: ' + str(reply)[:2000])
        if reply.get('event') == 'tokens':
            chunk = reply.get('token_ids')
            # Every event must deliver one freshly generated token.
            # This validates delivery granularity, not the candidate's internal algorithm;
            # the ban on speculative decoding also requires implementation review.
            if not isinstance(chunk, list) or len(chunk) != 1:
                raise InferenceError('decode probe requires one token per event, not buffered chunks')
            if any(type(token) is not int or not 0 <= token < vocab_size for token in chunk):
                raise InferenceError('invalid streamed token ID')
            if len(tokens) + len(chunk) > count:
                raise InferenceError('too many streamed tokens')
            arrival = received - started
            if not math.isfinite(arrival) or arrival < 0 or (arrivals and arrival < arrivals[-1]):
                raise InferenceError('invalid stream timestamps')
            tokens.extend(chunk)
            # Retain event diagnostics; every accepted event has exactly one token.
            # Record actual parent receipt times without interpolation.
            arrivals.extend([arrival] * len(chunk))
            chunk_sizes.append(len(chunk))
            chunk_arrivals.append(arrival)
        elif reply.get('event') == 'done':
            if len(tokens) != count:
                raise InferenceError('incomplete token stream')
            duration = arrivals[-1] - arrivals[0]
            if duration <= 0 or any(b < a for a, b in zip(arrivals, arrivals[1:])):
                raise InferenceError('invalid stream timestamps')
            return {'token_ids': tokens, 'arrival_seconds': arrivals,
                    'chunk_sizes': chunk_sizes, 'chunk_arrival_seconds': chunk_arrivals,
                    'time_to_first_token_seconds': arrivals[0],
                    'decode_seconds': duration, 'decode_intervals': count - 1,
                    'decode_tokens_per_second': (count - 1) / duration}
        else:
            raise InferenceError('unknown stream event')


#: How long cleanup waits for a SIGKILLed worker to be reaped.
KILL_WAIT_SECONDS = 300


class SubmittedRuntime:
    def __init__(self, submission, *, max_model_len, max_num_seqs=8,
                 gpu_memory_utilization=0.80, timeout=600):
        self.submission = Path(submission).resolve()
        self.options = dict(max_model_len=max_model_len, max_num_seqs=max_num_seqs,
                            gpu_memory_utilization=gpu_memory_utilization)
        self.timeout = timeout
        self.process = None
        self.log = None
        self.work = None
        self.memory_policy = model_memory.read_policy(Path(__file__).resolve().parent)
        self.memory_snapshots = []

    def memory_report(self):
        return model_memory.report(self.memory_snapshots, self.memory_policy)

    def __enter__(self):
        env = os.environ.copy()
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', VLLM_NO_USAGE_STATS='1',
                   VLLM_USE_FLASHINFER_SAMPLER='0', PYTHONDONTWRITEBYTECODE='1')
        self.log = tempfile.TemporaryFile()
        self.work = tempfile.TemporaryDirectory(prefix='quant-inference-work-')
        env.update(TMPDIR=self.work.name, HOME=self.work.name, XDG_CACHE_HOME=self.work.name)
        try:
            self.process = subprocess.Popen(
                [sys.executable, '-I', '-B', str(Path(__file__).resolve()), '--worker', str(self.submission)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                cwd=self.work.name, env=env, start_new_session=True)
            reply = self._request('load', options=self.options,
                                  inspect_model_memory=self.memory_policy is not None)
            if reply != RUNTIME_PROTOCOL:
                raise InferenceError('unsupported checker runtime protocol')
            return self
        except BaseException:
            self.close()
            raise

    def _request(self, operation, *, _stream_count=None, _vocab_size=None,
                 _reply_limit=MAX_REPLY_BYTES, **kwargs):
        results = queue.Queue(maxsize=1)

        def exchange():
            try:
                data = json.dumps(dict(operation=operation, **kwargs), allow_nan=False).encode() + b'\n'
                started = time.perf_counter()
                self.process.stdin.write(data)
                self.process.stdin.flush()
                if _stream_count is not None:
                    value = read_token_stream(self.process.stdout, _stream_count, _vocab_size, started)
                    results.put((True, value))
                    return
                line = self.process.stdout.readline(_reply_limit + 1)
                if not line or len(line) > _reply_limit or not line.endswith(b'\n'):
                    raise InferenceError('missing or oversized inference response')
                reply = json.loads(line)
                if not isinstance(reply, dict) or reply.get('ok') is not True:
                    raise InferenceError('submitted inference failed: ' + str(reply)[:2000])
                results.put((True, reply['result']))
            except BaseException as exc:
                results.put((False, exc))

        threading.Thread(target=exchange, daemon=True).start()
        try:
            success, value = results.get(timeout=self.timeout)
        except queue.Empty as exc:
            self.close()
            raise InferenceError(f'inference {operation} timed out after {self.timeout}s') from exc
        if not success:
            self.log.seek(0, os.SEEK_END)
            self.log.seek(max(0, self.log.tell() - 2000))
            detail = self.log.read().decode(errors='replace')
            raise InferenceError(f'{value}\n{detail}') from value
        if self.memory_policy is not None and operation in (
                "load", "generate", "token_logprobs", "stream_generate"):
            # Separate RPC AFTER completion: never charge the census to token latency.
            snapshot = self._request("model_memory")
            checked = model_memory.check_snapshot(snapshot, self.memory_policy, operation)
            self.memory_snapshots.append(checked)
        return value

    def generate(self, token_ids, *, max_new_tokens, vocab_size, ignore_eos=False):
        # Bound the JSON reply by the requested batch and token budget.
        # A 64 x 8192-token result can exceed the default 4 MiB limit.
        reply_limit = max(MAX_REPLY_BYTES,
                          len(token_ids) * (max_new_tokens * (len(str(vocab_size)) + 2) + 2) + 1024)
        result = self._request('generate', _reply_limit=reply_limit, token_ids=token_ids,
                               max_new_tokens=max_new_tokens, ignore_eos=ignore_eos)
        if not isinstance(result, list) or len(result) != len(token_ids):
            raise InferenceError('generation batch size mismatch')
        for row in result:
            if not isinstance(row, list) or not 0 < len(row) <= max_new_tokens:
                raise InferenceError('invalid generation length')
            if ignore_eos and len(row) != max_new_tokens:
                raise InferenceError('generation stopped before required token count')
            if any(type(x) is not int or not 0 <= x < vocab_size for x in row):
                raise InferenceError('invalid generated token ID')
        return result

    def token_logprobs(self, token_ids):
        result = self._request('token_logprobs', token_ids=token_ids)
        if not isinstance(result, list) or len(result) != len(token_ids):
            raise InferenceError('logprob batch size mismatch')
        for ids, row in zip(token_ids, result, strict=True):
            if not isinstance(row, list) or len(row) != len(ids) or row[0] is not None:
                raise InferenceError('logprobs must align with input tokens and start with null')
            if any(type(x) not in (int, float) or not math.isfinite(x) or x > 0 for x in row[1:]):
                raise InferenceError('logprobs must be finite, nonpositive natural logarithms')
        return result

    def measure_decode(self, token_ids, *, max_new_tokens, vocab_size):
        if max_new_tokens < 2:
            raise InferenceError('decode measurement needs at least two output tokens')
        return self._request('stream_generate', token_ids=token_ids,
                             max_new_tokens=max_new_tokens, ignore_eos=True,
                             _stream_count=max_new_tokens, _vocab_size=vocab_size)

    def close(self):
        if self.process is not None:
            # Kill only this worker's newly created process group, including GPU children.
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(self.process.pid, sig)
                except ProcessLookupError:
                    pass
                if sig == signal.SIGTERM:
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
            # A worker holding a large host allocation (offloaded or pinned memory)
            # can take minutes to be reaped after SIGKILL; the measurement it served
            # is complete, so cleanup waits for it and never fails the stage.
            try:
                self.process.wait(timeout=KILL_WAIT_SECONDS)
            except subprocess.TimeoutExpired:
                print(f'worker {self.process.pid} still exiting after {KILL_WAIT_SECONDS}s',
                      file=sys.stderr)
            for stream in (self.process.stdin, self.process.stdout):
                stream.close()
            self.process = None
        if self.log is not None:
            # Preserve a bounded backend log tail in the evaluator's archived stderr.
            self.log.seek(0, os.SEEK_END)
            self.log.seek(max(0, self.log.tell() - 65536))
            tail = self.log.read().decode(errors='replace')
            if tail:
                print(tail, file=sys.stderr, end='' if tail.endswith('\n') else '\n')
            self.log.close()
            self.log = None
        if self.work is not None:
            self.work.cleanup()
            self.work = None

    def __exit__(self, *_):
        self.close()


def generate_smoke(submission, tokenizer_dir, prompt, max_new_tokens, timeout):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True,
                                             trust_remote_code=False, use_fast=True)
    import task_policy
    if task_policy.current()['smoke_prompt'] == 'chat':
        ids = tokenizer.apply_chat_template([dict(role='user', content=prompt)], tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
    else:
        ids = tokenizer(prompt, add_special_tokens=False).input_ids
    with SubmittedRuntime(submission, max_model_len=max(2048, len(ids) + max_new_tokens),
                          max_num_seqs=1, timeout=timeout) as runtime:
        output = runtime.generate([ids], max_new_tokens=max_new_tokens, vocab_size=len(tokenizer))
    text = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    if not text:
        raise InferenceError('offline generation returned empty text')
    return text


def worker(submission):
    # Keep submitted Python/native logs separate from the internal response channel.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), 'w', buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    submission = Path(submission).resolve()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.append(str(submission))  # allow submitted helper modules
    engine = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            operation = request['operation']
            if operation == 'load':
                if request.get('inspect_model_memory'):
                    model_memory.install_vllm_adapter()
                spec = importlib.util.spec_from_file_location('candidate_inference', submission / 'inference.py')
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                if getattr(module, 'INFERENCE_API_VERSION', None) != API_VERSION:
                    raise ValueError('inference.py must set INFERENCE_API_VERSION = 1')
                engine = module.load(str(submission / 'model'), **request['options'])
                if not all(callable(getattr(engine, name, None)) for name in ('generate', 'token_logprobs')):
                    raise ValueError('load() must return generate() and token_logprobs() methods')
                result = RUNTIME_PROTOCOL
            elif engine is None:
                raise ValueError('load must run first')
            elif operation == 'model_memory':
                result = model_memory.collect(engine)
            elif operation == 'generate':
                result = engine.generate(request['token_ids'], max_new_tokens=request['max_new_tokens'],
                                         ignore_eos=request['ignore_eos'])
            elif operation == 'token_logprobs':
                result = engine.token_logprobs(request['token_ids'])
            elif operation == 'stream_generate':
                for chunk in engine.stream_generate(request['token_ids'],
                        max_new_tokens=request['max_new_tokens'], ignore_eos=request['ignore_eos']):
                    protocol.write(json.dumps({'ok': True, 'event': 'tokens',
                                               'token_ids': chunk}, allow_nan=False) + '\n')
                    protocol.flush()
                protocol.write(json.dumps({'ok': True, 'event': 'done'}) + '\n')
                protocol.flush()
                continue
            else:
                raise ValueError('unknown operation')
            response = json.dumps({'ok': True, 'result': result}, allow_nan=False)
        except Exception as exc:
            response = json.dumps({'ok': False, 'error': f'{type(exc).__name__}: {exc}'[:2000]})
        protocol.write(response + '\n')
        protocol.flush()


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] != '--worker':
        raise SystemExit('internal inference worker')
    worker(sys.argv[2])
