"""Task-owned decode measurement and the final score.

Decode latency is single-request streaming (one warmup, then 3 rounds of 3
requests, one token per event), measured and reported beside the score. The
score is `bf16-fidelity`: PPL against the split's BF16 baseline, and in the
hidden verdict the correctness gate (`fidelity.py`).
Review required: automated measurements are provisional until independent review.
"""
from importlib.metadata import version
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

from inference_runtime import SubmittedRuntime, review_metadata, model_memory
from ppl_benchmark import validate_result as validate_ppl_result, SCORING_PROTOCOL
import task_policy

ASSETS_DIR = Path(__file__).resolve().parent / 'evaluator-assets'


def config():
    return json.loads((ASSETS_DIR / 'decode_config.json').read_text())


def summarize(rows, cfg):
    expected = {(r, p) for r in range(cfg['rounds']) for p in range(cfg['requests_per_round'])}
    if len(rows) != len(expected) or {(r['round'], r['request']) for r in rows} != expected:
        raise ValueError('missing or duplicate decode requests')
    for row in rows:
        times = row['arrival_seconds']
        sizes, chunk_times = row['chunk_sizes'], row['chunk_arrival_seconds']
        if (not isinstance(sizes, list) or not sizes or sizes[0] != 1
                or any(type(n) is not int or n != 1 for n in sizes)
                or sum(sizes) != cfg['output_tokens']
                or not isinstance(chunk_times, list) or len(sizes) != len(chunk_times)
                or any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in chunk_times)
                or times != [t for n, t in zip(sizes, chunk_times) for _ in range(n)]):
            raise ValueError('invalid single-token events or receipt timestamps')
        if (row['input_tokens'] != cfg['input_tokens'] or row['output_tokens'] != cfg['output_tokens']
                or row['decode_intervals'] != cfg['output_tokens'] - 1
                or len(times) != cfg['output_tokens']
                or any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in times)
                or any(a > b for a, b in zip(times, times[1:]))):
            raise ValueError('invalid streamed token count or arrival timestamps')
        elapsed = times[-1] - times[0]
        if elapsed <= 0 or not math.isclose(row['decode_seconds'], elapsed, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError('decode time does not match first-to-last token interval')
    means = [statistics.mean(row['decode_seconds'] for row in rows if row['round'] == r)
             for r in range(cfg['rounds'])]
    median = statistics.median(means)
    return {'round_mean_seconds': means, 'median_round_mean_seconds': median,
            'decode_tokens_per_second': (cfg['output_tokens'] - 1) / median,
            'round_cv': statistics.pstdev(means) / statistics.mean(means),
            'round_spread_fraction': (max(means) - min(means)) / median}


def benchmark(submission, tokenizer_dir, *, timeout=600, samples_output=None):
    cfg = config()
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid',
                                      '--format=csv,noheader'], text=True).strip().splitlines()
    if len(devices) != 1:
        raise ValueError('decode measurement requires exactly one visible GPU')
    if version('vllm') != cfg['vllm_version']:
        raise ValueError('decode measurement requires the pinned vLLM environment')
    data = (ASSETS_DIR / 'decode_inputs.json').read_bytes()
    prompts = json.loads(data)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True,
                                             trust_remote_code=False, use_fast=True)
    vocab_size = len(tokenizer)
    measured = prompts['measured']
    if len(measured) != cfg['requests_per_round']:
        raise ValueError('wrong number of decode prompts')
    for ids in [prompts['warmup'], *measured]:
        if (len(ids) != cfg['input_tokens']
                or any(type(t) is not int or not 0 <= t < vocab_size for t in ids)):
            raise ValueError('invalid fixed decode prompt')
    rows = []
    log = Path(samples_output).open('x') if samples_output is not None else None
    try:
        started = time.perf_counter()
        with SubmittedRuntime(submission, max_model_len=cfg['max_model_len'],
                              max_num_seqs=cfg['max_num_seqs'],
                              gpu_memory_utilization=cfg['gpu_memory_utilization'], timeout=timeout) as engine:
            startup = time.perf_counter() - started
            warmup_start = time.perf_counter()
            warmup = engine.measure_decode(prompts['warmup'], max_new_tokens=cfg['output_tokens'],
                                            vocab_size=vocab_size)
            warmup_seconds = time.perf_counter() - warmup_start
            for r in range(cfg['rounds']):
                for p, ids in enumerate(measured):
                    sample = engine.measure_decode(ids, max_new_tokens=cfg['output_tokens'],
                                                   vocab_size=vocab_size)
                    row = {'round': r, 'request': p, 'input_tokens': len(ids),
                           'output_tokens': len(sample['token_ids']),
                           **{k: v for k, v in sample.items() if k != 'token_ids'}}
                    rows.append(row)
                    if log:
                        log.write(json.dumps(row) + '\n')
                        log.flush()
                    print(f'decode round {r+1}/3 request {p+1}/3: {row["decode_seconds"]:.6f}s', flush=True)
    finally:
        if log:
            log.close()
    result = {'valid': True, 'complete': True, **review_metadata(),
              'measurement_protocol': cfg['measurement_protocol'],
              'device': devices[0],
              'input_tokens': cfg['input_tokens'], 'output_tokens': cfg['output_tokens'],
              'gpu_memory_utilization': cfg['gpu_memory_utilization'],
              'concurrency': 1, 'warmup_requests': 1, 'measured_requests': len(rows),
              'timing_scope': 'parent first-single-token to final-token receipt; '
                              'excludes TTFT, includes stream delivery; speculative decoding prohibited',
              'engine_startup_seconds': startup, 'warmup_seconds': warmup_seconds,
              'warmup_decode_seconds': warmup['decode_seconds'], 'requests': rows,
              'model_memory': engine.memory_report(),
              **summarize(rows, cfg)}
    return result


def final_score(ppl_result, decode_result, *, public=False, ppl_assets=None,
                accuracy_result=None, accuracy_assets=None):
    """The `bf16-fidelity` score (`fidelity.py`): PPL against this split's BF16
    baseline; in the hidden verdict, behind the correctness gate. Decode latency is
    reported, not scored."""
    import fidelity
    cfg = config()
    ppl_assets = ppl_assets or ASSETS_DIR
    accuracy_assets = accuracy_assets or ppl_assets
    policy = model_memory.read_policy(ASSETS_DIR.parent)
    model_memory.validate_report(ppl_result.get("model_memory"), policy)
    model_memory.validate_report(decode_result.get("model_memory"), policy)
    if ppl_result.get('valid') is not True:
        raise ValueError('no valid quality measurement')
    validate_ppl_result(ppl_result, ppl_assets, expected_split='public' if public else 'hidden')
    if decode_result.get('valid') is not True or decode_result.get('complete') is not True:
        raise ValueError('no valid complete decode measurement')
    if decode_result.get('measurement_protocol') != cfg['measurement_protocol']:
        raise ValueError('decode result uses the wrong protocol')
    summary = summarize(decode_result['requests'], cfg)
    gate, extra = None, {}
    if not public:
        import accuracy_benchmark
        if not isinstance(accuracy_result, dict) or accuracy_result.get('valid') is not True:
            raise ValueError('no valid correctness-gate measurement')
        model_memory.validate_report(accuracy_result.get('model_memory'), policy)
        accuracy_benchmark.validate_result(accuracy_result, accuracy_assets, expected_split='hidden')
        base = fidelity.gate_baseline(accuracy_assets)
        gate = {'accuracy': accuracy_result['accuracy'], 'chance': base['chance'],
                'bf16_accuracy': base['bf16_accuracy'],
                'max_accuracy_drop': task_policy.current()['quality']['max_accuracy_drop']}
        extra = {'accuracy_protocol': accuracy_result['accuracy_protocol'],
                 'accuracy_split': accuracy_result['accuracy_split'],
                 'accuracy_benchmarks': {
                     name: {'accuracy': row['accuracy'], 'bf16_accuracy': base['bf16_benchmarks'].get(name),
                            'chance': row['chance'], 'kind': row['kind'], 'items': row['items']}
                     for name, row in (accuracy_result.get('benchmarks') or {}).items()}}
    metrics = fidelity.score(ppl_result['ppl'], bf16_ppl=fidelity.ppl_baseline(ppl_assets), gate=gate)
    passed = metrics.get('accuracy_gate_pass', True)
    return {'valid': True, **review_metadata(), 'scoring_version': SCORING_PROTOCOL,
            'capacity_protocol': model_memory.PROTOCOL,
            'model_memory': decode_result.get('model_memory'),
            **{key: ppl_result[key] for key in ('ppl_protocol', 'ppl_split', 'predicted_tokens')},
            'public_only': public, 'is_official_score': not public, **summary,
            'decode_latency_seconds': summary['median_round_mean_seconds'], 'decode_in_reward': False,
            'quality_gate_enabled': not public, 'quality_pass': passed if not public else None,
            'ranking_eligible': passed, **metrics, **extra}
