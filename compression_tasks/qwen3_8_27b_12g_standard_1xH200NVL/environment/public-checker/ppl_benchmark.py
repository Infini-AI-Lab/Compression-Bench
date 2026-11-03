"""Task-owned fixed perplexity blocks; shared by public Checker and hidden Verifier.

The protocol (and with it the count of unscored prefix tokens per block) comes
from the task policy; the debug-9b split always uses the raw C4 protocol.
"""
import json
import math
from pathlib import Path

import task_policy

BLOCKS = 256
CONTEXT = 2048
SCORING_PROTOCOL = task_policy.current()['scoring_protocol']


def protocol_for(split):
    """(protocol, unscored prefix tokens per block, scored tokens) for a split."""
    protocol = task_policy.ppl_protocol_for_split(split)
    prefix = task_policy.PPL_PROTOCOLS[protocol]['prefix_tokens']
    return protocol, prefix, BLOCKS * (CONTEXT - prefix)


def load_inputs(assets, tokenizer_dir, *, expected_split=None):
    assets, tokenizer_dir = Path(assets), Path(tokenizer_dir)
    protocol, prefix, scored = protocol_for(expected_split)
    cfg = json.loads((assets / 'ppl_config.json').read_text())
    if (cfg.get('protocol') != protocol or cfg.get('blocks') != BLOCKS
            or cfg.get('context_length') != CONTEXT or cfg.get('scored_tokens') != scored
            or cfg.get('input_tokens') != BLOCKS * CONTEXT):
        raise ValueError('wrong fixed PPL protocol')
    if expected_split is not None and cfg.get('split') != expected_split:
        raise ValueError('wrong public/hidden PPL split')
    data = (assets / 'ppl_inputs.json').read_bytes()
    rows = json.loads(data)
    if (len(rows) != BLOCKS or any(len(row) != CONTEXT for row in rows)
            or any(type(t) is not int or not 0 <= t < cfg['vocab_size'] for row in rows for t in row)):
        raise ValueError('invalid PPL input blocks or token IDs')
    chat = task_policy.chat_prefix_tokens(protocol)
    if chat and any(row[:chat] != cfg['prefix_tokens'] for row in rows):
        label = task_policy.PPL_PROTOCOLS[protocol]['prefix_label']
        raise ValueError(f'wrong {label} chat prefix')
    return cfg, rows


def measure(submission, assets, tokenizer_dir, runtime_class, *, timeout=600, expected_split=None):
    protocol, prefix, scored = protocol_for(expected_split)
    cfg, rows = load_inputs(assets, tokenizer_dir, expected_split=expected_split)
    windows = []
    # Fixed batching: batches of 16, engine capacity 32, no overlap.
    with runtime_class(submission, max_model_len=2049, max_num_seqs=32,
                       gpu_memory_utilization=0.9, timeout=timeout) as runtime:
        for offset in range(0, BLOCKS, 16):
            batch = rows[offset:offset + 16]
            outputs = runtime.token_logprobs(batch)
            if len(outputs) != len(batch):
                raise ValueError('missing PPL probability rows')
            for i, (ids, logprobs) in enumerate(zip(batch, outputs, strict=True)):
                if len(logprobs) != len(ids):
                    raise ValueError('misaligned PPL probabilities')
                values = logprobs[prefix:]
                if any(type(v) not in (int, float) or not math.isfinite(v) or v > 1e-5 for v in values):
                    raise ValueError('invalid PPL log probabilities')
                windows.append({'block': offset+i, 'nll': -sum(values), 'scored_tokens': CONTEXT-prefix})
    nll = sum(w['nll'] for w in windows)
    ppl = math.exp(nll / scored)
    if not math.isfinite(ppl) or ppl < 1:
        raise ValueError('PPL must be finite and >= 1')
    return dict(valid=True, model_memory=runtime.memory_report(), ppl=ppl, nll=nll, predicted_tokens=scored,
                input_tokens=BLOCKS*CONTEXT, blocks=BLOCKS, context_length=CONTEXT,
                stride=CONTEXT, batch_size=16, max_num_seqs=32, gpu_memory_utilization=0.9,
                ppl_protocol=protocol,
                ppl_split=cfg['split'],
                windows=windows, quality_gate_enabled=False)


def validate_result(result, assets, *, expected_split):
    """Do not score a result of another protocol or mix public and hidden measurements."""
    protocol, _, scored = protocol_for(expected_split)
    cfg = json.loads((Path(assets) / 'ppl_config.json').read_text())
    if (cfg.get('protocol') != protocol or cfg.get('split') != expected_split
            or result.get('ppl_protocol') != protocol
            or result.get('ppl_split') != expected_split
            or result.get('predicted_tokens') != scored):
        raise ValueError('PPL result uses the wrong protocol, split or token count')
