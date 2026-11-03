"""The reward and the correctness gate: fidelity to the task's own BF16 checkpoint.

The perplexity blocks are text of a hidden corpus continued by the BF16 checkpoint
itself (`task_policy.LIVE_PROTOCOLS`): the first 512 tokens of each block are
the corpus's text (a mix of web pages, code, mathematics, question answering and
reasoning traces, with the protocol's chat prefix), the last 1536 the BF16
checkpoint's own sample at temperature 1, and only those are scored. On text
drawn from BF16, a submission's excess cross-entropy over BF16 is, in
expectation, its KL divergence from BF16:

    R_ppl  = PPL_bf16 / PPL
           = exp(-(CE - CE_bf16)) ~ exp(-KL(BF16 || submission))
    reward = REWARD_SCALE * R_ppl

10 is a submission that predicts exactly like BF16; any change to the
distribution -- compression error, a temperature, training toward other text --
lowers it in expectation, and only sampling noise can lift it above 10, so the
reward has no maximum. The public Checker reports this on public blocks; the
verdict on hidden ones.

The correctness gate runs only in the hidden verdict, on questions only the
Verifier holds, answered by greedy generation (`accuracy_benchmark.py`). It rejects a
submission that keeps less than 1 - max_accuracy_drop of BF16's above-chance
accuracy on them, retention = (acc - chance) / (acc_bf16 - chance); a rejected
submission scores 0, whatever its reward. The agent is told the gate exists,
not its questions, their source or its threshold.
"""
import json
import math
from pathlib import Path

PROTOCOL = 'bf16-sample-ppl-gated-v5'
REWARD_SCALE = 10.0
FORMULA = ('reward = 10 * R_ppl, R_ppl = bf16_ppl / ppl on BF16-sampled continuations of the hidden contexts '
           '(~ exp(-KL)); in the verdict 0 if '
           '(accuracy - chance) / (bf16_accuracy - chance) < 1 - max_accuracy_drop')


def _finite(name, value, low=None, high=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f'{name} is out of range')
    return float(value)


def ppl_term(ppl, bf16_ppl):
    ppl, bf16_ppl = _finite('PPL', ppl, 1), _finite('BF16 PPL', bf16_ppl, 1)
    return bf16_ppl / ppl


def accuracy_retention(accuracy, bf16_accuracy, chance):
    """The unclipped share of BF16's above-chance accuracy kept."""
    accuracy = _finite('accuracy', accuracy, 0, 1)
    bf16_accuracy = _finite('BF16 accuracy', bf16_accuracy, 0, 1)
    chance = _finite('chance accuracy', chance, 0, 1)
    if bf16_accuracy <= chance:
        raise ValueError('BF16 accuracy must exceed chance')
    return (accuracy - chance) / (bf16_accuracy - chance)


def reward(r_ppl):
    return REWARD_SCALE * _finite('R_ppl', r_ppl, 0)


def ppl_baseline(ppl_assets):
    cfg = json.loads((Path(ppl_assets) / 'ppl_config.json').read_text())
    if cfg.get('baseline_status') != 'measured_new_protocol':
        raise ValueError('reward requires a measured task-owned BF16 PPL baseline')
    return _finite('BF16 PPL', cfg.get('bf16_ppl'), 1)


def gate_baseline(accuracy_assets):
    """The hidden gate's measured BF16 accuracy; refuses an unmeasured task."""
    cfg = json.loads((Path(accuracy_assets) / 'accuracy_config.json').read_text())
    bf16 = cfg.get('bf16') or {}
    if cfg.get('baseline_status') != 'measured' or cfg.get('split') != 'hidden':
        raise ValueError('the correctness gate requires a measured hidden BF16 accuracy baseline')
    return {'bf16_accuracy': _finite('BF16 accuracy', bf16.get('accuracy'), 0, 1),
            'chance': _finite('chance accuracy', cfg.get('chance'), 0, 1),
            'bf16_benchmarks': {name: row['accuracy'] for name, row in
                                (bf16.get('benchmarks') or {}).items()}}


def score(ppl, *, bf16_ppl, gate=None):
    """Every reward field. `gate` is the hidden verdict's correctness gate,
    {accuracy, chance, bf16_accuracy, max_accuracy_drop}; the public Checker has none."""
    r_ppl = ppl_term(ppl, bf16_ppl)
    out = {'reward': reward(r_ppl), 'reward_scale': REWARD_SCALE, 'reward_mapping': FORMULA,
           'ppl_fidelity': r_ppl, 'ppl': ppl, 'bf16_ppl': bf16_ppl,
           'ppl_increase_percent': 100 * (ppl / bf16_ppl - 1), 'correctness_gate': 'not_run'}
    if gate is None:
        return out
    drop = _finite('max_accuracy_drop', gate['max_accuracy_drop'], 0, 1)
    retention = accuracy_retention(gate['accuracy'], gate['bf16_accuracy'], gate['chance'])
    floor = 1.0 - drop
    passed = retention >= floor - 1e-12      # inclusive: equality passes, to rounding
    out.update({'reward': out['reward'] if passed else 0.0, 'correctness_gate': 'run',
                'accuracy': gate['accuracy'], 'bf16_accuracy': gate['bf16_accuracy'],
                'chance_accuracy': gate['chance'], 'accuracy_retention': retention,
                'accuracy_gate_floor': floor, 'accuracy_gate_pass': passed, 'max_accuracy_drop': drop,
                'accuracy_change_points': 100 * (gate['accuracy'] - gate['bf16_accuracy']),
                'quality_rejection_reason': None if passed else
                f'accuracy_below_{round(100 * floor)}_percent_of_bf16'})
    return out
