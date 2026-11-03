#!/usr/bin/env python3
"""Validate four completed verifications and losslessly archive portable evidence.

CPU only. Never launches evaluation or modifies executable artifacts.
"""
import csv
import json

import common
from common import read_json
from controlled_ablation import NAMES, OBSERVER, OUT, TASK, WORK, collect


def integrity():
    """Every frozen artifact, the evaluator and the verification wrapper still match their freeze records."""
    for kind in TASK:
        frozen = read_json(OUT / f'pair_freeze_{kind}.json')
        for name, artifact in frozen['artifacts'].items():
            assert common.manifest(artifact['root']) == artifact['files'], name
        assert common.manifest(common.REPO / 'compression_tasks' / TASK[kind] / 'tests') == frozen['evaluator'], kind
        assert common.file_sha256(OBSERVER) == frozen['observer_sha256']
    common.write_json(OUT / 'integrity_audit.json', {'checked_utc': common.utc_now(), 'all_four_artifact_manifests_unchanged': True,
                                                    'evaluator_and_observer_unchanged': True})


def main():
    state = read_json(OUT / 'parallel_state.json')
    assert state.get('finished_utc'), 'wait for the controller to finish'
    assert sorted(j['artifact'] for j in state['jobs']) == sorted(NAMES)
    assert all(j['returncode'] == 0 for j in state['jobs'])
    integrity()
    source_audit = read_json(OUT / 'moe_source_reproduction_audit.json')
    assert source_audit['all_exact_equal']
    assert all(read_json(OUT / 'dense_reproduction_check.json')['exact_equal'].values())
    collect()
    rows = read_json(OUT / 'results.json')
    assert len(rows) == 4
    for row in rows:
        name = row['artifact']
        limit = common.limit_bytes(TASK[row['task']])
        assert row['package_bytes'] <= limit and row['resident_model_bytes'] <= limit
        dest = OUT / 'verification' / name
        accuracy = read_json(dest / 'accuracy.json')
        decode = read_json(dest / 'decode.json')
        ppl = read_json(dest / 'ppl.json')
        assert row['verification_valid']
        assert len(ppl['windows']) == 256
        assert sum(w['scored_tokens'] for w in ppl['windows']) == 393216
        assert decode['measured_requests'] == len(decode['requests']) == 9
        assert decode['input_tokens'] == 2048 and decode['output_tokens'] == 128
        assert abs(row['tps'] - 127 / decode['median_round_mean_seconds']) < 1e-10
        assert not any(k in accuracy for k in ('generations', 'predictions'))
        for bench, verdicts in accuracy['correct_items'].items():
            assert len(verdicts) == 50
            assert sum(verdicts) == accuracy['benchmarks'][bench]['correct']
        assert len(list(csv.DictReader((dest / 'question_correctness.csv').open()))) == 150
    pairs = read_json(OUT / 'paired_results.json')
    assert len(pairs) == 2
    for pair in pairs:
        assert pair['package_relative_difference'] <= .005
        for bench in pair['per_benchmark'].values():
            assert bench['B_correct'] - bench['A_correct'] == bench['corrected_by_B'] - bench['newly_broken_by_B']
    assert len(list(csv.DictReader((OUT / 'paired_questions.csv').open()))) == 300
    build_seconds = sum(read_json(OUT / (k + '_build.json'))['seconds'] for k in TASK)
    verification_seconds = sum(j['gpu_seconds'] for j in state['jobs'])
    common.write_json(OUT / 'completion_checks.json', {
        'checked_utc': common.utc_now(), 'evaluated_artifacts': 4,
        'fresh_full_verifications': 4, 'new_agent_runs': 0,
        'paired_question_rows': 300, 'package_and_resident_gates_pass': True,
        'moe_selected_reproduction_exact_tensor_count': source_audit['exact_equal_count'],
        'all_raw_measurement_and_pairing_checks_pass': True,
        'successful_build_gpu_seconds': build_seconds,
        'verification_claim_gpu_seconds': verification_seconds,
        'accounted_gpu_hours': (build_seconds + verification_seconds) / 3600,
        'accounting_scope': 'Successful builder elapsed time plus complete verification GPU claims; excludes CPU audits and model staging.'})
    common.archive_portable(OUT, artifacts_retained_at=str(WORK / 'artifacts'))
    print(json.dumps(read_json(OUT / 'completion_checks.json'), indent=2))


if __name__ == '__main__':
    main()
