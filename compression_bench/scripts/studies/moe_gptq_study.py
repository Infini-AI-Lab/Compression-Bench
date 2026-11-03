#!/usr/bin/env python3
"""MoE ablation of GPTQ error compensation on fixed selected grids: one new artifact, two fresh verifications.

A is the decision ablation's selected MoE artifact (moe_A, controlled_ablation.py). B rounds the source routed-expert
weights to A's exact selected FP8 scale grids (same channel order), omitting GPTQ error-compensation updates
(moe_gptq_build.py); every other tensor is A's. One intervention, fixed in the preregistration before verification.

    moe_gptq_study.py run [--model-dir DIR]     # build B, freeze both, verify both on idle cards, collect
    moe_gptq_study.py collect                   # paired results, REPORT.md and the combined ablation overview
    moe_gptq_study.py archive                   # integrity checks and portable archives

Records in OUT (evaluation_progress/controlled_ablation/moe_gptq), which must hold `preregistration.json` before
`run`; B is built under WORK (compression_bench/evolve_space/studies/controlled_ablation_moe_gptq). The decision
ablation (controlled_ablation.py) must be complete; its records are read, never modified.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import math
import os.path
from pathlib import Path
import subprocess
import threading
import time

import common
from common import REPO, read_json, utc_now, write_json
import controlled_ablation as decision
from controlled_parallel import freeze_artifacts, verification_command

OUT = REPO / 'evaluation_progress/controlled_ablation/moe_gptq'
WORK = REPO / 'compression_bench/evolve_space/studies/controlled_ablation_moe_gptq'
PRIOR = decision.OUT
STUDY = decision.STUDY
TASK = decision.TASK['moe']
TESTS = REPO / 'compression_tasks' / TASK / 'tests'
DESIGN = ['moe_gptq_build.py', 'moe_gptq_study.py', 'controlled_verify.py', 'controlled_ablation.py', 'study_tensors.py']
lock = threading.Lock()


def locations(model_dir=None):
    """Where the source checkpoint, A, B and the prior MoE build record are; written once, before the build."""
    path = OUT / 'locations.json'
    if path.exists():
        return read_json(path)
    loc = {'source': str(model_dir or common.model_dir(TASK)), 'A': str(decision.WORK / 'artifacts/moe_A'),
           'B': str(WORK / 'artifacts/moe_B'), 'prior': str(PRIOR)}
    write_json(path, loc)
    return loc


def design_controls():
    """Hashes of the preregistration and the scripts that define the intervention, before any fresh verification."""
    files = [OUT / 'preregistration.json'] + [STUDY / n for n in DESIGN]
    write_json(OUT / 'design_controls.json', {'recorded_utc': utc_now(), 'before_fresh_verification': True,
                                             'files': {str(p.relative_to(REPO)): common.file_sha256(p) for p in files}})


def build(image):
    loc = locations()
    target = Path(loc['B'])
    assert not target.exists(), 'never overwrite an artifact'
    target.parent.mkdir(parents=True, exist_ok=True)
    # B starts as A without its weights: every non-weight file stays byte-identical.
    subprocess.run(['cp', '-a', '--reflink=auto', loc['A'], str(target)], check=True)
    for f in (target / 'model').glob('*.safetensors'):
        f.unlink()
    with lock:
        gpu, held = common.claim_gpu('moe-gptq-ablation-build')
    name = 'moe-gptq-fixed-grid-build'
    model = Path(loc['source'])
    cmd = common.docker_run(name=name, gpu=gpu)
    for src, dst, mode in [(loc['A'], '/selected', 'ro'), (loc['B'], '/alternate', 'rw'),
                           (model, '/input/model', 'ro'), (loc['prior'], '/prior', 'ro'),
                           (OUT, '/record', 'rw'), (STUDY, '/study', 'ro')]:
        cmd += ['-v', f'{src}:{dst}:{mode}']
    cmd += [*common.hf_cache_mounts(model), '--entrypoint', 'python3', image, '/study/moe_gptq_build.py']
    write_json(OUT / 'build_launch.json', {'started_utc': utc_now(), 'gpu': gpu, 'argv': cmd})
    try:
        rc = common.run_container(cmd, OUT / 'build.log', name, 3600)
        write_json(OUT / 'build_exit.json', {'finished_utc': utc_now(), 'returncode': rc})
        assert rc == 0, 'preserve build failure; no outcome-dependent alternative'
    finally:
        held.release()


def representation_audit(image):
    """CPU check that B represents different weights from A, not merely different codes on zero-valued grids."""
    loc = locations()
    cmd = [*common.docker_run(cpus='4', memory='64g'),
           '-v', f"{loc['A']}:/a:ro", '-v', f"{loc['B']}:/b:ro", '-v', f'{OUT}:/record', '-v', f'{STUDY}:/study:ro',
           '--entrypoint', 'python3', image, '/study/moe_gptq_representation_audit.py']
    with (OUT / 'representation_audit.log').open('w') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)


def freeze(image):
    loc = locations()
    artifacts = freeze_artifacts({'moe_' + s: Path(loc[s]) for s in 'AB'}, TASK)
    prior = read_json(PRIOR / 'pair_freeze_moe.json')['artifacts']['moe_A']['files']
    assert artifacts['moe_A']['files'] == prior, 'moe_A changed since the decision ablation froze it'
    record = {'frozen_utc': utc_now(), 'both_frozen_before_fresh_verification': True,
              'image_id': image, 'preregistration_sha256': common.file_sha256(OUT / 'preregistration.json'),
              'observer_sha256': common.file_sha256(decision.OBSERVER), 'artifacts': artifacts,
              'evaluator': common.manifest(TESTS)}
    assert not (OUT / 'verification').exists()
    write_json(OUT / 'freeze.json', record)
    return record


def verify(suffix, frozen, state):
    name = 'moe_' + suffix
    docker_name = 'moe-gptq-ablation-' + suffix.lower()
    with lock:
        gpu, held = common.claim_gpu('moe-gptq-ablation-' + suffix)
    try:
        dest = OUT / 'verification' / name
        dest.mkdir(parents=True, exist_ok=False)
        cmd = verification_command(frozen['image_id'], docker_name, gpu, frozen['artifacts'][name]['root'], TESTS, dest)
        job = {'artifact': name, 'gpu': gpu, 'gpu_uuid': common.gpu_uuid(gpu), 'hardware': common.gpu_hardware(gpu),
               'started_utc': utc_now(), 'status': 'running', 'argv': cmd}
        with lock:
            assert common.manifest(TESTS) == frozen['evaluator']
            assert common.file_sha256(decision.OBSERVER) == frozen['observer_sha256']
            state['jobs'].append(job)
            write_json(OUT / 'state.json', state)
            write_json(dest / 'configuration.json', job)
        start = time.monotonic()
        rc = common.run_container(cmd, dest / 'execution.log', docker_name, 5400)
        with lock:
            job.update(finished_utc=utc_now(), gpu_seconds=time.monotonic() - start,
                       returncode=rc, status='finished' if rc == 0 else 'execution_failed')
            write_json(dest / 'configuration.json', job)
            write_json(OUT / 'state.json', state)
    finally:
        held.release()


def report(rows, pairs):
    lines = ['# MoE ablation: GPTQ compensation on fixed selected grids', '',
        'One new ablated artifact and two fresh final verifications; no agents, searches, training, or result-dependent variants.', '',
        'A is the selected MoE model of the decision ablation (moe_A). B rounds the source routed-expert gate/up and down weights to A’s exact selected FP8 scale grids, omitting GPTQ error-compensation updates. Precision, group sizes, channel order, scale values, expert IDs, router weights, all nonexpert tensors, packing and inference code stay fixed. No recalibration is performed. This measures compensation conditional on the selected grids, not a separately scale-optimized RTN recipe. Both packages are frozen before fresh verification.', '',
        '| Condition | PPL | GSM8K (%) | HumanEval (%) | TriviaQA (%) | Mean (%) | Package GiB | Resident GiB | Sampled peak GiB | Decode s | TPS | Accuracy gate |',
        '|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|']
    for r in rows:
        lines.append(f"| {r['artifact']} | {r['verification_ppl']:.6f} | {100*r['GSM8K_accuracy']:.2f} | {100*r['HumanEval_accuracy']:.2f} | {100*r['TriviaQA_accuracy']:.2f} | {100*r['mean_accuracy']:.2f} | {r['package_bytes']/2**30:.3f} | {r['resident_model_bytes']/2**30:.3f} | {r['sampled_peak_device_bytes']/2**30:.3f} | {r['decode_seconds']:.4f} | {r['tps']:.2f} | {r['accuracy_gate_pass']} |")
    for p in pairs:
        ci = p['bootstrap']
        pci = '[' + ', '.join(f'{x:+.6f}' for x in ci['ppl_delta_95pct']) + ']'
        aci = '[' + ', '.join(f'{x:+.2f}' for x in ci['mean_accuracy_change_pp_95pct']) + ']'
        lines += ['', f"B − A: PPL {p['ppl_delta']:+.6f}; mean accuracy {p['mean_accuracy_change_pp']:+.2f} percentage points. Paired 95% bootstrap intervals: PPL {pci}; mean accuracy {aci} percentage points. Package difference {100*p['package_relative_difference']:.4f}%.", '',
            '| Benchmark | A correct | B correct | Corrected by B | Newly broken by B |',
            '|---|--:|--:|--:|--:|']
        for bench, v in p['per_benchmark'].items():
            lines.append(f"| {decision.BENCHMARK[bench]} | {v['A_correct']}/50 | {v['B_correct']}/50 | {v['corrected_by_B']} | {v['newly_broken_by_B']} |")
        likelihood = 'supports a likelihood benefit from compensation' if ci['ppl_delta_95pct'][0] > 0 else 'does not establish a likelihood benefit from compensation'
        accuracy = 'supports an accuracy benefit from compensation' if ci['mean_accuracy_change_pp_95pct'][1] < 0 else 'does not establish an accuracy benefit from compensation'
        lines += ['', f'This fixed comparison {likelihood} and {accuracy}. Keeping grids fixed isolates the sequential correction of quantization errors; it does not measure scale search, precision allocation, calibration-data selection or channel ordering. One selected calibration/design state and small benchmark samples do not establish an agent-wide effect.']
    lines += ['', 'Measurements use the unchanged final verifier: 256 BF16-continued PPL sequences; 50 questions each for GSM8K, HumanEval and TriviaQA; batch-one 2048-token prompt / 128 output tokens, one warmup then 3×3 decode requests. TPS uses 127 first-to-last intervals. Per-question correctness, corrected/newly broken indices, sequence NLLs and paired 10,000-resample bootstrap outputs are kept separately from design-time reward feedback.', '',
        'Resident storage is the official storage gate. Peak is observed total-device memory sampled at 10 ms across verification, including the standard vLLM cache reservation; shorter transients may be missed. CUDA graphs remain enabled. Parallel timing uses separate idle cards and includes card/session variability; it is not an isolated speed effect. Raw inference audits remain pending and scores provisional unless the evaluator explicitly says otherwise. The paired intervals are nominal on the same frozen evaluation split as the decision ablation, not an independent newly sampled holdout; they do not correct for the choice of this ablation after that study.', '',
        f'Completed full-result conditions: {len(rows)}/2. The decision ablation\'s records are read, never modified.']
    lines += ['', '| Condition | TPS range across three rounds | Inference audit / score status |',
              '|---|--:|---|']
    for r in rows:
        lines.append(f"| {r['artifact']} | {r['round_tps_min']:.4f}–{r['round_tps_max']:.4f} | {r['audit_status']} / {r['score_status']} |")
    state = read_json(OUT / 'state.json')
    if state.get('finished_utc'):
        seconds = read_json(OUT / 'build.json')['seconds'] + sum(j['gpu_seconds'] for j in state['jobs'])
        lines += ['', f'Successful construction plus fresh verification claims: {seconds/3600:.2f} GPU-hours, excluding CPU hashing and staging.']
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def collect():
    """Per-sequence and paired results of the two verifications, REPORT.md, and the combined overview."""
    assert read_json(OUT / 'freeze.json')['both_frozen_before_fresh_verification']
    access = read_json(OUT / 'analysis_access.json') if (OUT / 'analysis_access.json').exists() else {}
    write_json(OUT / 'analysis_access.json', {'first_collect_utc': access.get('first_collect_utc', utc_now()),
                                             'both_artifacts_already_frozen': True,
                                             'artifact_count': 2})
    names = ['moe_A', 'moe_B']
    paths = [OUT / 'verification' / name / 'ppl.json' for name in names]
    if all(common.available(p) for p in paths):
        a, b = [read_json(p)['windows'] for p in paths]
        assert [x['block'] for x in a] == [x['block'] for x in b]
        paired = []
        for x, y in zip(a, b):
            assert x['scored_tokens'] == y['scored_tokens']
            n = x['scored_tokens']
            paired.append({'block': x['block'], 'scored_tokens': n,
                           'A_nll': x['nll'], 'B_nll': y['nll'],
                           'A_sequence_ppl': math.exp(x['nll']/n),
                           'B_sequence_ppl': math.exp(y['nll']/n),
                           'nll_per_token_delta_B_minus_A': (y['nll']-x['nll'])/n})
        with (OUT / 'paired_sequences.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(paired[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(paired)
        write_json(OUT / 'sequence_summary.json', {
            'total': len(paired),
            'lower_nll_with_A': sum(r['B_nll'] > r['A_nll'] for r in paired),
            'lower_nll_with_B': sum(r['A_nll'] > r['B_nll'] for r in paired),
            'equal_nll': sum(r['A_nll'] == r['B_nll'] for r in paired),
            'all_sequences_retained': True})
    decision.collect_pairs(OUT, names, {'moe': TASK}, report)
    pairs = read_json(OUT / 'paired_results.json')
    if len(pairs) == 1 and (PRIOR / 'paired_results.json').exists():
        overview(pairs[0])


def overview(compensation):
    """All three ablations side by side, including those whose intervals include zero."""
    previous = {p['task']: p for p in read_json(PRIOR / 'paired_results.json')}
    cases = [('Dense: MSE range refinement', previous['dense']),
             ('MoE: channel ordering', previous['moe']),
             ('MoE: GPTQ compensation', compensation)]
    lines = ['# Controlled ablation results', '',
        'The GPTQ-compensation case isolates within-matrix error compensation on identical selected grids; it reuses the decision ablation\'s selected MoE artifact and frozen evaluation split.', '',
        '| Ablated technique | A PPL | B PPL | PPL change B − A (95% interval) | A mean accuracy | B mean accuracy | Accuracy change, pp (95% interval) |',
        '|---|--:|--:|---|--:|--:|---|']
    for name, p in cases:
        a, b = p['A_metrics'], p['B_metrics']
        pci = ', '.join(f'{x:+.5f}' for x in p['bootstrap']['ppl_delta_95pct'])
        aci = ', '.join(f'{x:+.2f}' for x in p['bootstrap']['mean_accuracy_change_pp_95pct'])
        lines.append(f"| {name} | {a['verification_ppl']:.5f} | {b['verification_ppl']:.5f} | {p['ppl_delta']:+.5f} [{pci}] | {100*a['mean_accuracy']:.2f}% | {100*b['mean_accuracy']:.2f}% | {p['mean_accuracy_change_pp']:+.2f} [{aci}] |")
    rel = lambda path: os.path.relpath(path, OUT.parent)
    lines += ['', 'A is the selected setting; B removes one decision. The dense comparison disables per-group MSE range refinement with precision allocation fixed. The channel-ordering control removes intermediate-channel ordering. The compensation control keeps the selected FP8 scale values and that ordering fixed, and rounds the source expert weights without compensating column-wise rounding errors; it measures compensation conditional on these grids, not an independently scale-optimized RTN method. Precision, expert IDs, router weights, parameter count and inference code are unchanged within each pair.', '',
        'Nominal paired intervals resample the fixed 256 PPL sequences and questions within each benchmark. They do not describe agent-seed variability. PPL support and generation-accuracy support are assessed separately. Numeric gate outcomes are reported alongside pending/provisional inference-audit status.', '',
        f'- [Dense and channel-ordering measurements]({rel(PRIOR / "REPORT.md")})',
        f'- [GPTQ-compensation measurements]({rel(OUT / "REPORT.md")})',
        f'- [GPTQ-compensation per-question paired changes]({rel(OUT / "paired_questions.csv")})']
    (OUT.parent / 'CURRENT_REPORT.md').write_text('\n'.join(lines) + '\n')


def archive():
    """Finalize portable evidence without a model call or changing old results."""
    state = read_json(OUT / 'state.json')
    assert state.get('finished_utc') and len(state['jobs']) == 2
    assert all(j['returncode'] == 0 for j in state['jobs'])
    collect()
    rows = read_json(OUT / 'results.json')
    assert len(rows) == 2
    frozen = read_json(OUT / 'freeze.json')
    with ThreadPoolExecutor(max_workers=2) as pool:
        checks = {name: pool.submit(common.manifest, a['root'])
                  for name, a in frozen['artifacts'].items()}
        for name, future in checks.items():
            assert future.result() == frozen['artifacts'][name]['files'], name
    assert common.manifest(TESTS) == frozen['evaluator']
    assert common.file_sha256(decision.OBSERVER) == frozen['observer_sha256']
    assert common.file_sha256(OUT / 'preregistration.json') == frozen['preregistration_sha256']
    assert read_json(OUT / 'representation_audit.json')['not_merely_different_zero_scale_codes']
    controls = read_json(OUT / 'design_controls.json')
    assert controls['before_fresh_verification']
    assert all(common.file_sha256(REPO / rel) == sha for rel, sha in controls['files'].items())
    # The decision ablation's records are read, never changed.
    old_manifest = read_json(PRIOR / 'evidence_manifest.json')
    assert all(common.file_sha256(PRIOR / rel) == info['sha256'] for rel, info in old_manifest['portable_files'].items())
    for row in rows:
        limit = common.limit_bytes(TASK)
        assert row['package_bytes'] <= limit and row['resident_model_bytes'] <= limit
        dest = OUT / 'verification' / row['artifact']
        acc = read_json(dest / 'accuracy.json')
        assert not any(k in acc for k in ('generations', 'predictions'))
        for bench, items in acc['correct_items'].items():
            assert len(items) == 50 and sum(items) == acc['benchmarks'][bench]['correct']
        ppl = read_json(dest / 'ppl.json')
        assert len(ppl['windows']) == 256 and sum(w['scored_tokens'] for w in ppl['windows']) == 393216
        decode = read_json(dest / 'decode.json')
        assert decode['measured_requests'] == len(decode['requests']) == 9
    assert len(list(csv.DictReader((OUT / 'paired_questions.csv').open()))) == 150
    assert len(list(csv.DictReader((OUT / 'paired_sequences.csv').open()))) == 256
    pair = read_json(OUT / 'paired_results.json')[0]
    assert pair['package_relative_difference'] <= .005
    for bench in pair['per_benchmark'].values():
        assert bench['B_correct'] - bench['A_correct'] == bench['corrected_by_B'] - bench['newly_broken_by_B']
    write_json(OUT / 'completion_checks.json', {
        'checked_utc': utc_now(), 'fresh_full_verifications': 2, 'new_artifacts': 1,
        'new_agent_runs': 0, 'all_measurement_pairing_and_integrity_checks_pass': True,
        'decision_ablation_evidence_unchanged': True,
        'combined_report_sha256': common.file_sha256(OUT.parent / 'CURRENT_REPORT.md'),
        'gpu_hours_build_plus_verification': (read_json(OUT / 'build.json')['seconds'] + sum(j['gpu_seconds'] for j in state['jobs'])) / 3600})
    common.archive_portable(OUT)
    print('Measurements, decision-ablation preservation, and portable archives checked.')


def run():
    assert not (OUT / 'state.json').exists(), 'never relaunch a study blindly'
    state = {'started_utc': utc_now(), 'jobs': [], 'fresh_full_verifications': 2}
    write_json(OUT / 'state.json', state)
    image = read_json(PRIOR / 'pair_freeze_moe.json')['image_id']
    build(image)
    representation_audit(image)
    frozen = freeze(image)
    design_controls()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(verify, x, frozen, state) for x in 'AB']
        for future in futures:
            future.result()
    state['finished_utc'] = utc_now()
    write_json(OUT / 'state.json', state)
    collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['run', 'collect', 'archive'])
    parser.add_argument('--model-dir', help='the MoE task\'s BF16 checkpoint (default: model_for_task.py)')
    args = parser.parse_args()
    if args.command == 'run':
        OUT.mkdir(parents=True, exist_ok=True)
        locations(args.model_dir)
    {'run': run, 'collect': collect, 'archive': archive}[args.command]()
