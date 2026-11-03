#!/usr/bin/env python3
"""Pair-wise freeze and independent parallel verifications; outcomes stay sealed until all four are frozen.

    controlled_parallel.py

Waits for each pair's build record, freezes the pair (file manifests, evaluator, image; the MoE pair after
controlled_audit.py's tensor audit), then verifies each artifact with controlled_verify.py on its own idle card.
Ends with controlled_ablation.collect(). Refuses to run twice.
"""
from concurrent.futures import ThreadPoolExecutor
import subprocess
import threading
import time

import common
from controlled_ablation import IMAGE, OBSERVER, OUT, STUDY, TASK, WORK, collect

lock = threading.Lock()
state = {'started_utc': common.utc_now(), 'schedule': 'parallel pair-wise freeze; outcome embargo until all four frozen', 'jobs': []}


def freeze_artifacts(roots, task):
    """{name: {root, files, bytes}} of a pair's artifacts (A then B), checked: same file set, both within the task's
    capacity, sizes within 0.5% of each other, every file but the weights byte-identical."""
    artifacts = {}
    for name, root in roots.items():
        files = common.manifest(root)
        artifacts[name] = {'root': str(root), 'files': files, 'bytes': sum(x['bytes'] for x in files.values())}
    a, b = artifacts.values()
    assert a['files'].keys() == b['files'].keys()
    assert max(a['bytes'], b['bytes']) <= common.limit_bytes(task)
    assert abs(b['bytes'] - a['bytes']) / a['bytes'] <= .005
    for p in a['files']:
        if not p.endswith('.safetensors'):
            assert a['files'][p]['sha256'] == b['files'][p]['sha256'], p
    return artifacts


def verification_command(image, docker_name, gpu, root, tests, dest):
    """controlled_verify.py on one frozen artifact, in the runtime image, on one card."""
    return [*common.docker_run(name=docker_name, gpu=gpu), '-e', f'STUDY_GPU_UUID={common.gpu_uuid(gpu)}',
            '-v', f'{root}:/app/submission:ro', '-v', f'{tests}:/tests:ro', '-v', f'{dest}:/result',
            '-v', f'{STUDY}:/study:ro', '--entrypoint', 'python3', image, '/study/controlled_verify.py']


def freeze_pair(kind):
    path = OUT / ('pair_freeze_' + kind + '.json')
    assert not path.exists()
    assert (OUT / (kind + '_build.json')).exists()
    if kind == 'moe':
        for name in ('stats.json', 'progress.json'):
            p = WORK / 'artifacts/moe_A' / name
            if p.exists():
                p.replace(OUT / ('moe_encoder_' + name))
        with (OUT / 'moe_tensor_audit.log').open('w') as log:
            subprocess.run([*common.docker_run(memory='64g'), '-v', f'{WORK}/artifacts/moe_A:/a:ro',
                            '-v', f'{WORK}/artifacts/moe_B:/b:ro', '-v', f'{OUT}:/record', '-v', f'{STUDY}:/study:ro',
                            '--entrypoint', 'python3', IMAGE, '/study/controlled_audit.py'],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    record = {'frozen_utc': common.utc_now(), 'task': kind, 'outcomes_accessed': False,
              'preregistration_sha256': common.file_sha256(OUT / 'preregistration.json'),
              'observer_sha256': common.file_sha256(OBSERVER),
              'artifacts': freeze_artifacts({kind + '_' + s: WORK / 'artifacts' / (kind + '_' + s) for s in 'AB'}, TASK[kind]),
              'evaluator': common.manifest(common.REPO / 'compression_tasks' / TASK[kind] / 'tests'),
              'image_id': subprocess.check_output(['docker', 'image', 'inspect', IMAGE, '--format', '{{.Id}}'], text=True).strip()}
    common.write_json(path, record)
    return record


def verify(name, frozen):
    with lock:
        gpu, held = common.claim_gpu('controlled-parallel-' + name, wait_s=10)
        kind = name.split('_')[0]
        root = WORK / 'artifacts' / name
        tests = common.REPO / 'compression_tasks' / TASK[kind] / 'tests'
        dest = OUT / 'verification' / name
        dest.mkdir(parents=True, exist_ok=False)
        assert common.manifest(root) == frozen['artifacts'][name]['files']
        assert common.file_sha256(OBSERVER) == frozen['observer_sha256']
        docker_name = 'controlled-verify-' + name.lower().replace('_', '-')
        cmd = verification_command(frozen['image_id'], docker_name, gpu, root, tests, dest)
        job = {'artifact': name, 'started_utc': common.utc_now(), 'gpu': gpu, 'gpu_uuid': common.gpu_uuid(gpu),
               'hardware': common.gpu_hardware(gpu), 'argv': cmd, 'status': 'running', 'pair_frozen_utc': frozen['frozen_utc']}
        state['jobs'].append(job)
        common.write_json(OUT / 'parallel_state.json', state)
        common.write_json(dest / 'configuration.json', job)
    t = time.monotonic()
    try:
        rc = common.run_container(cmd, dest / 'execution.log', docker_name, 5400)
        with lock:
            job.update(finished_utc=common.utc_now(), gpu_seconds=time.monotonic() - t, returncode=rc,
                       status='finished' if rc == 0 else 'execution_failed')
            common.write_json(dest / 'configuration.json', job)
            common.write_json(OUT / 'parallel_state.json', state)
    finally:
        held.release()
    # No fresh metric files are opened by this scheduler before all four freeze.


def main():
    assert not (OUT / 'parallel_state.json').exists(), 'refuse replacement verifications'
    common.write_json(OUT / 'parallel_state.json', state)
    with ThreadPoolExecutor(max_workers=4) as pool:
        while not (OUT / 'dense_build.json').exists():
            time.sleep(10)
        dense = freeze_pair('dense')
        futures = [pool.submit(verify, 'dense_' + s, dense) for s in 'AB']
        while not (OUT / 'moe_build.json').exists():
            time.sleep(15)
        moe = freeze_pair('moe')
        common.write_json(OUT / 'all_four_frozen.json', {'frozen_utc': common.utc_now(), 'pairs': ['dense', 'moe'],
                                                         'no_outcome_access_before_all_four_frozen': True})
        futures += [pool.submit(verify, 'moe_' + s, moe) for s in 'AB']
        for f in futures:
            f.result()
    state['finished_utc'] = common.utc_now()
    state['verification_gpu_hours'] = sum(j['gpu_seconds'] for j in state['jobs']) / 3600
    common.write_json(OUT / 'parallel_state.json', state)
    collect()


if __name__ == '__main__':
    main()
