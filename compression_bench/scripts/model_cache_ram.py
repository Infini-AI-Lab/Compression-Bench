#!/usr/bin/env python3
"""Move an idle, pinned HF checkpoint's weights to RAM to free disk, and verify its container mounts.

    model_cache_ram.py spill compression_tasks/<task>    # the task's cached checkpoint -> RAM
    model_cache_ram.py watch <record.json>               # free the RAM once the cache is evicted

The HF cache remains readable. Only checksum-verified download-cache blobs move;
experiment checkpoints are not accepted. Each spill writes an operation record
under `compression_bench/evolve_space/model-cache-ram/`; `watch RECORD` releases
the RAM once the scheduler has evicted the owning HF cache.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'compression_bench/harness'), str(Path(__file__).parent)]
from lifecycle import containers
from lifecycle.utils import read_contract
import model_for_task


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def link(target, path):
    temp = path.with_name(path.name + '.compression-bench-ram-link')
    temp.unlink(missing_ok=True)
    temp.symlink_to(target)
    os.replace(temp, path)


def idle(model_id):
    queue = json.loads((REPO / 'compression_bench/evolve_space/scheduler/queue.json').read_text())
    for item in queue['items']:
        if item.get('status') != 'running':
            continue
        contract = read_contract(REPO / 'compression_tasks' / item['task'])
        if contract['model']['id'] == model_id:
            raise RuntimeError(f"Cannot migrate an active evolve source: {item['id']}")


def spill(task):
    contract = read_contract(task)
    identity = contract['model']
    model = model_for_task.resolve(task)
    assert model is not None
    model = model.resolve()
    cache = model_for_task.hf_cache() / ('models--' + identity['id'].replace('/', '--'))
    assert model == cache / 'snapshots' / identity['revision']
    assert sorted(p.name for p in (cache / 'snapshots').iterdir()) == [identity['revision']]
    assert not list((cache / 'blobs').glob('*.incomplete')), 'Download still active'
    idle(identity['id'])
    index = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
    names = sorted(set(index.values()))
    assert names and all(n.endswith('.safetensors') for n in names)
    uid = os.getuid()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    suffix = cache.name + '-' + identity['revision'][:8] + '-' + stamp
    roots = [Path(f'/run/user/{uid}/compression-bench-hf-ram') / suffix,
             Path(f'/dev/shm/compression-bench-hf-ram-{uid}') / suffix]
    for root in roots:
        root.mkdir(parents=True, mode=0o700)
        assert root.stat().st_uid == uid and not root.is_symlink()
    private_budget = min(132 * 2**30, shutil.disk_usage(roots[0]).free - 20 * 2**30)
    used_private = 0
    rows = []
    for name in names:
        source = model / name
        assert source.is_symlink()
        blob = source.resolve()
        assert blob.parent == cache / 'blobs' and re.fullmatch('[0-9a-f]{64}', blob.name)
        size = blob.stat().st_size
        target_root = roots[0] if used_private + size <= private_budget else roots[1]
        if target_root == roots[0]:
            used_private += size
        rows.append({'filename': name, 'bytes': size, 'sha256': blob.name,
                     'disk_blob': str(blob), 'ram_blob': str(target_root / blob.name),
                     'original_snapshot_link': os.readlink(source)})
    total = sum(r['bytes'] for r in rows)
    available = int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines()
                         if l.startswith('MemAvailable:'))) * 1024
    assert available > total + 64 * 2**30
    assert shutil.disk_usage(roots[1]).free > total - used_private + 24 * 2**30
    record = REPO / 'compression_bench/evolve_space/model-cache-ram' / f'hf-ram-{stamp}.json'
    record.parent.mkdir(parents=True, exist_ok=True)
    data = {'status': 'copying', 'started_utc': stamp, 'model': identity, 'model_dir': str(model),
            'cache': str(cache), 'ram_roots': [str(p) for p in roots], 'files': rows,
            'checkpoint_bytes': total, 'config_sha256': sha(model / 'config.json'),
            'filesystem_free_before_bytes': shutil.disk_usage(REPO).free}
    write(record, data)
    print('Operation record:', record, flush=True)
    for i, row in enumerate(rows):
        target = Path(row['ram_blob'])
        reserve = 20 if target.parent == roots[0] else 24
        assert shutil.disk_usage(target.parent).free > row['bytes'] + reserve * 2**30
        temp = target.with_suffix('.partial')
        shutil.copy2(row['disk_blob'], temp)
        assert temp.stat().st_size == row['bytes'] and sha(temp) == row['sha256'], row['filename']
        os.replace(temp, target)
        print(f"Verified RAM copy {i + 1}/{len(rows)}: {row['filename']}", flush=True)
    idle(identity['id'])
    # Keep the checkpoint unavailable to new lookups only during publication.
    # All disk blobs remain intact until the complete container test passes.
    config = model / 'config.json'
    held_config = model / '.compression-bench-ram-config.json'
    os.replace(config, held_config)
    try:
        for row in rows:
            link(row['ram_blob'], model / row['filename'])
    except BaseException:
        for row in rows:
            link(row['original_snapshot_link'], model / row['filename'])
        raise
    finally:
        os.replace(held_config, config)
    try:
        assert model_for_task.is_checkpoint(model)
        overlay = containers.model_link_mounts(model)
        volumes = overlay['services']['main']['volumes']
        command = ['docker', 'run', '--rm', '--runtime', 'runc', '--ipc=private', '--network=none',
                   '--user', f'{uid}:{os.getgid()}', '-e', 'CUDA_VISIBLE_DEVICES=',
                   '-e', 'NVIDIA_VISIBLE_DEVICES=void', '-v', f'{model}:/input/model:ro']
        for volume in volumes:
            command += ['-v', f"{volume['source']}:{volume['target']}:ro"]
        code = '''import json,sys,hashlib
from pathlib import Path
from safetensors import safe_open
p=Path('/input/model'); expected=json.loads(sys.argv[1])
assert hashlib.sha256((p/'config.json').read_bytes()).hexdigest()==sys.argv[2]
for name,size in expected.items():
 assert (p/name).stat().st_size==size,name
 with safe_open(p/name,framework='pt',device='cpu') as f: assert len(f.keys())>0,name
print(json.dumps({'status':'passed','indexed_shards':len(expected),'ipc_mode':'private','gpu_used':False}))
'''
        image = json.loads((REPO / 'compression_bench/environments/images.json').read_text())['platform-gguf']['tag']
        command += ['--entrypoint', 'python3', image, '-c', code,
                    json.dumps({r['filename']: r['bytes'] for r in rows}), data['config_sha256']]
        result = subprocess.run(command, text=True, capture_output=True, timeout=180)
        assert result.returncode == 0, result.stderr[-3000:]
        data['container_validation'] = json.loads(result.stdout.strip().splitlines()[-1])
        data['container_mounts'] = volumes
    except BaseException:
        for row in rows:
            link(row['original_snapshot_link'], model / row['filename'])
        data['status'] = 'validation_failed_disk_blobs_preserved'
        write(record, data)
        raise
    for row in rows:
        link(row['ram_blob'], Path(row['disk_blob']))
    data.update(status='complete', filesystem_free_after_bytes=shutil.disk_usage(REPO).free,
                finished_utc=datetime.now(timezone.utc).isoformat())
    write(record, data)
    print(f"Moved {total / 2**30:.3f} GiB of verified cache weights to RAM; free disk: "
          f"{data['filesystem_free_after_bytes'] / 2**30:.1f} GiB", flush=True)
    print('Container checkpoint validation passed; cache remains available to the scheduler', flush=True)
    return record


def watch(record):
    data = json.loads(record.read_text())
    assert data['status'] == 'complete'
    data['watch_pid'] = os.getpid()
    write(record, data)
    while Path(data['cache']).exists():
        time.sleep(60)
    for raw in data['ram_roots']:
        root = Path(raw)
        assert root.stat().st_uid == os.getuid() and not root.is_symlink()
        assert root.parent in (Path(f'/run/user/{os.getuid()}/compression-bench-hf-ram'),
                               Path(f'/dev/shm/compression-bench-hf-ram-{os.getuid()}'))
        shutil.rmtree(root)
    data.update(ram_status='released_after_cache_eviction', released_utc=datetime.now(timezone.utc).isoformat())
    write(record, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('spill', 'watch'))
    parser.add_argument('target', type=Path)
    args = parser.parse_args()
    if args.action == 'spill':
        spill(args.target.resolve())
    else:
        watch(args.target.resolve())
