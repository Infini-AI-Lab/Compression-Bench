#!/usr/bin/env python3
"""Archive each measured GPTQ latency record, then delete its temporary weights.

    clean_baseline_weights.py [--watch] [--interval S]

A GPTQ checkpoint is deleted only once its record (decode samples, hidden PPL, correctness gate) is complete
and verified and its runtime and recipe are archived; each deletion is logged in `<root>/weights-cleanup.jsonl`.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time

from archive_baseline_setting import ROOT, REPO, archive, digest, read
from collect_baseline_latency import SETTINGS, validate, validate_quality


def sweep():
    complete = 0
    # Only temporary (regenerated or staged) checkpoints under these roots are ever deleted.
    allowed = [REPO / 'compression_bench/evolve_space/baselines/latency-regenerated',
               Path(f'/dev/shm/compression-bench-baseline-packed-{os.getuid()}'),
               Path(f'/run/user/{os.getuid()}/compression-bench-baseline-quantized')]
    for _, stem, capacities in SETTINGS:
        for capacity in capacities:
            setting = f'{stem}_{capacity}g'
            record = ROOT / 'gptq' / setting
            try:
                summary, config = read(record / 'SUMMARY.json'), read(record / 'configuration.json')
            except (OSError, json.JSONDecodeError):
                continue
            if summary.get('status') != 'measured' or any(summary.get(key, {}).get('status') != 'measured'
                    for key in ('ppl_verification', 'gate_verification')):
                continue
            complete += 1
            model = Path(config['model_dir'])
            if not model.exists():
                continue
            assert not model.is_symlink() and model.name == 'model' and model.parent.name == setting
            assert model.stat().st_uid == os.getuid()
            assert any(model.resolve().is_relative_to(base.resolve()) for base in allowed), model
            validate(record, summary)
            validate_quality(record, summary, config)
            archive('gptq', setting)
            before = shutil.disk_usage(model).free
            size = sum(p.stat().st_size for p in model.rglob('*') if p.is_file() and not p.is_symlink())
            event = {'utc': datetime.now(timezone.utc).isoformat(), 'model_dir': str(model),
                     'setting': setting, 'checkpoint_bytes': size,
                     'configuration_sha256': digest(record / 'configuration.json'),
                     'summary_sha256': digest(record / 'SUMMARY.json'),
                     'reason': 'Nine decode samples, PPL, and accuracy are complete and verified; exact runtime and recipe archived.',
                     'status': 'planned', 'filesystem_free_before_bytes': before}
            log = ROOT / 'weights-cleanup.jsonl'
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('a') as stream:
                stream.write(json.dumps(event) + '\n')
            shutil.rmtree(model)
            event.update(status='complete', filesystem_free_after_bytes=shutil.disk_usage(model.parent).free)
            with log.open('a') as stream:
                stream.write(json.dumps(event) + '\n')
            print(f'Deleted {setting}: {size / 2**30:.3f} GiB of temporary weights; all experiment data preserved', flush=True)
    return complete


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=float, default=30)
    args = parser.parse_args()
    while True:
        finished = sweep()
        if not args.watch or finished == sum(len(c) for _, _, c in SETTINGS):
            break
        time.sleep(args.interval)
