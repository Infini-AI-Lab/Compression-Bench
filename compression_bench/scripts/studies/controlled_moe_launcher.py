#!/usr/bin/env python3
"""Build the MoE pair (moe_A selected, moe_B ablated) from a selected run's retained source; refuse overwrite.

    controlled_moe_launcher.py --source RUN_DIR [--model-dir DIR]

RUN_DIR is the selected MoE run's directory (its `submission/` and `work/`, with the encoder at work/tools/encode.py
and its calibration tokens under work/data/). --model-dir is the task's BF16 checkpoint (default: model_for_task.py).
controlled_moe_build.py re-runs the selected encoder in the platform image, writing both models in one pass. After
the build, source_reproduction_audit.py compares the rebuilt moe_A with the selected submission's weights
(moe_source_reproduction_audit.json).
"""
import argparse
from pathlib import Path
import subprocess

import common
from controlled_ablation import IMAGE, OUT, STUDY, TASK, WORK


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', required=True)
    p.add_argument('--model-dir')
    a = p.parse_args()
    src = Path(a.source).resolve()
    model = a.model_dir or common.model_dir(TASK['moe'])
    common.write_json(OUT / 'moe_source.json', {'source_run_dir': str(src), 'model_dir': str(model), 'recorded_utc': common.utc_now()})
    for name in ['moe_A', 'moe_B']:
        d = WORK / 'artifacts' / name
        if d.exists():
            assert not list((d / 'model').glob('*.safetensors')), 'refuse overwrite built model'
        else:
            d.mkdir(parents=True)
            subprocess.run(['cp', '-a', '--reflink=auto', str(src / 'submission') + '/.', str(d)], check=True)
        for f in (d / 'model').glob('*.safetensors'):
            f.unlink()
    study = ['-v', f'{OUT}:/record', '-v', f'{STUDY}:/study:ro']
    gpu, claim = common.claim_gpu('controlled-ablation-moe-build')
    try:
        name = 'controlled-moe-build'
        cmd = [*common.docker_run(name=name, gpu=gpu),
               '-v', f'{src}/work:/archive/work:ro', '-v', f'{model}:/input/model:ro', *common.hf_cache_mounts(model),
               '-v', f'{WORK}/artifacts/moe_A:/selected', '-v', f'{WORK}/artifacts/moe_B:/alternate', *study,
               '--entrypoint', 'python3', IMAGE, '/study/controlled_moe_build.py']
        common.write_json(OUT / 'moe_build_launch.json', {'gpu': gpu, 'argv': cmd, 'started_utc': common.utc_now()})
        rc = common.run_container(cmd, OUT / 'moe_build.log', name, 5400)
        assert rc == 0, rc
    finally:
        claim.release()
    audit = [*common.docker_run(memory='64g'),
             '-v', f'{src}/submission/model:/original:ro', '-v', f'{WORK}/artifacts/moe_A/model:/reproduced:ro', *study,
             '--entrypoint', 'python3', IMAGE, '/study/source_reproduction_audit.py']
    with (OUT / 'moe_source_reproduction_audit.log').open('w') as f:
        subprocess.run(audit, stdout=f, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
