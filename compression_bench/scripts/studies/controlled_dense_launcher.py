#!/usr/bin/env python3
"""Build the dense pair (dense_A selected, dense_B ablated) from a selected run's retained source; refuse overwrite.

    controlled_dense_launcher.py --source RUN_DIR [--model-dir DIR]

RUN_DIR is the selected dense run's directory (its `submission/` and `work/`, with work/build/quantize.py and the
cached Hessians under work/hess-calib/). --model-dir is the task's BF16 checkpoint (default: model_for_task.py).
dense_A is a copy of the selected submission; controlled_build.py writes dense_B's weights in the platform image.
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
    model = a.model_dir or common.model_dir(TASK['dense'])
    common.write_json(OUT / 'dense_source.json', {'source_run_dir': str(src), 'model_dir': str(model), 'recorded_utc': common.utc_now()})
    for name in ['dense_A', 'dense_B']:
        d = WORK / 'artifacts' / name
        if d.exists():
            raise RuntimeError('refuse overwrite ' + name)
        d.mkdir(parents=True)
        subprocess.run(['cp', '-a', '--reflink=auto', str(src / 'submission') + '/.', str(d)], check=True)
    scratch = WORK / 'dense_chunks'
    scratch.mkdir(exist_ok=True)
    for f in (WORK / 'artifacts/dense_B/model').glob('*.safetensors'):
        f.unlink()
    gpu, claim = common.claim_gpu('controlled-ablation-dense-build')
    try:
        name = 'controlled-dense-build'
        cmd = [*common.docker_run(name=name, gpu=gpu),
               '-v', f'{src}/work:/archive/work:ro', '-v', f'{model}:/input/model:ro',
               '-v', f'{WORK}/artifacts/dense_A:/selected:ro', '-v', f'{WORK}/artifacts/dense_B:/alternate',
               '-v', f'{scratch}:/scratch', '-v', f'{OUT}:/record', '-v', f'{STUDY}:/study:ro',
               '--entrypoint', 'python3', IMAGE, '/study/controlled_build.py']
        common.write_json(OUT / 'dense_build_launch.json', {'gpu': gpu, 'argv': cmd, 'started_utc': common.utc_now()})
        rc = common.run_container(cmd, OUT / 'dense_build.log', name, 4800)
        assert rc == 0, rc
    finally:
        claim.release()


if __name__ == '__main__':
    main()
