#!/usr/bin/env python3
"""Restore each task's official tokenizer files from the Hugging Face Hub.

The large tokenizer files are not stored in the repository. Each task pins its
source checkpoint (`model.id`, `model.revision` in `contract.yaml`); this script
downloads its tokenizer files (TOKENIZER_FILES) at that revision and writes them
into `assets/tokenizer/` and the two rendered copies (`environment/public-checker/evaluator-assets/tokenizer/`
and `tests/evaluator-assets/tokenizer/`).

Gated checkpoints need a token: HF_TOKEN (or HUGGING_FACE_HUB_TOKEN), else the
token saved by `huggingface-cli login` ($HF_HOME/token).

    python compression_bench/scripts/deploy/fetch_tokenizers.py            # every task
    python compression_bench/scripts/deploy/fetch_tokenizers.py <task> ...  # selected tasks
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import urllib.request

REPO = Path(__file__).resolve().parents[3]
TASKS = REPO / 'compression_tasks'
TOKENIZER_FILES = ('tokenizer.json',)
COPIES = ('assets/tokenizer',
          'environment/public-checker/evaluator-assets/tokenizer',
          'tests/evaluator-assets/tokenizer')


def task_model(task: Path) -> tuple[str, str]:
    text = (task / 'contract.yaml').read_text()
    match = re.search(r'^model:\n\s+id:\s*(\S+)\n\s+revision:\s*(\S+)', text, re.M)
    if not match:
        raise SystemExit(f'{task.name}: contract.yaml names no model id/revision')
    return match.group(1), match.group(2)


def download(model: str, revision: str, name: str, cache: Path) -> Path:
    target = cache / model.replace('/', '__') / revision / name
    if target.is_file() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f'https://huggingface.co/{model}/resolve/{revision}/{name}'
    request = urllib.request.Request(url, headers={'User-Agent': 'compression-bench-fetch'})
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    token_file = Path(os.environ.get('HF_HOME', Path.home() / '.cache' / 'huggingface')) / 'token'
    if not token and token_file.is_file():
        token = token_file.read_text().strip()
    if token:
        request.add_header('Authorization', f'Bearer {token}')
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
        with urllib.request.urlopen(request) as response:
            shutil.copyfileobj(response, tmp)
    os.replace(tmp.name, target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('tasks', nargs='*', help='task ids (default: every task)')
    parser.add_argument('--cache', type=Path,
                        default=Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
                        / 'compression-bench' / 'tokenizers')
    args = parser.parse_args()

    tasks = ([TASKS / t for t in args.tasks] if args.tasks
             else sorted(p.parent for p in TASKS.glob('*/contract.yaml')
                         if not p.parent.name.startswith('_')))
    for task in tasks:
        model, revision = task_model(task)
        for name in TOKENIZER_FILES:
            source = download(model, revision, name, args.cache)
            for copy in COPIES:
                directory = task / copy
                if directory.is_dir():
                    shutil.copyfile(source, directory / name)
        print(f'{task.name}: {", ".join(TOKENIZER_FILES)} from {model}@{revision[:12]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
