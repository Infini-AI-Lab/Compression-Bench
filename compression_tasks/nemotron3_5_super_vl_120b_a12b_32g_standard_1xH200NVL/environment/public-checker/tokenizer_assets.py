"""Task-owned tokenizer packaging; stdlib only, no model or GPU load.

The official tokenizer files and which package files the private evaluation
view links (rather than copies) come from the task policy.
"""

from contextlib import contextmanager
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import task_policy


OFFICIAL_FILES = tuple(task_policy.current()['official_tokenizer_files'])
RESERVED_FILES = frozenset((*OFFICIAL_FILES, 'vocab.json', 'merges.txt',
                            'special_tokens_map.json', 'added_tokens.json',
                            'tokenizer.model', 'spiece.model', 'chat_template.jinja',
                            'chat_template.json'))


def is_tokenizer_asset(relative: Path) -> bool:
    return (relative.parts[0] == 'chat_templates'
            or (len(relative.parts) == 1 and relative.name in RESERVED_FILES))


def is_linked_payload(relative: Path, path: Path) -> bool:
    """Large weight files enter the private view as links into the read-only
    source; everything else is copied."""
    if task_policy.current()['link_rule'] == 'safetensors-or-gguf':
        return path.suffix in ('.safetensors', '.gguf')
    return relative.parts[0] == 'model' and path.suffix not in ('.json', '.py', '.md', '.txt')


def official_files(tokenizer_dir: Path) -> list[Path]:
    files = [tokenizer_dir / name for name in OFFICIAL_FILES]
    for path in files:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'missing or invalid task-owned tokenizer file: {path}')
    return files


def packaged_inventory(submission: Path, tokenizer_dir: Path) -> dict:
    """Measure final bytes, replacing any Candidate tokenizer assets with ours."""
    model = submission / 'model'
    if submission.is_symlink() or model.is_symlink() or not model.is_dir():
        raise ValueError('submission/model must be a real directory')
    trusted = official_files(tokenizer_dir)
    total_bytes = sum(path.stat().st_size for path in trusted)
    tokenizer_bytes = total_bytes
    file_count = len(trusted)
    for path in submission.rglob('*'):
        if path.is_symlink():
            raise ValueError(f'symbolic links are not allowed: {path}')
        if not path.is_file() and not path.is_dir():
            raise ValueError(f'unsupported submission entry: {path}')
        relative = path.relative_to(submission)
        owned = (relative.parts[0] == 'model' and len(relative.parts) > 1
                 and is_tokenizer_asset(Path(*relative.parts[1:])))
        if owned:
            if path.is_dir() and relative.parts[1] in RESERVED_FILES:
                raise ValueError(f'tokenizer filename must not be a directory: {path}')
            continue
        if path.is_file():
            total_bytes += path.stat().st_size
            file_count += 1
    return {'total_bytes': total_bytes, 'file_count': file_count,
            'tokenizer_bytes': tokenizer_bytes, 'tokenizer_source': 'task-owned',
            'size_basis': 'final package including official tokenizer'}


@contextmanager
def prepared_submission(submission: Path, tokenizer_dir: Path):
    """Private evaluation view: official assets first; no weight-file copies.

    Candidate symlinks are rejected before construction. Only this trusted helper
    creates weight links, pointing into the evaluator's read-only source mount.
    The private view is never returned to the Agent or archived as a submission.
    """
    expected = packaged_inventory(submission, tokenizer_dir)
    source = submission.resolve()
    with tempfile.TemporaryDirectory(prefix='quant-package-') as temp_dir:
        package = Path(temp_dir) / 'submission'
        package.mkdir()
        for path in source.rglob('*'):
            relative = path.relative_to(source)
            # Recheck before linking/copying; Agent-created links are never used.
            if path.is_symlink():
                raise ValueError(f'symbolic links are not allowed: {path}')
            if (relative.parts[0] == 'model' and len(relative.parts) > 1
                    and is_tokenizer_asset(Path(*relative.parts[1:]))):
                continue
            target = package / relative
            if path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif path.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                if is_linked_payload(relative, path):
                    target.symlink_to(path)
                else:
                    shutil.copyfile(path, target)
            else:
                raise ValueError(f'unsupported submission entry: {path}')
        for official in official_files(tokenizer_dir):
            target = package / 'model' / official.name
            shutil.copyfile(official, target)
            target.chmod(0o444)
        actual = sum(p.stat().st_size for p in package.rglob('*') if p.is_file())
        if actual != expected['total_bytes']:
            raise ValueError('submission size changed during evaluation packaging')
        yield package


def finalize_submission(submission: Path, tokenizer_dir: Path) -> dict:
    """Materialize the same official assets in the archive after evaluation."""
    expected = packaged_inventory(submission, tokenizer_dir)
    model = submission / 'model'
    if tokenizer_dir.resolve() == model.resolve() or tokenizer_dir.resolve().is_relative_to(model.resolve()):
        raise ValueError('official tokenizer source must be outside submission/model')
    # Preflight above rejects symlinks and special files before any mutation.
    for path in model.iterdir():
        if path.name in RESERVED_FILES:
            path.unlink()
        elif path.name == 'chat_templates':
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    for source in official_files(tokenizer_dir):
        with tempfile.NamedTemporaryFile(dir=model, prefix='.official-tokenizer-', delete=False) as output:
            temporary = Path(output.name)
            try:
                with source.open('rb') as input_file:
                    shutil.copyfileobj(input_file, output)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            temporary.chmod(0o444)
            os.replace(temporary, model / source.name)
        finally:
            temporary.unlink(missing_ok=True)
    actual = sum(path.stat().st_size for path in submission.rglob('*') if path.is_file())
    if actual != expected['total_bytes']:
        raise ValueError('final package size changed during packaging')
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('submission', type=Path)
    parser.add_argument('--tokenizer-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize_submission(args.submission, args.tokenizer_dir)))


if __name__ == '__main__':
    main()
