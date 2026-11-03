#!/usr/bin/env python3
"""Download the papers of the offline library from their original sources; present files need no download.

The papers under `compression_bench/environments/agent-base/library/` are not
tracked. Two lists name every one of them and where it is published:

- `compression_bench/environments/reference-lock.json`: the papers of the
  Awesome-LLM-Compression catalogue and the curated references, each a library
  path and the URL of its publisher's PDF (arXiv at a fixed version, OpenReview,
  PMLR, ACL Anthology, conference proceedings);
- `compression_bench/environments/agent-base/papers/supplement.json`: the arXiv
  supplement, each paper's versioned arXiv PDF, kept at `supplement/arxiv-<id>.pdf`.

A missing PDF is downloaded from its URL, at least 1 s between requests to one
host. The agent reads each paper through its text extract beside the PDF
(`.txt`, `pdftotext -layout -enc UTF-8`, poppler); a missing extract is made
from the PDF.

    prepare_references.py            # download and extract what is missing
    prepare_references.py --verify   # report what is missing; download and write nothing
"""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCK = ROOT / 'compression_bench/environments/reference-lock.json'
DEFAULT_SUPPLEMENT = ROOT / 'compression_bench/environments/agent-base/papers/supplement.json'
DEFAULT_LIBRARY = ROOT / 'compression_bench/environments/agent-base/library'
SUPPLEMENT_DIR = 'supplement'
ARXIV_ID = re.compile(r'\d{4}\.\d{4,5}')
USER_AGENT = 'compression-bench-reference-fetcher'
REQUEST_INTERVAL_S = 1.0
PDFTOTEXT = ('pdftotext', '-layout', '-enc', 'UTF-8')


def safe_path(root, name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or str(p) != name:
        raise ValueError(f'Unsafe library path: {name}')
    target = root.joinpath(*p.parts)
    # Library files are files, never symlinks (including parents).
    if any(part.is_symlink() for part in [target, *target.parents]):
        raise ValueError(f'Symlink in library destination: {name}')
    return target


def text_of(pdf):
    """The text extract's library path: beside the PDF, `.txt`."""
    return pdf[:-len('.pdf')] + '.txt'


def load_lock(path):
    """The catalogue and curated papers: [{"pdf": library path, "url": https URL}]."""
    data = json.loads(path.read_text())
    if data.get('schema_version') != 2:
        raise ValueError(f'{path.name}: unsupported schema_version {data.get("schema_version")}')
    seen = set()
    for paper in data['papers']:
        safe_path(Path('/library-validation-root'), paper['pdf'])
        if not paper['pdf'].endswith('.pdf') or paper['pdf'] in seen:
            raise ValueError(f'Invalid or duplicate library path: {paper["pdf"]}')
        if urllib.parse.urlparse(paper['url']).scheme != 'https':
            raise ValueError(f'Not an https source: {paper["url"]}')
        seen.add(paper['pdf'])
    return data['papers']


def load_supplement(path):
    """The arXiv supplement as the same records."""
    data = json.loads(path.read_text())
    if data.get('schema_version') != 1:
        raise ValueError('Unsupported supplement version')
    seen = set()
    papers = []
    for entry in data['papers']:
        arxiv, version = entry['arxiv_id'], entry['version']
        if not ARXIV_ID.fullmatch(arxiv) or not re.fullmatch(r'v[1-9]\d*', version) or arxiv in seen:
            raise ValueError(f'Invalid or duplicate supplement entry: {arxiv}{version}')
        if entry['pdf_url'] != f'https://arxiv.org/pdf/{arxiv}{version}':
            raise ValueError(f'Supplement PDF URL is not the pinned version: {arxiv}{version}')
        seen.add(arxiv)
        papers.append({'pdf': f'{SUPPLEMENT_DIR}/arxiv-{arxiv}.pdf', 'url': entry['pdf_url']})
    return papers


def present(path):
    return path.is_file() and not path.is_symlink() and path.stat().st_size > 0


def extract_text(pdf, target):
    """`pdftotext -layout -enc UTF-8 PDF TXT`, written atomically; how the library's extracts are made."""
    if not shutil.which(PDFTOTEXT[0]):
        raise ValueError('pdftotext (poppler-utils) is needed to make the paper text extracts')
    with tempfile.TemporaryDirectory(prefix='.paper-text-', dir=target.parent) as tmp:
        out = Path(tmp) / target.name
        subprocess.run([*PDFTOTEXT, str(pdf), str(out)], check=True, capture_output=True, timeout=600)
        if not out.is_file() or out.stat().st_size == 0:
            raise ValueError(f'Empty text extract: {target.name}')
        out.chmod(0o644)
        os.replace(out, target)


_last_request = {}


def download_pdf(url, target):
    """The PDF at `url`, written atomically; refused unless the response is a PDF."""
    host = urllib.parse.urlparse(url).netloc
    wait = REQUEST_INTERVAL_S - (time.monotonic() - _last_request.get(host, float('-inf')))
    if wait > 0:
        time.sleep(wait)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.download-', delete=False) as stream:
        tmp = Path(stream.name)
        try:
            request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
            _last_request[host] = time.monotonic()
            with urllib.request.urlopen(request, timeout=180) as response:
                shutil.copyfileobj(response, stream)
            stream.close()
            with tmp.open('rb') as head:
                if head.read(5) != b'%PDF-':
                    raise ValueError(f'Not a PDF: {url}')
            tmp.chmod(0o644)
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)


def restore(papers, destination, verify=False):
    """Download each missing PDF, make each missing extract; returns the number of papers."""
    fetch, extract = [], []
    for paper in papers:
        pdf = safe_path(destination, paper['pdf'])
        text = safe_path(destination, text_of(paper['pdf']))
        if not present(pdf):
            fetch.append((paper['url'], pdf))
        if not present(text) or not present(pdf):
            extract.append((pdf, text))
    if verify:
        if fetch or extract:
            raise ValueError(f'{len(fetch)} paper PDFs and {len(extract)} text extracts are missing')
        print(f'Library verified: {len(papers)} papers.', flush=True)
        return len(papers)
    failed = []
    for n, (url, pdf) in enumerate(fetch, 1):
        try:
            download_pdf(url, pdf)
        except (OSError, ValueError) as error:
            failed.append(f'{pdf.relative_to(destination)} ({url}): {error}')
        if n % 100 == 0:
            print(f'Downloaded {n}/{len(fetch)} PDFs', flush=True)
    for pdf, text in extract:
        if present(pdf):
            extract_text(pdf, text)
    if failed:
        raise ValueError(f'{len(failed)} PDFs could not be downloaded:\n  ' + '\n  '.join(failed))
    print(f'Library ready: {len(papers)} papers; downloaded {len(fetch)} PDFs, '
          f'made {len(extract)} text extracts.', flush=True)
    return len(papers)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--lock', type=Path, default=DEFAULT_LOCK)
    parser.add_argument('--supplement', type=Path, default=DEFAULT_SUPPLEMENT)
    parser.add_argument('--destination', type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument('--verify', action='store_true', help='Report missing files without downloading or writing')
    args = parser.parse_args()
    try:
        restore(load_lock(args.lock) + load_supplement(args.supplement), args.destination, args.verify)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f'Reference preparation failed: {error}\n')


if __name__ == '__main__':
    main()
