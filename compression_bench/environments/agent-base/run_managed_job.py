#!/usr/bin/env python3
"""Run one job and return once on completion/failure/deadline; no model polling."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid


def run(command, log, timeout):
    path = Path(log).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    markers = Path(os.environ.get('COMPRESSION_BENCH_JOB_DIR', '/app/work/compression-bench-jobs'))
    markers.mkdir(parents=True, exist_ok=True)
    base = markers / ('managed-' + uuid.uuid4().hex)
    child = None
    def stop():
        if child is not None:
            try: os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError: return
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired: pass
            try: os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            child.wait()
    def interrupted(signum, frame):
        raise InterruptedError(signum)
    old = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    rc = 1
    status = 'failed'
    try:
        with path.open('w') as out:
            child = subprocess.Popen(['bash', '-c', command], stdin=subprocess.DEVNULL,
                                     stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
            base.with_suffix('.pid').write_text(str(child.pid))
            base.with_suffix('.running').touch()
            try:
                rc = child.wait(timeout=timeout)
                status = 'completed' if rc == 0 else 'failed'
            except subprocess.TimeoutExpired:
                rc, status = 124, 'timed_out'
                stop()
    except InterruptedError as exc:
        rc, status = 128 + exc.args[0], 'interrupted'
        stop()
    finally:
        for s, handler in old.items(): signal.signal(s, handler)
        base.with_suffix('.exit').write_text(str(rc))
        base.with_suffix('.running').unlink(missing_ok=True)
        base.with_suffix('.done').write_text(status)
    print(f'Job {status}; exit={rc}; log={path}', flush=True)
    # Bound file I/O and tool output even for very large logs.
    if path.exists():
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - 16000))
            print('\n'.join(stream.read().decode(errors='replace').splitlines()[-20:]))
    return rc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--command', required=True)
    parser.add_argument('--log', required=True)
    parser.add_argument('--timeout', type=float, default=7200)
    args = parser.parse_args()
    if not 0 < args.timeout <= 28800: parser.error('timeout must be in (0, 28800]')
    return run(args.command, args.log, args.timeout)


if __name__ == '__main__': sys.exit(main())
