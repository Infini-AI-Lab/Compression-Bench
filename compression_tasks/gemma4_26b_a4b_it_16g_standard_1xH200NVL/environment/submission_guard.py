"""Health probe for the task's single-copy submission lock.

`main`'s keepalive (`flock -s` in docker-compose.yaml) holds a shared lock on
the lock file; `ready` exits 0 once that lock is held, i.e. once no exclusive
lock can be taken.
"""
import argparse
import fcntl
from pathlib import Path

LOCK = Path('/run/compression-bench-submission.lock')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['ready'])
    parser.add_argument('--lock', type=Path, default=LOCK)
    args = parser.parse_args()
    # Bind-mounted read-only: Agent cannot replace the lock inode. LOCK_EX on
    # a read-only fd works with Linux flock (local Docker host filesystem).
    with args.lock.open('rb') as held:
        try:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
