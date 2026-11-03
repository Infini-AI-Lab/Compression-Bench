#!/bin/bash

set -uo pipefail

# Fail closed before any grading if main's keepalive still owns the shared lock.
# FD 9 remains open in this shell for the entire evaluation, including children.
exec 9</run/compression-bench-submission.lock || exit 75
python3 -c 'import fcntl; fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)' 9<&9 || {
    echo "Refusing verification: Agent submission is still writable by a live main container." >&2
    exit 75
}

export TORCH_DISABLE_NATIVE_JIT=1

# Python owns the stage sequence, logs and reward.txt; this shell keeps FD 9 locked.
python3 /tests/evaluate.py run /app/submission --output-dir /logs/verifier
