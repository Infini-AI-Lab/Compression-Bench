#!/usr/bin/env bash
# The gate a change passes before it is committed: every task in step with its
# sources, the harness importable, the CPU suite green. No GPU, model or network.
#
#   bash compression_bench/scripts/correctness-gate.sh [--quick]    # --quick skips evaluator parity
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
. "$repo/compression_bench/scripts/env.sh"
cd "$repo"

echo "== taskctl check"
"$COMPRESSION_BENCH_PY" compression_tasks/_manage/taskctl.py check

echo "== harness imports"
"$COMPRESSION_BENCH_PY" - <<'EOF'
import importlib, pkgutil, sys
sys.path.insert(0, "compression_bench/harness")
import lifecycle
for info in pkgutil.iter_modules(lifecycle.__path__):
    importlib.import_module(f"lifecycle.{info.name}")
print("lifecycle modules import")
EOF

echo "== tests"
# Test scratch goes under the gitignored run space, not the host's /tmp: the
# evaluator parity tests copy whole task directories.
basetemp="$repo/compression_bench/evolve_space/pytest-tmp"
mkdir -p "$repo/compression_bench/evolve_space"
trap 'rm -rf "$basetemp"' EXIT
if [[ ${1:-} == --quick ]]; then
    "$COMPRESSION_BENCH_PY" -m pytest compression_bench/tests -q -p no:cacheprovider --basetemp="$basetemp" -k "not parity"
else
    "$COMPRESSION_BENCH_PY" -m pytest compression_bench/tests -q -p no:cacheprovider --basetemp="$basetemp"
fi
echo "COMPRESSION_BENCH_CORRECTNESS_READY"
