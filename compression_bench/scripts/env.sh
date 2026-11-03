# Source me: resolves this host's two interpreters and puts the harness first
# on PYTHONPATH. Never falls back to a bare `python3`.
#
#   COMPRESSION_BENCH_PY         runs the harness, taskctl and the tests
#                                (Python 3.10+, stdlib + pyyaml + pytest; no torch, no harbor)
#   COMPRESSION_BENCH_HARBOR_PY  runs scripts/agent/harbor_env.py (harbor==0.21.0)
#
# Resolution order for each: the gitignored compression_bench/scripts/env.local.sh
# (sourced first, so what it assigns wins), then the environment variable, then
# compression_bench/.venv/bin/python.
# env.local.sh is plain shell, for example:
#   COMPRESSION_BENCH_PY=/path/to/python3
#   COMPRESSION_BENCH_HARBOR_PY=/path/to/harbor-venv/bin/python
#   export COMPRESSION_BENCH_MODELS_ROOT=/path/to/models
#   export COMPRESSION_BENCH_AUTO_EXCLUDED_GPUS="0"
_compression_bench_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ -f "$_compression_bench_repo/compression_bench/scripts/env.local.sh" ]; then
    # shellcheck disable=SC1091
    . "$_compression_bench_repo/compression_bench/scripts/env.local.sh"
fi
_compression_bench_venv="$_compression_bench_repo/compression_bench/.venv/bin/python"
: "${COMPRESSION_BENCH_PY:=$([ -x "$_compression_bench_venv" ] && echo "$_compression_bench_venv")}"
: "${COMPRESSION_BENCH_HARBOR_PY:=$([ -x "$_compression_bench_venv" ] && echo "$_compression_bench_venv")}"

_compression_bench_refuse() { echo "env.sh: $*" >&2; return 1 2>/dev/null || exit 1; }

if [ -z "${COMPRESSION_BENCH_PY:-}" ] || ! "$COMPRESSION_BENCH_PY" -c 'import sys, yaml; assert sys.version_info >= (3, 10)' 2>/dev/null; then
    _compression_bench_refuse "COMPRESSION_BENCH_PY is unset or lacks Python>=3.10 + pyyaml; bind it in compression_bench/scripts/env.local.sh"
fi
if [ -z "${COMPRESSION_BENCH_HARBOR_PY:-}" ] || ! "$COMPRESSION_BENCH_HARBOR_PY" -c 'from importlib.metadata import version; assert version("harbor") == "0.21.0"' 2>/dev/null; then
    _compression_bench_refuse "COMPRESSION_BENCH_HARBOR_PY is unset or does not have harbor==0.21.0; create compression_bench/.venv or bind it in compression_bench/scripts/env.local.sh"
fi

export COMPRESSION_BENCH_REPO="$_compression_bench_repo" COMPRESSION_BENCH_PY COMPRESSION_BENCH_HARBOR_PY
export PYTHONPATH="$_compression_bench_repo/compression_bench/harness:$_compression_bench_repo${PYTHONPATH:+:$PYTHONPATH}"
unset _compression_bench_repo _compression_bench_venv
