#!/usr/bin/env bash
# Launch one evolve run of one task: resolve it, pick a card, find its checkpoint.
#
#   compression_bench/scripts/compress_evolve/evolve-task.sh <task id | task hash | task dir> [extra evolve args]
#
# Environment knobs (all optional):
#   GPU=auto|N          a free card is picked unless N is given (default auto)
#   HOURS=H             agent budget; default the contract's budget.agent_hours
#   AGENT_CLI=claude|codex   (default claude)
#   AGENT_MODEL, AGENT_EFFORT  (default claude-opus-5-5 / max for claude,
#                              gpt-5.6-sol / max for codex)
#   SEED_WORKSPACE=RUN  an ended run of the same lineage whose work/ files the fresh run starts with
#   AGENT_ENV=FILE      an API-key provider route: a KEY=VALUE file with ANTHROPIC_BASE_URL,
#                       ANTHROPIC_API_KEY, ANTHROPIC_MODEL...; kept outside the repository,
#                       mode 600 (templates: compression_bench/scripts/agent/routes/*.env.example)
#   MODEL_DIR=...       the pinned BF16 checkpoint (default: model_for_task.py)
#   DEBUG_MODEL=1       also mount the task's debug model when it has one
#   COMPRESSION_BENCH_EXCLUDED_GPUS="0 1"       cards never to use
#   COMPRESSION_BENCH_AUTO_EXCLUDED_GPUS="1"    cards never to pick automatically
#
# Detach for a real run and record the launcher's pid from ps:
#   setsid nohup compression_bench/scripts/compress_evolve/evolve-task.sh <task> \
#       < /dev/null > evolve.log 2>&1 &
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck disable=SC1091
. "$repo/compression_bench/scripts/env.sh"

if [ $# -lt 1 ]; then
    sed -n '2,23p' "$0" >&2
    exit 2
fi
task="$1"; shift

task_dir="$("$COMPRESSION_BENCH_PY" -c '
import sys
from lifecycle.run import resolve_task_dir
print(resolve_task_dir(sys.argv[1]))' "$task")"
title="$("$COMPRESSION_BENCH_PY" -c '
import sys
from lifecycle.utils import read_contract
c = read_contract(sys.argv[1])
print(c["task_id"], "-", c["description"])' "$task_dir")"
echo "task: $title"

AGENT_CLI="${AGENT_CLI:-claude}"
if [ "$AGENT_CLI" = codex ]; then
    AGENT_MODEL="${AGENT_MODEL:-gpt-5.6-sol}"; AGENT_EFFORT="${AGENT_EFFORT:-max}"
else
    AGENT_MODEL="${AGENT_MODEL:-claude-opus-5-5}"; AGENT_EFFORT="${AGENT_EFFORT:-max}"
fi

GPU="${GPU:-auto}"
if [ "$GPU" = auto ]; then
    GPU="$("$COMPRESSION_BENCH_PY" -m lifecycle.gpus pick)" || { echo "no free GPU" >&2; exit 1; }
else
    why="$("$COMPRESSION_BENCH_PY" -m lifecycle.gpus check "$GPU")" || { echo "GPU $GPU: $why" >&2; exit 1; }
fi
echo "gpu: $GPU"
reserved="$("$COMPRESSION_BENCH_PY" -m lifecycle.gpus reservations "$GPU")"
[ "$reserved" = none ] || echo "warning: GPU $GPU is reserved by idle container(s): $reserved" >&2

MODEL_DIR="${MODEL_DIR:-$("$COMPRESSION_BENCH_PY" "$repo/compression_bench/scripts/model_for_task.py" "$task_dir")}"
echo "model: $MODEL_DIR"

args=(--task "$task_dir" --gpu "$GPU" --model-dir "$MODEL_DIR"
      --agent-cli "$AGENT_CLI" --agent-model "$AGENT_MODEL" --agent-effort "$AGENT_EFFORT")
if [ -n "${HOURS:-}" ]; then
    args+=(--hours "$HOURS")
fi
if [ -n "${SEED_WORKSPACE:-}" ]; then
    args+=(--seed-workspace "$SEED_WORKSPACE")
    echo "seed workspace: $SEED_WORKSPACE"
fi
if [ -n "${AGENT_ENV:-}" ]; then
    [ -f "$AGENT_ENV" ] || { echo "AGENT_ENV $AGENT_ENV is not a file" >&2; exit 1; }
    args+=(--agent-env "$AGENT_ENV")
    echo "agent env: $AGENT_ENV (provider route)"
fi
if [ "${DEBUG_MODEL:-0}" = 1 ]; then
    if debug_dir="$("$COMPRESSION_BENCH_PY" "$repo/compression_bench/scripts/model_for_task.py" "$task_dir" --debug 2>/dev/null)"; then
        args+=(--debug-model-dir "$debug_dir")
        echo "debug model: $debug_dir"
    else
        echo "debug model: none found; running without it" >&2
    fi
fi

exec "$COMPRESSION_BENCH_PY" "$repo/compression_bench/scripts/evolve" "${args[@]}" "$@"
