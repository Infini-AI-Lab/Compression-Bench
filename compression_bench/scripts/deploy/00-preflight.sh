#!/usr/bin/env bash
# Can this host run compression-bench? Read-only: it changes nothing.
#
#   bash compression_bench/scripts/deploy/00-preflight.sh [task ...]
#
# Checks Docker and the NVIDIA runtime, the two interpreters (env.sh), Harbor,
# the agent CLI and its credentials, image freshness, every named task's
# `taskctl check`, the task's model checkpoint, and disk. Prints one line per
# check and exits non-zero if any required check fails.
set -uo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
fail=0
ok()   { printf '  ok    %s\n' "$*"; }
bad()  { printf '  FAIL  %s\n' "$*"; fail=1; }
warn() { printf '  warn  %s\n' "$*"; }

echo "== host"
command -v docker >/dev/null && docker info >/dev/null 2>&1 && ok "docker daemon" || bad "docker daemon unreachable"
docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia && ok "nvidia container runtime" \
    || warn "no nvidia runtime registered (CDI/--gpus may still work)"
if command -v nvidia-smi >/dev/null; then
    ok "GPUs: $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) visible"
    busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u | wc -l)
    [[ $busy -gt 0 ]] && warn "$busy GPU(s) have compute processes right now" || ok "no compute processes on any GPU"
else
    bad "nvidia-smi missing"
fi

echo "== interpreters"
if . "$repo/compression_bench/scripts/env.sh" 2>/dev/null; then
    ok "COMPRESSION_BENCH_PY=$COMPRESSION_BENCH_PY"
    ok "COMPRESSION_BENCH_HARBOR_PY=$COMPRESSION_BENCH_HARBOR_PY"
    if harbor_version=$("$COMPRESSION_BENCH_HARBOR_PY" -c "import harbor, importlib.metadata as m; print(m.version('harbor'))" 2>/dev/null); then
        ok "harbor $harbor_version importable"
    else
        bad "harbor not importable by COMPRESSION_BENCH_HARBOR_PY"
    fi
    "$COMPRESSION_BENCH_PY" -c "import yaml" 2>/dev/null && ok "pyyaml for the harness" || bad "COMPRESSION_BENCH_PY lacks pyyaml"
else
    bad "compression_bench/scripts/env.sh refused (bind COMPRESSION_BENCH_PY / COMPRESSION_BENCH_HARBOR_PY in env.local.sh)"
    COMPRESSION_BENCH_PY=python3
fi

echo "== agent CLI"
claude_bin=$(readlink -f "$(command -v claude 2>/dev/null || echo "$HOME/.local/bin/claude")" 2>/dev/null)
[[ -x $claude_bin ]] && ok "claude CLI $("$claude_bin" --version 2>/dev/null | head -1)" || warn "no claude CLI on this host"
[[ -s $HOME/.claude/.credentials.json || -n ${ANTHROPIC_API_KEY:-} ]] && ok "claude credentials present" \
    || warn "no ~/.claude/.credentials.json and no ANTHROPIC_API_KEY"
command -v codex >/dev/null && ok "codex CLI present" || warn "no codex CLI (only needed with AGENT_CLI=codex)"

echo "== images"
"$COMPRESSION_BENCH_PY" "$repo/compression_bench/scripts/agent/image_freshness.py" check vllm-runtime agent-base platform-gguf \
    | sed 's/^/        /'
[[ ${PIPESTATUS[0]} -eq 0 ]] && ok "vllm-runtime, agent-base and platform-gguf present and fresh" \
    || bad "build images: bash compression_bench/scripts/deploy/01-build-images.sh"

echo "== tasks"
tasks=("$@")
for task in "${tasks[@]}"; do
    if "$COMPRESSION_BENCH_PY" "$repo/compression_tasks/_manage/taskctl.py" check "$task" >/dev/null; then
        ok "taskctl check $task"
    else
        bad "taskctl check $task (run it for details)"
    fi
    if [[ -n ${MODEL_DIR:-} ]]; then
        "$COMPRESSION_BENCH_PY" -c "import sys; sys.path.insert(0, sys.argv[1]); from model_for_task import is_checkpoint; from pathlib import Path; sys.exit(not is_checkpoint(Path(sys.argv[2])))" \
            "$repo/compression_bench/scripts" "$MODEL_DIR" \
            && ok "model for $task: $MODEL_DIR (MODEL_DIR)" || bad "MODEL_DIR=$MODEL_DIR holds no config.json + weight files"
    else
        model=$("$COMPRESSION_BENCH_PY" "$repo/compression_bench/scripts/model_for_task.py" "$repo/compression_tasks/$task" 2>/dev/null) \
            && ok "model for $task: $model" || bad "no local checkpoint for $task (set MODEL_DIR)"
    fi
done
[[ ${#tasks[@]} -eq 0 ]] && warn "no task named; pass task ids to check their checkpoints"

echo "== disk"
# Run directories hold the agent's candidates (a 27B model is ~30-55 GB per copy);
# Docker holds only the thin per-run image layers above the shared bases.
docker_root=$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)
for spec in "$repo:100:run directories and candidate weights" "$docker_root:20:per-run image layers"; do
    IFS=: read -r path need why <<<"$spec"
    free=$(df -BG --output=avail "$path" 2>/dev/null | tail -1 | tr -dc 0-9)
    [[ -z $free ]] && continue
    if [[ $free -lt $need ]]; then bad "$path: ${free}G free (< ${need}G for $why)"
    else ok "$path: ${free}G free ($why)"; fi
done

exit $fail
