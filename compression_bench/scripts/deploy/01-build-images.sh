#!/usr/bin/env bash
# Build the images every task renders FROM (compression_bench/environments/images.json),
# each labelled with the digest of its inputs so a run can refuse a stale one.
#
#   bash compression_bench/scripts/deploy/01-build-images.sh            # agent-base, platform-gguf, pull vllm-runtime
#   bash compression_bench/scripts/deploy/01-build-images.sh agent-base # just one
#
# Needs network (pip, git, the vLLM base, the papers from their publishers). Docker reuses unchanged layers, so a
# rebuild after a documentation edit re-runs only the layers after that COPY.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
freshness="$repo/compression_bench/scripts/agent/image_freshness.py"
manifest="$repo/compression_bench/environments/images.json"
py=${COMPRESSION_BENCH_PY:-python3}

field() { "$py" -c "import json,sys; print(json.load(open('$manifest'))[sys.argv[1]].get(sys.argv[2]) or '')" "$1" "$2"; }

build() {
    local key=$1 tag context digest args=()
    tag=$(field "$key" tag)
    if [[ $(field "$key" pulled) == True ]]; then
        echo "== pull $tag"
        docker image inspect "$tag" >/dev/null 2>&1 || docker pull "$tag"
        return
    fi
    context="$repo/$(field "$key" context)"
    digest=$("$py" "$freshness" digest "$key")
    while IFS='=' read -r arg base; do
        [[ -n $arg ]] && args+=(--build-arg "$arg=$(field "$base" tag)")
    done < <("$py" -c "import json,sys; [print(f'{k}={v}') for k,v in json.load(open('$manifest'))[sys.argv[1]].get('build_args',{}).items()]" "$key")
    echo "== build $tag (inputs ${digest:0:12})"
    docker build "${args[@]}" --label "compression_bench.image_inputs_sha256=$digest" -t "$tag" "$context"
}

keys=("$@")
[[ ${#keys[@]} -gt 0 ]] || keys=(vllm-runtime agent-base platform-gguf)
# The library's papers and the training datasets are not tracked: restore (or
# verify) them before agent-base hashes its context, and refuse a paper holding
# an unlisted copy of the evaluation benchmark.
# (prepare_datasets.py and benchmark_guard.py need pyarrow and huggingface_hub.)
if [[ " ${keys[*]} " == *" agent-base "* ]]; then
    "$py" "$repo/compression_bench/scripts/deploy/prepare_references.py"
    "${DEPLOY_PY:-python3}" "$repo/compression_bench/scripts/deploy/prepare_datasets.py"
    "${DEPLOY_PY:-python3}" "$repo/compression_bench/scripts/deploy/benchmark_guard.py" check
fi
for key in "${keys[@]}"; do build "$key"; done
"$py" "$freshness" check "${keys[@]}"
