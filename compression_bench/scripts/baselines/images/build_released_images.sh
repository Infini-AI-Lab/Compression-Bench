#!/bin/bash
# Build the released-baseline runtime: compression_bench/released-llamacpp:0.3.36.
# The context (the pinned source archive, checked by sha256 in the Dockerfile) lives in the
# gitignored evolve_space; the image is labelled with its base image's ID.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
CTX="$REPO/compression_bench/evolve_space/baselines/released/image-context"
BASE=$(docker image inspect compression_bench/platform-gguf:v1 --format '{{.Id}}')
fetch() {   # url file
    [ -s "$CTX/$2" ] || curl -sSfL -m 1800 -o "$CTX/$2.part" "$1" && { [ -s "$CTX/$2" ] || mv "$CTX/$2.part" "$CTX/$2"; }
}
build() {   # name tag dockerfile
    mkdir -p "$CTX/$1"
    cp "$HERE/$3" "$CTX/$1/Dockerfile"
    docker build --label "compression_bench.base_image=$BASE" --label "compression_bench.dockerfile_sha256=$(sha256sum "$HERE/$3" | cut -d' ' -f1)" \
        -t "$2" "$CTX/$1"
}
mkdir -p "$CTX"
fetch https://files.pythonhosted.org/packages/ec/e9/e7de2b0463ea3ffbf0ede6cb21b58c1258a8f6521aae45ca773a59fe7cf3/llama_cpp_python-0.3.36.tar.gz llama_cpp_python-0.3.36.tar.gz
mkdir -p "$CTX/llamacpp" && ln -f "$CTX/llama_cpp_python-0.3.36.tar.gz" "$CTX/llamacpp/"
build llamacpp compression_bench/released-llamacpp:0.3.36 llamacpp.Dockerfile
