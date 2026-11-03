# The paper collection the agent searches (/opt/compression-papers): the papers of
# the restored library (library/, the Awesome-LLM-Compression catalogue,
# references/papers.json and the arXiv supplement pinned by papers/supplement.json)
# in one flat directory, with every paper that holds an evaluation benchmark
# question withheld (library-exclude.txt, written by
# scripts/deploy/benchmark_guard.py), a catalogue, a full-text index and the
# compression-papers skill (papers/skill/). Built in its own stage, so no layer of
# the final image carries a withheld paper or any author source code
# (.dockerignore also keeps library/**/upstream out of the build context).
# The sources are bind-mounted (read-only), not copied, so the stage keeps one copy.
FROM vllm/vllm-openai:v0.32.0 AS papers
RUN --mount=type=bind,source=library,target=/tmp/library \
    --mount=type=bind,source=library-exclude.txt,target=/tmp/library-exclude.txt \
    --mount=type=bind,source=papers,target=/tmp/papers \
    python3 /tmp/papers/build_collection.py --library /tmp/library --exclude /tmp/library-exclude.txt \
        --skill /tmp/papers/skill --out /opt/compression-papers \
    && chmod -R a=rX /opt/compression-papers

FROM vllm/vllm-openai:v0.32.0

# The runtime is vLLM 0.32.0 as released, with the humming-kernels it pins
# (0.1.17, whose API its Humming integration calls); nothing in it is patched.
RUN python3 -c "from importlib.metadata import version; assert version('vllm') == '0.32.0'; assert version('humming-kernels') == '0.1.17'"

ENTRYPOINT []

# git: the platform image clones the pinned vLLM GGUF plugin revision.
RUN if ! command -v git >/dev/null; then \
      apt-get update && apt-get install -y --no-install-recommends git \
      && rm -rf /var/lib/apt/lists/*; \
    fi

# Training data (scripts/deploy/prepare_datasets.py): pinned public datasets,
# none an evaluation input, rows holding a benchmark question removed.
COPY datasets/ /opt/datasets/
RUN chmod -R a=rX /opt/datasets

# Runs one command in the foreground until it ends or times out.
COPY run_managed_job.py /app/run_managed_job.py
RUN chmod 644 /app/run_managed_job.py

# The base interpreter reads Parquet; CUDA extensions build for the evaluation
# GPU only (H200, sm_90).
ENV MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=9.0
RUN python3 -m pip install --no-cache-dir --no-deps pyarrow==25.0.1 \
    && python3 -c "import pyarrow.parquet; from importlib.metadata import version; assert version('vllm') == '0.32.0'"

COPY --from=papers /opt/compression-papers/ /opt/compression-papers/
RUN chmod a=rX /opt/compression-papers \
    && python3 /opt/compression-papers/skill/scripts/search.py --categories | tail -1 | grep -q " papers$" \
    && test ! -e /opt/compression-library

# The base runtime's only pip-check findings are the release's own: torch's NCCL
# pin and OpenTelemetry's SDK pin (both present in vllm/vllm-openai:v0.32.0).
RUN python3 -m pip check 2>&1 | grep -v "nvidia-nccl-cu13\|opentelemetry-sdk" | (! grep -q .)
