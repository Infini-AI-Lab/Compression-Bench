# The GGUF platform: the agent base plus the pinned vllm-gguf-plugin.
# Built by compression_bench/scripts/deploy/01-build-images.sh, which passes BASE_IMAGE
# (images.json "agent-base") and labels the result with its inputs digest.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
# Pinned plugin revision.
ARG GGUF_PLUGIN_REV=e2b8ad532b8b5ea175100202c30430c1d2b5e6a8
ENV MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=9.0
RUN git clone https://github.com/vllm-project/vllm-gguf-plugin /tmp/vllm-gguf-plugin \
    && cd /tmp/vllm-gguf-plugin && git checkout "$GGUF_PLUGIN_REV" \
    && python3 -m pip install --no-cache-dir --no-build-isolation --no-deps . \
    && rm -rf /tmp/vllm-gguf-plugin
# The plugin's two Qwen3.5 config imports, moved to Transformers' classes for vLLM 0.32.0.
COPY compat_vllm_0_32.py /tmp/compat_vllm_0_32.py
RUN python3 /tmp/compat_vllm_0_32.py && rm /tmp/compat_vllm_0_32.py
RUN python3 -m pip install --no-cache-dir 'gguf==0.19.0' 'huggingface_hub==2.2.0' \
    && python3 -c "import vllm_gguf_plugin; from importlib.metadata import version; assert version('vllm') == '0.32.0'"
