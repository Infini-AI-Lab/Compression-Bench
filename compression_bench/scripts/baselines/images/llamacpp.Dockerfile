# Released-baseline runtime for GGUF: the Verifier's base image plus llama.cpp (llama-cpp-python 0.3.36,
# its vendored llama.cpp built for the H200's sm_90 with the CUDA backend). Nothing of the base changes:
# the build refuses a changed vLLM, torch or transformers. Built by build_released_images.sh.
FROM compression_bench/platform-gguf:v1

COPY llama_cpp_python-0.3.36.tar.gz /opt/released/llama_cpp_python-0.3.36.tar.gz
RUN echo "832db0699007f1be95a7e41ef12e88926b02ba836461e36a36372db2760c1a2e  /opt/released/llama_cpp_python-0.3.36.tar.gz" | sha256sum -c - \
 && pip freeze > /opt/released/base-freeze.txt \
 && pip install --no-cache-dir "scikit-build-core[pyproject]>=0.9.2" cmake \
 && CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=90 -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF" \
    pip install --no-cache-dir --no-build-isolation --no-deps /opt/released/llama_cpp_python-0.3.36.tar.gz \
 && pip install --no-cache-dir "diskcache>=5.6.1" \
 && pip freeze > /opt/released/freeze.txt \
 && python3 -c "from importlib.metadata import version as v; assert v('vllm') == '0.32.0' and v('torch').startswith('2.13.0') and v('transformers') == '5.19.0' and v('llama_cpp_python') == '0.3.36', 'base changed'"
