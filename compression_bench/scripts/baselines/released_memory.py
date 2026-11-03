"""Evaluator-owned resident-model-storage adapter for the released baselines' native runtime.

The task's census (`model_memory.py`, resident-model-storage-v1) audits vLLM engines only
and refuses "custom/CUDA-native backends" without an evaluator-owned adapter. This is
the adapter, installed beside it in the baseline's copy of the evaluator (see
`released_baseline.py`, which records the patch). It reads the runtime's own objects,
never a number the submission reports:

* llama.cpp (`llamacpp-model-buffers-v1`): the weight buffers llama.cpp's loader allocates,
  per device, from its load report (captured by a log hook installed before the submission
  is imported), and `llama_model_size()`, the bytes of every loaded weight tensor. Weight
  bytes in a host buffer are reported as `host_model_bytes` (CPU-resident weights).

The worker reports its runtime's format in `foreign_quant_methods`: a released
quantization is never the submission's own compression, and the patched `check_snapshot`
records that instead of stopping the measurement.
"""
import ctypes
import re
import sys

PROTOCOL = "resident-model-storage-v1"
_LOG = []
_CALLBACK = []
BUFFER = re.compile(r"(\S+) model buffer size =\s*([\d.]+) MiB")


def install():
    """Before the submission is imported: keep llama.cpp's log lines (they still reach stderr)."""
    try:
        import llama_cpp
    except Exception:
        return

    last = [1]

    @llama_cpp.llama_log_callback
    def callback(level, text, user_data):
        # ggml's levels: 1 DEBUG, 2 INFO, 3 WARN, 4 ERROR, 5 continuation. INFO and above are
        # kept; the DEBUG lines, a graph dump per decode step, would slow every generated token.
        level = last[0] if level == 5 else level
        last[0] = level
        if level not in (2, 3, 4):
            return
        line = text.decode("utf-8", errors="replace")
        if "model buffer size" in line:
            _LOG.append(line)
        sys.stderr.write(line)

    _CALLBACK.append(callback)          # ctypes callbacks must outlive their registration
    llama_cpp.llama_log_set(callback, ctypes.c_void_p(0))


def _candidates(engine):
    return [engine, *vars(engine).values()]


def _llamacpp(engine):
    try:
        from llama_cpp import _internals
        import llama_cpp
    except Exception:
        return None
    models = [o for o in _candidates(engine) if isinstance(o, _internals.LlamaModel)]
    if not models:
        return None
    if len(models) != 1:
        raise ValueError("model-memory: expose one llama.cpp model on the engine")
    model = models[0]
    buffers = [(name, float(mib)) for name, mib in BUFFER.findall("".join(_LOG))]
    if not buffers:
        raise ValueError("model-memory: no llama.cpp loader report was captured")
    device = [(n, mib) for n, mib in buffers if n.startswith("CUDA")]
    host = [(n, mib) for n, mib in buffers if not n.startswith("CUDA")]
    storages = [{"bytes": round(mib * 2**20), "device": "cuda:" + n[4:], "dtype": "ggml buffer",
                 "first_path": f"llama.cpp {n} model buffer"} for n, mib in device]
    total = sum(s["bytes"] for s in storages)
    if total <= 0:
        raise ValueError("no resident model storage found; unsupported or offloaded model")
    return {"bytes": total, "storage_count": len(storages), "storages": storages, "worker_rank": 0,
            "adapter": "llamacpp-model-buffers-v1",
            "foreign_quant_methods": ["llama.cpp GGUF: " + _desc(llama_cpp, model)],
            "tensor_bytes": int(llama_cpp.llama_model_size(model.model)),
            "n_params": int(llama_cpp.llama_model_n_params(model.model)),
            "host_model_bytes": round(sum(mib for _, mib in host) * 2**20),
            "loader_buffers_mib": dict(buffers),
            "precision": "llama.cpp reports each buffer in MiB to 2 decimals (within 5.2 kB)"}


def _desc(llama_cpp, model):
    buf = ctypes.create_string_buffer(256)
    llama_cpp.llama_model_desc(model.model, buf, ctypes.sizeof(buf))
    return buf.value.decode("utf-8", errors="replace")


def collect(engine):
    """The census snapshot of a released runtime's engine, or None if it is not one."""
    worker = _llamacpp(engine)
    if worker is None:
        return None
    return {"protocol": PROTOCOL, "workers": [worker]}
