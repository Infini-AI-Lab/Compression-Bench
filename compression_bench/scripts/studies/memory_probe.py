#!/usr/bin/env python3
"""Peak-memory probe of one released GGUF baseline package: diagnostic instrumentation, never a submission or a
storage gate. Also provides NvmlSampler, the 10 ms device-memory sampler of controlled_verify.py.

    memory_probe.py --submission DIR --evaluator DIR --out DIR

Run by memory_study.py in the released llama.cpp runtime under Nsight Systems. Loads the package through its
inference.py with a 4,096-token context and one sequence, then generates 128 tokens (EOS ignored) from the task's
first measured 2,048-token decode prompt. NVTX ranges delimit cold load, prefill through the first token, and the
remaining 127 decode token intervals; each phase records the resident model storage (released_memory.py) and the
torch peak counters, and NVML samples device memory every 10 ms. Writes measurement.json and
device_memory_samples.jsonl to --out.
"""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time


class NvmlSampler:
    """Samples the device-used memory of the card named by $STUDY_GPU_UUID every 10 ms, tagged with the current
    phase; close() writes the samples as JSON lines."""

    class Info(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

    def __init__(self, out):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        assert self.lib.nvmlInit_v2() == 0
        self.handle = ctypes.c_void_p()
        assert self.lib.nvmlDeviceGetHandleByUUID(os.environ["STUDY_GPU_UUID"].encode(), ctypes.byref(self.handle)) == 0
        self.phase = "startup"
        self.rows = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        self.out = out

    def sample(self):
        while not self.stop.is_set():
            info = self.Info()
            if self.lib.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(info)) == 0:
                self.rows.append({"monotonic_ns": time.monotonic_ns(), "phase": self.phase, "device_used_bytes": info.used})
            self.stop.wait(0.01)

    def close(self):
        self.stop.set()
        self.thread.join()
        self.out.write_text("".join(json.dumps(r) + "\n" for r in self.rows))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--submission", required=True)
    p.add_argument("--evaluator", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [a.evaluator, a.submission, str(Path(__file__).parent)]
    import torch
    import released_memory
    released_memory.install()
    spec = importlib.util.spec_from_file_location("study_inference", Path(a.submission) / "inference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    prompts = json.loads((Path(a.evaluator) / "evaluator-assets/decode_inputs.json").read_text())
    ids = prompts["measured"][0]
    assert len(ids) == 2048
    sampler = NvmlSampler(out / "device_memory_samples.jsonl")
    engine = None
    phases = []
    result = {"status": "running", "context_token_capacity": 4096, "max_sequences": 1,
              "input_tokens": 2048, "output_tokens": 128,
              "prompt_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
              "nvml_sampling_interval_seconds": 0.01}

    def begin(name):
        sampler.phase = name
        torch.cuda.nvtx.range_push("study:" + name)
        return time.monotonic_ns()

    def end(name, start):
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        phases.append({"phase": name, "start_monotonic_ns": start, "end_monotonic_ns": time.monotonic_ns(),
                       "workers": [{"resident_model": released_memory._llamacpp(engine),
                                    "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                    "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved()}]})

    try:
        started = begin("loading")
        engine = mod.load(Path(a.submission) / "model", max_model_len=4096, max_num_seqs=1, gpu_memory_utilization=0.9)
        end("loading", started)
        assert engine.n_ctx == 4096, engine.n_ctx
        torch.cuda.reset_peak_memory_stats()
        started = begin("prefill")
        tokens = iter(engine.stream_generate(ids, max_new_tokens=128, ignore_eos=True))
        first = next(tokens)
        assert len(first) == 1
        end("prefill", started)
        torch.cuda.reset_peak_memory_stats()
        started = begin("decode")
        count = 1
        for chunk in tokens:
            assert len(chunk) == 1
            count += 1
        assert count == 128, count
        end("decode", started)
        result.update(status="measured", generated_tokens=count)
    except Exception as e:
        result.update(status="execution_failed", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        sampler.phase = "finished"
        sampler.close()
        result["phases"] = phases
        for phase in phases:
            values = [r["device_used_bytes"] for r in sampler.rows if r["phase"] == phase["phase"]]
            phase["sampled_peak_device_bytes"] = max(values) if values else None
        (out / "measurement.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
