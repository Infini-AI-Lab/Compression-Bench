#!/usr/bin/env python3
"""Released-baseline inference: a published GGUF served unchanged by llama.cpp on one GPU.

The task's submission interface (INFERENCE_API_VERSION 1) over llama.cpp's CUDA backend
(llama-cpp-python 0.3.36). Every weight tensor, the token embedding included, is placed
in the GPU's buffer (llama.cpp keeps the embedding on the host unless told otherwise):
no CPU offload. The weights stay in their packed GGUF types and are multiplied by
llama.cpp's own kernels for them. MTP layers are not loaded and nothing is speculated:
one forward pass per generated token.

    generate        greedy (argmax of the logits), stopping at the end-of-sequence IDs of
                    the checkpoint's config.json and generation_config.json, as the task's
                    starter does; the stop token is part of the returned IDs, as in vLLM
    token_logprobs  natural-log softmax of llama.cpp's float32 logits, every position
    stream_generate one new token per event
"""
import ctypes
import json
import os
from pathlib import Path

INFERENCE_API_VERSION = 1

#: One forward pass holds a whole 2048-token PPL block, with its logits at every position.
N_BATCH = 2048
N_UBATCH = 512


def load(model_dir, *, max_model_len=2049, max_num_seqs=8, gpu_memory_utilization=0.80):
    return Engine(model_dir, max_model_len=max_model_len)


def end_of_sequence_ids(model_dir):
    """The end-of-sequence IDs of the checkpoint's config.json and generation_config.json."""
    found = []
    for name in ('config.json', 'generation_config.json'):
        path = Path(model_dir) / name
        if not path.is_file():
            continue
        cfg = json.loads(path.read_text())
        for value in (cfg.get('eos_token_id'), (cfg.get('text_config') or {}).get('eos_token_id')):
            for token in (value if isinstance(value, list) else [value]):
                if isinstance(token, int) and token not in found:
                    found.append(token)
    return found


def gguf_entry(model_dir):
    """The file llama.cpp opens: the single GGUF, or the first shard of a split one."""
    files = sorted(Path(model_dir).glob('*.gguf'))
    first = [f for f in files if '-00001-of-' in f.name]
    if first:
        return first[0]
    if len(files) != 1:
        raise ValueError(f'expected one GGUF or a split set in {model_dir}: {[f.name for f in files]}')
    return files[0]


class _Override(ctypes.Structure):
    _fields_ = [('pattern', ctypes.c_char_p), ('buft', ctypes.c_void_p)]


def cuda_buffer_type(llama_cpp):
    """llama.cpp's buffer type of the (only visible) GPU, from ggml's device registry."""
    lib = Path(llama_cpp.__file__).parent / 'lib'
    ggml = ctypes.CDLL(str(lib / 'libggml.so'))
    ggml.ggml_backend_dev_by_name.restype = ctypes.c_void_p
    ggml.ggml_backend_dev_by_name.argtypes = [ctypes.c_char_p]
    ggml.ggml_backend_dev_buffer_type.restype = ctypes.c_void_p
    ggml.ggml_backend_dev_buffer_type.argtypes = [ctypes.c_void_p]
    dev = ggml.ggml_backend_dev_by_name(b'CUDA0')
    if not dev:
        raise RuntimeError('llama.cpp has no CUDA0 device')
    return ggml.ggml_backend_dev_buffer_type(dev)


class Engine:
    def __init__(self, model_dir, *, max_model_len):
        import llama_cpp
        from llama_cpp import _internals
        import numpy as np
        import torch
        self.np, self.torch, self.llama_cpp = np, torch, llama_cpp
        self.stop_token_ids = end_of_sequence_ids(model_dir)
        llama_cpp.llama_backend_init()
        params = llama_cpp.llama_model_default_params()
        params.n_gpu_layers = 0x7FFFFFFF
        params.split_mode = llama_cpp.LLAMA_SPLIT_MODE_NONE
        params.main_gpu = 0
        params.load_mtp = False
        params.use_extra_bufts = False
        # Every tensor, the token embedding included, in the GPU's buffer.
        self._overrides = (_Override * 2)()
        self._overrides[0].pattern = b'.'
        self._overrides[0].buft = cuda_buffer_type(llama_cpp)
        params.tensor_buft_overrides = ctypes.cast(self._overrides, ctypes.c_void_p)
        self.model = _internals.LlamaModel(path_model=str(gguf_entry(model_dir)), params=params, verbose=True)
        cparams = llama_cpp.llama_context_default_params()
        cparams.n_ctx = max(max_model_len, N_BATCH + 1)
        cparams.n_batch = N_BATCH
        cparams.n_ubatch = N_UBATCH
        cparams.n_seq_max = 1
        cparams.n_threads = cparams.n_threads_batch = min(8, os.cpu_count() or 1)
        cparams.offload_kqv = True
        cparams.op_offload = True
        cparams.no_perf = True
        self.context = _internals.LlamaContext(model=self.model, params=cparams, verbose=True)
        self.batch = _internals.LlamaBatch(n_tokens=N_BATCH, embd=0, n_seq_max=1, verbose=True)
        self.n_ctx = llama_cpp.llama_n_ctx(self.context.ctx)
        self.n_vocab = llama_cpp.llama_vocab_n_tokens(self.model.vocab)
        self.device = torch.device('cuda', 0)

    # -- llama.cpp calls --------------------------------------------------------------
    def _reset(self):
        self.llama_cpp.llama_memory_clear(self.context.memory, True)

    def _decode(self, tokens, start, *, all_logits):
        b = self.batch.batch
        b.n_tokens = len(tokens)
        for i, token in enumerate(tokens):
            b.token[i] = token
            b.pos[i] = start + i
            b.n_seq_id[i] = 1
            b.seq_id[i][0] = 0
            b.logits[i] = all_logits
        b.logits[len(tokens) - 1] = True
        code = self.llama_cpp.llama_decode(self.context.ctx, b)
        if code != 0:
            raise RuntimeError(f'llama_decode returned {code}')

    def _last_logits(self):
        ptr = self.llama_cpp.llama_get_logits_ith(self.context.ctx, -1)
        return self.np.ctypeslib.as_array(ptr, shape=(self.n_vocab,))

    def _prefill(self, token_ids):
        if not token_ids or len(token_ids) > N_BATCH:
            raise ValueError(f'prompt of {len(token_ids)} tokens; 1..{N_BATCH} supported')
        self._reset()
        self._decode(token_ids, 0, all_logits=False)
        return self._last_logits()

    def _next(self, logits):
        return int(self.np.argmax(logits))

    # -- the task's interface ---------------------------------------------------------
    def generate(self, token_ids, *, max_new_tokens, ignore_eos=False):
        """Batch of prompts -> batch of NEW token IDs (do not include prompts)."""
        out = []
        for ids in token_ids:
            if len(ids) + max_new_tokens > self.n_ctx:
                raise ValueError('prompt and generation exceed the context')
            logits, row = self._prefill(list(ids)), []
            while True:
                token = self._next(logits)
                row.append(token)
                if len(row) == max_new_tokens or (not ignore_eos and token in self.stop_token_ids):
                    break
                self._decode([token], len(ids) + len(row) - 1, all_logits=False)
                logits = self._last_logits()
            out.append(row)
        return out

    def token_logprobs(self, token_ids):
        """Return aligned natural-log probabilities from this submitted model."""
        torch, result = self.torch, []
        for ids in token_ids:
            ids = list(ids)
            if not 1 < len(ids) <= N_BATCH:
                raise ValueError(f'sequence of {len(ids)} tokens; 2..{N_BATCH} supported')
            self._reset()
            self._decode(ids, 0, all_logits=True)
            ptr = self.llama_cpp.llama_get_logits(self.context.ctx)
            logits = self.np.ctypeslib.as_array(ptr, shape=(len(ids), self.n_vocab))
            with torch.inference_mode():
                x = torch.from_numpy(logits[:-1]).to(self.device, non_blocking=False)
                target = torch.tensor(ids[1:], device=self.device, dtype=torch.long)
                values = torch.log_softmax(x, dim=-1).gather(1, target[:, None])[:, 0]
                values = values.double().cpu().tolist()
                del x
            result.append([None] + values)
        return result

    def stream_generate(self, token_ids, *, max_new_tokens, ignore_eos=False):
        """Single prompt -> incremental NEW token IDs, never timestamps or scores."""
        ids = list(token_ids)
        if len(ids) + max_new_tokens > self.n_ctx:
            raise ValueError('prompt and generation exceed the context')
        logits = self._prefill(ids)
        for step in range(max_new_tokens):
            token = self._next(logits)
            yield [token]
            if step + 1 == max_new_tokens or (not ignore_eos and token in self.stop_token_ids):
                return
            self._decode([token], len(ids) + step, all_logits=False)
            logits = self._last_logits()
