"""Evaluator-owned resident model storage census, not total GPU allocation.

This is an audited-v1 protocol, not a security boundary against submitted Python.
Unsupported storage/backends fail closed. CPU unit tests do not validate CUDA RPC.
"""
import json
import math
from pathlib import Path
from contextlib import contextmanager
from contextvars import ContextVar

PROTOCOL = "resident-model-storage-v1"
_PROFILE_LIMIT = ContextVar("evaluator_model_memory_profile", default=None)


@contextmanager
def using_profile_limit(limit):
    # Selected by the evaluator-owned public debug profile, never submission code.
    if type(limit) not in (int, float) or not math.isfinite(limit) or limit <= 0:
        raise ValueError("invalid profile model-memory limit")
    token = _PROFILE_LIMIT.set(limit)
    try:
        yield
    finally:
        _PROFILE_LIMIT.reset(token)



def read_policy(directory):
    path = Path(directory) / "evaluator-assets/model_memory_policy.json"
    if not path.is_file():
        raise ValueError("missing task-owned model-memory policy")
    policy = json.loads(path.read_text())
    limit = policy.get("limit_gib")
    if (policy.get("protocol") != PROTOCOL or type(limit) not in (int, float)
            or not math.isfinite(limit) or limit <= 0):
        raise ValueError("invalid evaluator-owned model memory policy")
    override = _PROFILE_LIMIT.get()
    if override is not None:
        policy = dict(policy, limit_gib=override)
    return policy


def inventory_model(model, *, excluded=(), device_type="cuda"):
    """Count unique full storage bytes, including unregistered tensor attributes.

    Exclusions are backend-owned KV/state cache storages, never candidate numbers.
    Sharing/views count once; a tiny view still retains its entire backing storage.
    Non-tensor native allocations and opaque closures require an audited adapter.
    """
    import torch
    import types
    seen, storages, excluded_keys = set(), {}, set()
    runtime_fields_skipped = []
    empty_gguf_placeholders = []
    # Pinned vLLM backend objects: Graph outputs and compiler sample inputs
    # are runtime data. Do not apply these exclusions to arbitrary field names.
    runtime_fields = {
        ("vllm.compilation.cuda_graph", "CUDAGraphEntry"): {"output"},
        ("vllm.compilation.caching", "VllmSerializableFunction"): {"example_inputs"},
    }

    def empty_gguf_type_placeholder(obj):
        # Only the pinned plugin's exact empty type-metadata placeholder is special.
        # Actual packed weights and arbitrary UninitializedParameter subclasses fail closed.
        cls = type(obj)
        if (cls.__module__, cls.__name__) != (
                "vllm_gguf_plugin.quantization.params", "GGUFUninitializedWeightTypeParameter"):
            return False
        from vllm_gguf_plugin.quantization.params import GGUFUninitializedWeightTypeParameter
        if cls is not GGUFUninitializedWeightTypeParameter:
            raise ValueError("untrusted GGUF placeholder class")
        if not isinstance(obj, torch.nn.parameter.UninitializedParameter):
            raise ValueError("unexpected materialized GGUF placeholder type")
        # .data is the underlying Tensor; numel alone would miss a zero-sized view
        # retaining nonempty storage. Both must be empty.
        if obj.data.numel() != 0 or obj.data.untyped_storage().nbytes() != 0:
            raise ValueError("GGUF placeholder unexpectedly retains storage")
        return True

    def key(t):
        s = t.untyped_storage()
        return (str(t.device), s.data_ptr(), s.nbytes())

    def tensors(obj, path):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, torch.Tensor):
            if empty_gguf_type_placeholder(obj):
                empty_gguf_placeholders.append(path)
                # Metadata may carry auxiliary tensors; inspect them normally.
                yield from tensors(vars(obj), path + ".attributes")
                return
            if obj.device.type == device_type:
                if obj.layout != torch.strided:
                    raise ValueError("unsupported model tensor layout: " + path)
                yield path, obj
        elif isinstance(obj, dict):
            for name, child in obj.items():
                yield from tensors(child, path + "." + str(name))
        elif isinstance(obj, (tuple, list)):
            for i, child in enumerate(obj):
                yield from tensors(child, path + "." + str(i))
        elif isinstance(obj, (set, frozenset)):
            for i, child in enumerate(obj):
                yield from tensors(child, path + ".set." + str(i))
        elif isinstance(obj, (types.FunctionType, types.MethodType)):
            for i, cell in enumerate(getattr(obj, "__closure__", None) or ()):
                try:
                    child = cell.cell_contents
                except ValueError:
                    continue
                yield from tensors(child, path + ".closure." + str(i))
            yield from tensors(getattr(obj, "__defaults__", None), path + ".defaults")
            yield from tensors(getattr(obj, "__kwdefaults__", None), path + ".kwdefaults")
        elif not isinstance(obj, (type, types.ModuleType)) and hasattr(obj, "__dict__"):
            attributes = vars(obj)
            skip = runtime_fields.get((type(obj).__module__, type(obj).__name__), set())
            if skip:
                runtime_fields_skipped.extend(path + "." + name for name in skip if name in attributes)
            for name, child in attributes.items():
                if name not in skip:
                    yield from tensors(child, path + "." + name)

    for _, tensor in tensors(excluded, "backend_cache"):
        excluded_keys.add(key(tensor))
    seen.clear()
    # A cache exclusion must never hide registered model parameters.
    for parameter in model.parameters():
        if empty_gguf_type_placeholder(parameter):
            continue
        if parameter.device.type == device_type and key(parameter) in excluded_keys:
            raise ValueError("backend cache aliases a model parameter")
    for path, tensor in tensors(model, "model"):
        storage_key = key(tensor)
        if storage_key in excluded_keys:
            continue
        if storage_key not in storages:
            storages[storage_key] = dict(bytes=storage_key[2], device=str(tensor.device),
                                        dtype=str(tensor.dtype), first_path=path)
    rows = list(storages.values())
    total = sum(row["bytes"] for row in rows)
    if total <= 0:
        raise ValueError("no resident model storage found; unsupported or offloaded model")
    return dict(bytes=total, storage_count=len(rows), storages=rows,
                excluded_cache_storage_count=len(excluded_keys),
                excluded_runtime_fields=runtime_fields_skipped,
                empty_gguf_type_placeholders=empty_gguf_placeholders)


#: Modules whose quantization methods are not a submission's own (own_compression.py; kept
#: here because this module runs in vLLM workers and imports no evaluator sibling).
FOREIGN_METHOD_MODULES = ("vllm.model_executor.layers.quantization.", "vllm_gguf_plugin", "humming",
                          "compressed_tensors", "gguf", "bitsandbytes", "auto_gptq", "gptqmodel",
                          "awq", "hqq", "optimum", "torchao", "llmcompressor", "quanto", "aqlm",
                          "exllama", "marlin", "flute", "vptq", "auto_round", "modelopt")
OWN_METHOD_BASES = ("vllm.model_executor.layers.quantization.base_config",
                    "vllm.model_executor.layers.quantization.kv_cache")


def foreign_quant_methods(model):
    """Every built-in or packaged quantization method class serving a layer of the model,
    directly or as a base class of the submission's own."""
    found = set()
    for module in model.modules():
        method = getattr(module, "quant_method", None)
        if method is None:
            continue
        for cls in type(method).__mro__:
            name = cls.__module__ or ""
            if name in OWN_METHOD_BASES:
                continue
            for prefix in FOREIGN_METHOD_MODULES:
                if name.startswith(prefix) if prefix.endswith(".") else (name == prefix or name.startswith(prefix + ".")):
                    found.add(f"{name}.{cls.__qualname__}")
    return sorted(found)


def worker_inventory(worker):
    """Runs on each vLLM GPU worker, including multiprocess engines."""
    import torch
    runner = getattr(worker, "model_runner", None)
    model = getattr(runner, "model", None)
    if model is None:
        raise ValueError("unsupported vLLM model runner; memory cannot be verified")
    torch.cuda.synchronize()
    # vLLM runner owns KV + hybrid state cache. Never subtract a log estimate.
    caches = getattr(runner, "kv_caches", ())
    result = inventory_model(model, excluded=caches)
    result["worker_rank"] = getattr(worker, "rank", 0)
    result["foreign_quant_methods"] = foreign_quant_methods(model)
    return result


class MemoryWorkerExtension:
    def evaluator_resident_model_inventory(self):
        return worker_inventory(self)


def install_vllm_adapter():
    """Use a named worker method; do not enable pickle-based RPC serialization."""
    from vllm import LLM
    original = LLM.__init__
    if getattr(original, "_evaluator_memory_adapter", False):
        return

    def initialize(self, *args, **kwargs):
        extension = kwargs.get("worker_extension_cls")
        if extension not in (None, "", "model_memory.MemoryWorkerExtension"):
            raise ValueError("existing worker extension needs an audited memory adapter")
        kwargs["worker_extension_cls"] = "model_memory.MemoryWorkerExtension"
        return original(self, *args, **kwargs)

    initialize._evaluator_memory_adapter = True
    LLM.__init__ = initialize


def collect(engine):
    """Only evaluator-supported adapters are accepted, not a self-reported byte count."""
    from vllm import LLM
    candidates = [engine, *vars(engine).values()]
    llms = {id(obj): obj for obj in candidates if isinstance(obj, LLM)}
    if len(llms) != 1:
        raise ValueError("model-memory unsupported: expose one vLLM LLM on the engine; "
                         "custom/CUDA-native backends require an evaluator-owned adapter")
    llm = next(iter(llms.values()))
    workers = llm.collective_rpc("evaluator_resident_model_inventory")
    if not isinstance(workers, list) or len(workers) != 1:
        raise ValueError("model-memory requires exactly one GPU worker")
    # Wrapper storage belongs to a different process. Reject rather than omit
    # it or assume aliases across processes; includes nested wrapper containers.
    import torch
    import types
    seen = {id(llm)}
    def check_wrapper(obj):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, torch.Tensor):
            if obj.is_cuda:
                raise ValueError("wrapper-held CUDA storage requires an audited adapter")
        elif isinstance(obj, dict):
            for child in obj.values():
                check_wrapper(child)
        elif isinstance(obj, (tuple, list, set, frozenset)):
            for child in obj:
                check_wrapper(child)
        elif not isinstance(obj, (type, types.ModuleType, types.FunctionType,
                                  types.MethodType)) and hasattr(obj, "__dict__"):
            check_wrapper(vars(obj))
    check_wrapper(engine)
    return {"protocol": PROTOCOL, "workers": workers}


def check_snapshot(snapshot, policy, phase):
    if snapshot.get("protocol") != PROTOCOL:
        raise ValueError("invalid model-memory snapshot protocol")
    workers = snapshot.get("workers")
    if not isinstance(workers, list) or len(workers) != 1:
        raise ValueError("missing model-memory worker inventory")
    amounts = []
    for worker in workers:
        foreign = worker.get("foreign_quant_methods") or []
        if foreign:
            raise ValueError("the compression is not the submission's own: weights are served by "
                             + ", ".join(foreign[:6]))
        rows = worker.get("storages")
        if not isinstance(rows, list) or not rows:
            raise ValueError("missing model storage evidence")
        if any(type(row.get("bytes")) is not int or row["bytes"] < 0 for row in rows):
            raise ValueError("invalid model storage bytes")
        amount = sum(row["bytes"] for row in rows)
        if type(worker.get("bytes")) is not int or worker["bytes"] != amount or amount <= 0:
            raise ValueError("inconsistent model storage inventory")
        amounts.append(amount)
    total = sum(amounts)
    limit = int(policy["limit_gib"] * 2**30)
    result = dict(snapshot, phase=phase, resident_bytes=total,
                  resident_gib=total / 2**30, limit_bytes=limit, passed=total <= limit)
    if total > limit:
        raise ValueError(f"model resident memory exceeds limit after {phase}: "
                         f"{total} bytes > {limit} bytes")
    return result


def report(snapshots, policy):
    if policy is None:
        return None
    return dict(protocol=PROTOCOL, limit_gib=policy["limit_gib"],
                passed=bool(snapshots) and all(s["passed"] for s in snapshots),
                max_resident_bytes=max((s["resident_bytes"] for s in snapshots), default=0),
                snapshots=snapshots, audit_status="pending", review_required=True)


def validate_report(value, policy):
    if policy is None:
        return
    if not isinstance(value, dict) or value.get("protocol") != PROTOCOL:
        raise ValueError("missing model-memory evidence")
    snapshots = value.get("snapshots", [])
    if (not snapshots or snapshots[0].get("phase") != "load"
            or not any(s.get("phase") != "load" for s in snapshots)):
        raise ValueError("model memory requires load and post-inference snapshots")
    for snapshot in snapshots:
        check_snapshot(snapshot, policy, snapshot["phase"])
