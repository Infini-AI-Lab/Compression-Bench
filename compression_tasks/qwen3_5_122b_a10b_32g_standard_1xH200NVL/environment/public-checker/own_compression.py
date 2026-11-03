"""The submission's compression is its own.

A weight of the submission may not be served by a compression implementation it did
not write: none of vLLM's built-in quantization methods, their kernels and helpers, nor
another package's. Two checks enforce what can be read mechanically; the reviewer judges
the rest (why each method was chosen):

* `scan(submission)` -- the package stage, from the submission's code: no import of a
  built-in quantization method or its utilities (`vllm.model_executor.layers.quantization.*`
  except the registration hook and the abstract `base_config`), of vLLM's compiled ops or
  quantized-MoE kernels, or of another quantization package; no use of vLLM's built-in ops
  namespaces (`torch.ops._C`, `torch.ops._moe_C`); no quantized-expert switch of vLLM's
  fused-MoE kernel; no `quantization=` naming a built-in method; no switch of vLLM's own
  online compression (`VLLM_HUMMING_*_QUANT_CONFIG`); and no weight format registered under
  a built-in method's name.
* `model_memory.foreign_quant_methods()` -- the census inside the vLLM worker: no layer's
  quantization method, nor a class it inherits from, comes from those modules.

Stdlib only; a scan never imports or runs the submission.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

PROTOCOL = "own-compression-v1"

#: Allowed in `vllm.model_executor.layers.quantization`: the registration hook and the
#: abstract bases a submission's own format subclasses (and KV-cache scaling, not weights).
ALLOWED_QUANTIZATION = ("vllm.model_executor.layers.quantization",
                        "vllm.model_executor.layers.quantization.base_config",
                        "vllm.model_executor.layers.quantization.kv_cache")
#: Implementations no submission may use, by module prefix.
FORBIDDEN_MODULES = (
    "vllm.model_executor.layers.quantization.",
    "vllm._custom_ops", "vllm._C", "vllm._moe_C",
    "vllm.model_executor.layers.fused_moe.fused_marlin_moe",
    "vllm.model_executor.layers.fused_moe.cutlass_moe",
    "vllm.model_executor.layers.fused_moe.flashinfer_cutlass_moe",
    "vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe",
    "vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe",
    "vllm.model_executor.layers.fused_moe.deep_gemm_moe",
    "vllm_gguf_plugin", "gguf", "humming", "compressed_tensors", "triton_kernels",
    "bitsandbytes", "auto_gptq", "gptqmodel", "awq", "autoawq", "hqq", "optimum", "torchao",
    "llmcompressor", "quanto", "optimum.quanto", "aqlm", "exllama", "exllamav2", "exllamav3",
    "marlin", "flute", "vptq", "auto_round", "modelopt", "quip_sharp", "qtip", "spqr",
    "squeezellm", "fbgemm_gpu", "deep_gemm",
)
#: `torch.ops` namespaces of vLLM's and its kernels' compiled ops.
FORBIDDEN_OPS = ("_C", "_moe_C", "_rocm_C", "_C_cache_ops", "humming", "_gguf")
#: The quantized-expert switches of vLLM's fused-MoE kernel (its unquantized path is runtime).
FORBIDDEN_KEYWORDS = ("use_int4_w4a16", "use_int8_w8a16", "use_int8_w8a8", "use_fp8_w8a8",
                      "use_mxfp4_w4a4", "use_nvfp4_w4a4", "use_mxfp4_w4a16", "use_int4_w4a8")
#: Environment switches that make vLLM compress weights itself at load (Humming online
#: compression, measured in OPERATIONS §1).
FORBIDDEN_ENV = frozenset(("VLLM_HUMMING_ONLINE_QUANT_CONFIG", "VLLM_HUMMING_INPUT_QUANT_CONFIG"))
#: vLLM 0.32's built-in `quant_method` names: a submission's format registers its own.
BUILTIN_METHODS = frozenset((
    "aqlm", "auto-round", "auto_round", "awq", "awq_marlin", "awq_triton", "bitblas",
    "bitsandbytes", "compressed-tensors", "compressed_tensors", "cpu_awq", "deepspeedfp",
    "experts_int8", "fbgemm_fp8", "fp8", "fp_quant", "gguf", "gptq", "gptq_bitblas",
    "gptq_marlin", "gptq_marlin_24", "hqq", "humming", "inc", "ipex", "marlin", "modelopt",
    "modelopt_fp4", "moe_wna16", "mxfp4", "mxfp8", "petit_nvfp4", "ptpc_fp8", "qqq", "quark",
    "rtn", "torchao", "tpu_int8"))


#: What a submission may import from the quantization package itself: the registration
#: hook and the abstract bases (anything else there is a built-in method or its helpers).
ALLOWED_QUANTIZATION_NAMES = frozenset(("register_quantization_config", "QuantizationConfig",
                                        "QuantizeMethodBase", "QuantizationMethods",
                                        "QUANTIZATION_METHODS", "base_config", "kv_cache"))


def forbidden_module(name: str) -> bool:
    if not name or name in ALLOWED_QUANTIZATION:
        return False
    for prefix in FORBIDDEN_MODULES:
        if prefix.endswith("."):
            if name.startswith(prefix):
                return True
        elif name == prefix or name.startswith(prefix + "."):
            return True
    return False


def forbidden_from(module: str, name: str) -> bool:
    """`from module import name`: a forbidden module, or a forbidden name of the package."""
    if module == "vllm.model_executor.layers.quantization":
        return name not in ALLOWED_QUANTIZATION_NAMES
    if module in ALLOWED_QUANTIZATION:
        return False
    return forbidden_module(f"{module}.{name}")


def _dotted(node) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def scan_source(text: str, where: str) -> list[str]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []           # not Python source (Python cannot import it either); the census still runs
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [f"{where}:{node.lineno}: imports {a.name}" for a in node.names if forbidden_module(a.name)]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if forbidden_module(node.module):
                found.append(f"{where}:{node.lineno}: imports from {node.module}")
            else:
                found += [f"{where}:{node.lineno}: imports {node.module}.{a.name}" for a in node.names
                          if forbidden_from(node.module, a.name)]
        elif isinstance(node, ast.Attribute):
            chain = _dotted(node)
            head = chain.split(".")
            if len(head) >= 3 and head[:2] == ["torch", "ops"] and head[2] in FORBIDDEN_OPS:
                found.append(f"{where}:{node.lineno}: uses torch.ops.{head[2]}")
        elif isinstance(node, ast.Call):
            name = _dotted(node.func)
            if name.split(".")[-1] in ("import_module", "__import__") and node.args \
                    and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str) \
                    and forbidden_module(node.args[0].value):
                found.append(f"{where}:{node.lineno}: imports {node.args[0].value}")
            found += [f"{where}:{node.lineno}: calls with {k.arg}=" for k in node.keywords
                      if k.arg in FORBIDDEN_KEYWORDS]
            found += [f"{where}:{node.lineno}: asks vLLM for its built-in format {k.value.value!r}"
                      for k in node.keywords
                      if k.arg in ("quantization", "quant_method") and isinstance(k.value, ast.Constant)
                      and isinstance(k.value.value, str) and k.value.value.lower() in BUILTIN_METHODS]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in FORBIDDEN_ENV:
            found.append(f"{where}:{node.lineno}: sets {node.value} (vLLM compresses the weights itself)")
        elif isinstance(node, ast.Dict):
            found += [f"{where}:{node.lineno}: asks vLLM for its built-in format {v.value!r}"
                      for k, v in zip(node.keys, node.values)
                      if isinstance(k, ast.Constant) and k.value in ("quantization", "quant_method")
                      and isinstance(v, ast.Constant) and isinstance(v.value, str)
                      and v.value.lower() in BUILTIN_METHODS]
    return sorted(set(found))


def scan(submission: Path | str) -> dict:
    """Every finding in the submission's code and configuration (none: the check passes)."""
    root = Path(submission)
    findings = []
    files = sorted(p for p in root.rglob("*.py") if p.is_file() and "__pycache__" not in p.parts)
    for path in files:
        findings += scan_source(path.read_text(encoding="utf-8", errors="replace"),
                                str(path.relative_to(root)))
    config = root / "model" / "config.json"
    try:
        method = ((json.loads(config.read_text()).get("quantization_config") or {}).get("quant_method"))
    except (OSError, json.JSONDecodeError):
        method = None
    if isinstance(method, str) and method.lower() in BUILTIN_METHODS:
        findings.append(f"model/config.json: quant_method {method!r} is a vLLM built-in format")
    return {"protocol": PROTOCOL, "python_files": len(files), "findings": findings,
            "passed": not findings}
