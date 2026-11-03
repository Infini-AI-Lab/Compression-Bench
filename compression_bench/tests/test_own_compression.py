"""The compression is the submission's own (own_compression.py, model_memory.foreign_quant_methods).

Serving weights through vLLM's built-in quantization methods or kernels, or another
package's, is refused mechanically, from the code in the package stage and from the
loaded model in the census -- e.g. a helper that serves experts on vLLM's GPTQ-Marlin
path by importing `vllm.model_executor.layers.quantization.auto_gptq`.
"""
from __future__ import annotations

import importlib.util
import json
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
FINAL_VALIDATE = REPO / "compression_bench" / "harness" / "final_validate"


def load(name):
    spec = importlib.util.spec_from_file_location(f"own_{name}", FINAL_VALIDATE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


own = load("own_compression")
model_memory = load("model_memory")

OWN_FORMAT = '''
import torch, triton
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.fused_moe import FusedMoE, fused_experts
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
torch.ops.myformat.linear(x)
fused_experts(x, w1, w2, topk_weights, topk_ids)
'''

FOREIGN = {
    "from vllm.model_executor.layers.quantization.gptq_marlin import GPTQMarlinConfig": "gptq_marlin",
    "from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig": "auto_gptq",
    "from vllm.model_executor.layers.quantization import gptq_marlin": "gptq_marlin",
    "from vllm.model_executor.layers.quantization import get_quantization_config": "get_quantization_config",
    "import vllm.model_executor.layers.quantization.utils.marlin_utils as mu": "marlin_utils",
    "from vllm import _custom_ops as ops": "_custom_ops",
    "from vllm.model_executor.layers.fused_moe.fused_marlin_moe import fused_marlin_moe": "fused_marlin_moe",
    "import importlib\nimportlib.import_module('humming')": "humming",
    "import vllm_gguf_plugin": "vllm_gguf_plugin",
    "import bitsandbytes": "bitsandbytes",
    "torch.ops._C.gptq_marlin_gemm(a, b)": "torch.ops._C",
    "fused_experts(x, w1, w2, tw, ti, use_int4_w4a16=True)": "use_int4_w4a16",
    "LLM(model='/sub/model', quantization='fp8')": "'fp8'",
    "kwargs = {'model': m, 'quantization': 'bitsandbytes'}": "'bitsandbytes'",
    "os.environ['VLLM_HUMMING_ONLINE_QUANT_CONFIG'] = '{\"quant_method\": \"channel\"}'": "VLLM_HUMMING_ONLINE",
}


def test_a_submissions_own_format_passes():
    assert own.scan_source(OWN_FORMAT, "own.py") == []
    assert own.scan_source("LLM(model=m, quantization='myformat')\n{'quantization': 'myformat'}", "own.py") == []


@pytest.mark.parametrize("code", sorted(FOREIGN))
def test_a_built_in_or_packaged_implementation_is_found(code):
    found = own.scan_source(code, "sub.py")
    assert found and FOREIGN[code] in found[0], found


def test_a_built_in_format_name_is_refused_and_the_package_stage_reports_it(tmp_path):
    (tmp_path / "model").mkdir()
    (tmp_path / "inference.py").write_text(OWN_FORMAT)
    (tmp_path / "model" / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "gptq"}}))
    report = own.scan(tmp_path)
    assert not report["passed"] and "vLLM built-in format" in report["findings"][0]
    (tmp_path / "model" / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "myformat"}}))
    assert own.scan(tmp_path)["passed"]


def _classes(module_name, *names):
    out = []
    base = object
    for name in names:
        base = type(name, (base,), {"__module__": module_name})
        out.append(base)
    return out


def test_the_census_finds_a_built_in_method_even_as_a_base_class():
    builtin, = _classes("vllm.model_executor.layers.quantization.gptq_marlin", "GPTQMarlinMoEMethod")
    abstract, = _classes("vllm.model_executor.layers.quantization.base_config", "QuantizeMethodBase")
    subclass = type("MyMoE", (builtin,), {"__module__": "my_submission.moe"})
    mine = type("MyLinear", (abstract,), {"__module__": "my_submission.linear"})
    unquantized = type("UnquantizedLinearMethod", (abstract,), {"__module__": "vllm.model_executor.layers.linear"})

    class Model:
        def __init__(self, methods):
            self.layers = [types.SimpleNamespace(quant_method=m) for m in methods]

        def modules(self):
            return iter(self.layers)

    assert model_memory.foreign_quant_methods(Model([mine(), unquantized(), None])) == []
    assert model_memory.foreign_quant_methods(Model([mine(), subclass()])) == [
        "vllm.model_executor.layers.quantization.gptq_marlin.GPTQMarlinMoEMethod"]


def test_a_foreign_method_in_the_census_fails_the_snapshot():
    snapshot = {"protocol": model_memory.PROTOCOL, "workers": [{
        "bytes": 10, "storages": [{"bytes": 10}],
        "foreign_quant_methods": ["vllm.model_executor.layers.quantization.gptq_marlin.GPTQMarlinMoEMethod"]}]}
    with pytest.raises(ValueError, match="not the submission's own"):
        model_memory.check_snapshot(snapshot, {"limit_gib": 8}, "load")
    snapshot["workers"][0]["foreign_quant_methods"] = []
    assert model_memory.check_snapshot(snapshot, {"limit_gib": 8}, "load")["passed"]


def test_every_evaluator_copy_carries_the_check():
    import sys
    sys.path.insert(0, str(REPO / "compression_tasks" / "_manage"))
    import render
    assert "own_compression.py" in render.VERIFIER_MODULES
