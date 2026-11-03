"""Pinned compatibility edit: the pinned vllm-gguf-plugin against vLLM 0.32.0.

vLLM 0.32.0 dropped its private copies of the Qwen3.5 configuration classes
(`vllm.transformers_utils.configs.qwen3_5`, `.qwen3_5_moe`) for the ones
Transformers 5.19 ships (`transformers.Qwen3_5Config`, `Qwen3_5MoeConfig`).
The pinned plugin revision (e2b8ad53) imports the old modules, so it fails to
import at all. This points those two imports at Transformers' classes, which the
plugin only constructs as `config_cls(text_config=...)`; nothing else changes.
"""
import hashlib
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("vllm_gguf_plugin")
if spec is None or not spec.origin:
    raise SystemExit("the pinned vllm_gguf_plugin is missing")
path = Path(spec.origin).parent / "weights_adapter" / "qwen3_5.py"
ORIGINAL = "ed19007102c518a50e89a92fa2084c4034980f59aabd0881ce13df78df1fa2d5"
EDITS = [
    ("from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config\n",
     "from transformers import Qwen3_5Config\n"),
    ("from vllm.transformers_utils.configs.qwen3_5_moe import Qwen3_5MoeConfig\n",
     "from transformers import Qwen3_5MoeConfig\n"),
]
text = path.read_text()
if hashlib.sha256(text.encode()).hexdigest() != ORIGINAL:
    raise SystemExit(f"{path} is not the file this edit was written for; refusing an unreviewed change")
for old, new in EDITS:
    if text.count(old) != 1:
        raise SystemExit(f"expected {old.strip()!r} once in {path}")
    text = text.replace(old, new)
compile(text, str(path), "exec")
path.write_text(text)
print("vllm-gguf-plugin Qwen3.5 configs -> transformers:", hashlib.sha256(path.read_bytes()).hexdigest())
