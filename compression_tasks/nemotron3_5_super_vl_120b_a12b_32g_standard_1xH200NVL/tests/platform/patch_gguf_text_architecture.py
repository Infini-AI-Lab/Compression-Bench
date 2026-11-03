"""Pinned platform compatibility fix; no quantization/kernel/math changes."""
import hashlib
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec('vllm_gguf_plugin')
if spec is None or not spec.origin:
    raise RuntimeError('Required pinned vllm_gguf_plugin is missing')
path = Path(spec.origin).parent / 'weights_adapter/qwen3_5.py'
text = path.read_text()
changes = [
    ('    "qwen3_5_text": "Qwen3_5ForConditionalGeneration",',
     '    "qwen3_5_text": "Qwen3_5ForCausalLM",'),
    ('    "qwen3_5_moe_text": "Qwen3_5MoeForConditionalGeneration",',
     '    "qwen3_5_moe_text": "Qwen3_5MoeForCausalLM",'),
]
for old, new in changes:
    if text.count(old) == 1 and text.count(new) == 0:
        text = text.replace(old, new, 1)
    elif text.count(old) == 0 and text.count(new) == 1:
        pass
    else:
        raise RuntimeError('Unexpected GGUF architecture map; refusing unreviewed patch')
# No packages installed or updated. The only permitted mutations are these mappings.
compile(text, str(path), 'exec')
if text != path.read_text():
    path.write_text(text)
print('GGUF text architecture compatibility fix:', hashlib.sha256(path.read_bytes()).hexdigest())
