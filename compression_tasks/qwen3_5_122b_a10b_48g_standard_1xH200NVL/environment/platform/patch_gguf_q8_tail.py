"""Pinned Q8 tail correctness repair: use existing Triton GEMM for K tails."""
import hashlib
import importlib.util
from pathlib import Path
spec = importlib.util.find_spec('vllm_gguf_plugin')
if spec is None or not spec.origin:
    raise RuntimeError('Required pinned GGUF plugin missing')
p = Path(spec.origin).parent / 'ops.py'
s = p.read_text()
old = '''def ggml_mul_mat_a8(
    W: torch.Tensor,
    X: torch.Tensor,
    quant_type: int,
    row: int,
) -> torch.Tensor:
'''
new = old + '''    # Pinned CUDA Q8 tail read repair; packed weights remain unchanged.
    if int(quant_type) == 8 and X.shape[-1] % 256:
        return ggml_mul_mat_a8_triton(W, X, quant_type, row)
'''
original = 'daac821e871ff73df30b1836c17a7ef9574f8be6e4bedd5f6456c442b69c90db'
if new in s:
    restored = s.replace(new, old, 1)
    assert hashlib.sha256(restored.encode()).hexdigest() == original
else:
    assert hashlib.sha256(s.encode()).hexdigest() == original
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
compile(s, str(p), 'exec')
p.write_text(s)
print('Q8 tail correctness repair:', hashlib.sha256(p.read_bytes()).hexdigest())
