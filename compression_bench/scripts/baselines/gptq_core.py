"""GPTQ (Frantar et al., ICLR 2023, arXiv 2210.17323) as the official code implements it
(IST-DASLab/gptq: gptq.py `GPTQ.fasterquant`, quant.py `Quantizer`, datautils.py
`get_c4`), with one batch axis added so a layer's experts are quantized together,
plus the package accounting of the baseline. Torch is imported only where needed;
the sampler and the accounting are plain Python (gptq_codec.py gives the packed sizes).

Faithful to the official code:
  * H = 2/n * sum(x x^T) in float32, n counted the way `GPTQ.add_batch` counts
    samples when the official per-sequence loop calls a layer once per sequence;
  * dampening 1% of mean(diag H); H^-1 by Cholesky, then its upper Cholesky factor;
  * lazy batch updates over blocks of 128 columns in natural order (no act-order);
  * asymmetric min-max grid including 0 (xmin = min(x, 0), xmax = max(x, 0), an
    all-zero row -> [-1, 1]), scale = (xmax - xmin) / maxq, zero = round(-xmin / scale),
    q = clamp(round(x / scale) + zero, 0, maxq), groups of 128 columns whose grid is
    set lazily from the updated weights when the group is reached.
Deviations (each forced):
  * a column whose input is always zero in calibration (diag H = 0) keeps its
    weights and is rounded with H_ii = 1; the official code zeroes it;
  * the scale is stored in float16 and the grid uses the float16-rounded scale
    (floored at the smallest float16 subnormal), so the stored codes decode to
    exactly the weights GPTQ's error feedback assumed; zero is clamped to [0, maxq].
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
import re
import struct
from pathlib import Path

from gptq_codec import packed_bytes

BLOCK_SIZE = 128
PERCDAMP = 0.01
GROUP_SIZE = 128
NSAMPLES = 128
SEQLEN = 2048
SEED = 0
FP16_TINY = 2.0 ** -24          # smallest positive float16 (subnormal)


# --------------------------------------------------------------------------- calibration

def c4_documents(path) -> list[str]:
    """The texts of a C4 json.gz shard in file order (the order `datasets` gives them)."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line)["text"] for line in handle]


def sample_c4(texts, tokenize, *, nsamples=NSAMPLES, seqlen=SEQLEN, seed=SEED):
    """The official draw (datautils.get_c4): random.seed(seed); for each sample, random
    documents until one tokenizes to more than `seqlen` tokens, then a random window."""
    rng = random.Random(seed)            # random.seed(seed) + module functions: the same stream
    windows, draws = [], []
    for _ in range(nsamples):
        while True:
            doc = rng.randint(0, len(texts) - 1)
            ids = list(tokenize(texts[doc]))
            if len(ids) > seqlen:
                break
        start = rng.randint(0, len(ids) - seqlen - 1)
        windows.append(ids[start:start + seqlen])
        draws.append([doc, start])
    return windows, draws


def token_hash(windows) -> str:
    digest = hashlib.sha256()
    for window in windows:
        digest.update(struct.pack(f"<{len(window)}I", *window))
    return digest.hexdigest()


# --------------------------------------------------------------------------- GPTQ

def find_params(x, maxq: int):
    """Quantizer.find_params(x, weight=True) with perchannel, asymmetric, no MSE search:
    x [..., rows, columns] float32 -> (scale, zero) [..., rows]."""
    import torch
    zero_t = torch.zeros((), device=x.device, dtype=x.dtype)
    xmin = torch.minimum(x.amin(dim=-1), zero_t)
    xmax = torch.maximum(x.amax(dim=-1), zero_t)
    both = (xmin == 0) & (xmax == 0)
    xmin = torch.where(both, -torch.ones_like(xmin), xmin)
    xmax = torch.where(both, torch.ones_like(xmax), xmax)
    scale = (xmax - xmin) / maxq
    scale = scale.to(torch.float16).float().clamp_min(FP16_TINY)       # stored as float16
    zero = torch.round(-xmin / scale).clamp(0, maxq)
    return scale, zero


def quantize(x, scale, zero, maxq: int):
    """quant.py `quantize`, also returning the integer code: (code, scale * (code - zero))."""
    import torch
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q, scale * (q - zero)


def gptq(W, H, bits: int, *, group_size=GROUP_SIZE, block_size=BLOCK_SIZE, percdamp=PERCDAMP):
    """GPTQ.fasterquant on a stack of matrices.

    W [E, rows, columns] (any float dtype), H [E, columns, columns] float32 (consumed).
    Returns (Q float32 [E, rows, columns], codes uint8, scales float16 [E, rows, G],
    zeros uint8 [E, rows, G], dead columns per matrix [E]).
    """
    import torch
    if group_size % block_size and block_size % group_size:
        raise ValueError("groups and blocks must nest")
    E, rows, columns = W.shape
    maxq = 2 ** bits - 1
    groups = math.ceil(columns / group_size)
    W = W.float().clone()
    diag = torch.diagonal(H, dim1=-2, dim2=-1)
    dead = diag == 0
    dead_count = dead.sum(dim=-1)
    diag[dead] = 1.0                    # official: H[dead, dead] = 1 (and W[:, dead] = 0: not done here)
    damp = percdamp * diag.mean(dim=-1)
    diag += damp[:, None]
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    Hinv = torch.linalg.cholesky(H, upper=True)
    del H
    Q = torch.zeros_like(W)
    codes = torch.zeros((E, rows, columns), dtype=torch.uint8, device=W.device)
    scales = torch.zeros((E, rows, groups), dtype=torch.float16, device=W.device)
    zeros = torch.zeros((E, rows, groups), dtype=torch.uint8, device=W.device)
    scale = zero = None
    for i1 in range(0, columns, block_size):
        i2 = min(i1 + block_size, columns)
        count = i2 - i1
        W1 = W[:, :, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[:, i1:i2, i1:i2]
        for i in range(count):
            column = i1 + i
            w = W1[:, :, i]
            d = Hinv1[:, i, i]
            if column % group_size == 0:
                scale, zero = find_params(W[:, :, column:column + group_size], maxq)
                g = column // group_size
                scales[:, :, g] = scale.to(torch.float16)
                zeros[:, :, g] = zero.to(torch.uint8)
            q, deq = quantize(w, scale, zero, maxq)
            Q1[:, :, i] = deq
            codes[:, :, column] = q.to(torch.uint8)
            err = (w - deq) / d[:, None]
            W1[:, :, i:] -= err[:, :, None] * Hinv1[:, i, i:][:, None, :]
            Err1[:, :, i] = err
        Q[:, :, i1:i2] = Q1
        W[:, :, i2:] -= torch.matmul(Err1, Hinv[:, i1:i2, i2:])
    return Q, codes, scales, zeros, dead_count


def rtn(W, bits: int, *, group_size=GROUP_SIZE):
    """Round-to-nearest on the same grid and groups (the embedding, a lookup table)."""
    import torch
    rows, columns = W.shape[-2:]
    W = W.float()
    maxq = 2 ** bits - 1
    groups = math.ceil(columns / group_size)
    lead = W.shape[:-2]
    codes = torch.empty(W.shape, dtype=torch.uint8, device=W.device)
    scales = torch.empty((*lead, rows, groups), dtype=torch.float16, device=W.device)
    zeros = torch.empty((*lead, rows, groups), dtype=torch.uint8, device=W.device)
    Q = torch.empty_like(W)
    for g in range(groups):
        sl = slice(g * group_size, (g + 1) * group_size)
        scale, zero = find_params(W[..., sl], maxq)
        q, deq = quantize(W[..., sl], scale[..., None], zero[..., None], maxq)
        codes[..., sl], Q[..., sl] = q.to(torch.uint8), deq
        scales[..., g], zeros[..., g] = scale.to(torch.float16), zero.to(torch.uint8)
    return Q, codes, scales, zeros


# --------------------------------------------------------------------------- package accounting

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1,
               "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}


def safetensors_header(path) -> dict:
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    header.pop("__metadata__", None)
    return header


def checkpoint_tensors(model_dir) -> dict[str, tuple[str, list[int], str]]:
    """name -> (dtype, shape, shard file) for every tensor a checkpoint's index lists."""
    model_dir = Path(model_dir)
    weight_map = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for shard in sorted(set(weight_map.values())):
        for name, info in safetensors_header(model_dir / shard).items():
            out[name] = (info["dtype"], info["shape"], shard)
    return out


def tensor_bytes(dtype: str, shape) -> int:
    return DTYPE_BYTES[dtype] * math.prod(shape)


def quantized_bytes(shape, bits: int, group_size=GROUP_SIZE) -> int:
    """Packed codes, packed zeros and float16 scales of one quantized tensor."""
    *lead, rows, columns = shape
    groups = (math.prod(lead) if lead else 1) * rows * math.ceil(columns / group_size)
    return packed_bytes(math.prod(shape), bits) + packed_bytes(groups, bits) + 2 * groups


def account(tensors: dict, layout: dict, block_bits: int, head_bits: int | None,
            extra_files: dict[str, int] | None = None) -> dict:
    """The package of one recipe, in bytes, by tensor class.

    `layout`: text (regex: the text model's tensors), quantize (regex: the block linear
    layers GPTQ quantizes), embed / head (names). head_bits None keeps both BF16.
    """
    text, quant = re.compile(layout["text"]), re.compile(layout["quantize"])
    out = {"blocks_quantized": 0, "kept": 0, "embed": 0, "head": 0, "files": sum((extra_files or {}).values()),
           "quantized_tensors": 0, "kept_tensors": 0}
    for name, (dtype, shape, _) in tensors.items():
        if not text.search(name):
            continue
        if name in (layout["embed"], layout["head"]):
            key = "embed" if name == layout["embed"] else "head"
            out[key] = tensor_bytes(dtype, shape) if head_bits is None else quantized_bytes(shape, head_bits)
        elif quant.search(name):
            out["blocks_quantized"] += quantized_bytes(shape, block_bits)
            out["quantized_tensors"] += 1
        else:
            out["kept"] += tensor_bytes(dtype, shape)
            out["kept_tensors"] += 1
    out["total"] = out["blocks_quantized"] + out["kept"] + out["embed"] + out["head"] + out["files"]
    out["total_gib"] = out["total"] / 2 ** 30
    return out


#: Allowance for what the plan cannot know exactly before writing: gptq_meta.json,
#: the index and the safetensors headers (measured after writing; well under 1 MiB).
METADATA_ALLOWANCE = 2 * 2 ** 20


def choose_recipe(tensors: dict, layout: dict, limit_gib: float, extra_files: dict[str, int]) -> dict:
    """The rule fixed before any run: the largest uniform b in (4, 3, 2) for the block
    linear layers whose package fits; embeddings and head BF16 if the package still fits,
    else 8-bit, else 4-bit."""
    limit = limit_gib * 2 ** 30
    tried = []
    for bits in (4, 3, 2):
        for head_bits in (None, 8, 4):
            acc = account(tensors, layout, bits, head_bits, extra_files)
            fits = acc["total"] + METADATA_ALLOWANCE <= limit
            tried.append({"block_bits": bits, "embed_head": "bf16" if head_bits is None else f"{head_bits}-bit",
                          "gib": round((acc["total"] + METADATA_ALLOWANCE) / 2 ** 30, 4), "fits": fits})
            if fits:
                return {"block_bits": bits, "head_bits": head_bits, "accounting": acc, "tried": tried}
    raise SystemExit(f"no recipe fits {limit_gib} GiB: {tried}")
