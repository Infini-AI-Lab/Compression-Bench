"""The storage format of the GPTQ compression baseline, and its decoder.

A GPTQ-quantized tensor of shape [*lead, out, in] (lead = the expert axis of a
stacked expert tensor, else empty) is stored as three safetensors entries:

    <name>.gptq_codes   uint8 [ceil(N*bits/8)]   N = prod(shape) codes, row-major
    <name>.gptq_zeros   uint8 [ceil(M*bits/8)]   M = prod(lead)*out*ceil(in/group_size)
    <name>.gptq_scales  float16 [*lead, out, ceil(in/group_size)]

Codes and zeros are packed densely: every run of 8 values is one little-endian
(8*bits)-bit integer holding value k in bits [k*bits, (k+1)*bits) (so 8 values
take `bits` bytes; the last run is zero-padded; at 8 bits a value is one byte). Column c of a row belongs to
group c // group_size and decodes to

    w = float32(scale) * (float32(code) - float32(zero))   -> cast to bfloat16,

the arithmetic the quantizer itself uses for the weights it feeds forward
(compression_bench/scripts/baselines/gptq_baseline.py), so the stored checkpoint decodes
to exactly the model GPTQ produced. `gptq_meta.json` lists every quantized
tensor (`tensors`: name, shape, bits, group_size, and `split` when a stacked
expert tensor stands for one checkpoint tensor per expert, `{e}` the expert).
gptq_packed_runtime.py serves the packed codes in vLLM without dequantizing them.
"""
from __future__ import annotations

import math

FORMAT = "compression-bench-gptq-sim-v1"
SUFFIXES = {"codes": ".gptq_codes", "scales": ".gptq_scales", "zeros": ".gptq_zeros"}
#: Runs of 8 codes processed per step when packing or unpacking (bounds temporary memory).
CHUNK_RUNS = 1 << 22


def packed_bytes(count: int, bits: int) -> int:
    """Bytes of `count` values packed at `bits` bits each (8-value runs, zero-padded; 8-bit: plain bytes)."""
    return count if bits == 8 else math.ceil(count / 8) * bits


def pack(values, bits: int):
    """uint8 values < 2**bits (any shape) -> the packed uint8 stream (1-D)."""
    import torch
    flat = values.reshape(-1).to(torch.uint8)
    if bits == 8:
        return flat.clone()
    if bits not in (2, 3, 4):
        raise ValueError(f"unsupported bit width {bits}")
    pad = (-flat.numel()) % 8
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    runs = flat.view(-1, 8)
    out = torch.empty(runs.shape[0], bits, dtype=torch.uint8, device=flat.device)
    shifts = torch.arange(8, device=flat.device, dtype=torch.int64) * bits
    byte_shifts = torch.arange(bits, device=flat.device, dtype=torch.int64) * 8
    for start in range(0, runs.shape[0], CHUNK_RUNS):
        part = runs[start:start + CHUNK_RUNS].to(torch.int64)
        word = (part << shifts).sum(dim=1)              # disjoint bit fields: the sum is their OR
        out[start:start + CHUNK_RUNS] = ((word[:, None] >> byte_shifts) & 0xFF).to(torch.uint8)
    return out.reshape(-1)


def unpack(stream, bits: int, count: int):
    """The packed uint8 stream -> `count` uint8 values (1-D)."""
    import torch
    stream = stream.reshape(-1)
    if stream.numel() != packed_bytes(count, bits):
        raise ValueError(f"{stream.numel()} packed bytes do not hold {count} {bits}-bit values")
    if bits == 8:
        return stream.to(torch.uint8)
    runs = stream.view(-1, bits)
    out = torch.empty(runs.shape[0], 8, dtype=torch.uint8, device=stream.device)
    shifts = torch.arange(8, device=stream.device, dtype=torch.int64) * bits
    byte_shifts = torch.arange(bits, device=stream.device, dtype=torch.int64) * 8
    mask = (1 << bits) - 1
    for start in range(0, runs.shape[0], CHUNK_RUNS):
        word = (runs[start:start + CHUNK_RUNS].to(torch.int64) << byte_shifts).sum(dim=1)
        out[start:start + CHUNK_RUNS] = ((word[:, None] >> shifts) & mask).to(torch.uint8)
    return out.reshape(-1)[:count]


def dequantize_rows(codes, scales, zeros, group_size: int):
    """codes uint8 [..., out, in], scales/zeros [..., out, groups] -> float32 weights."""
    import torch
    columns = codes.shape[-1]
    group = torch.arange(columns, device=codes.device) // group_size
    scale = scales.float().index_select(-1, group)
    zero = zeros.float().index_select(-1, group)
    return scale * (codes.float() - zero)


def original_names(entry: dict) -> list[str]:
    """The checkpoint tensors one quantized entry replaces."""
    if entry.get("split"):
        stack = math.prod(int(n) for n in entry["shape"][:-2])
        return [entry["split"].format(e=e) for e in range(stack)]
    return [entry["name"]]
