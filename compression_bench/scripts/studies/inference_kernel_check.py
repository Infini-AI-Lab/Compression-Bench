#!/usr/bin/env python3
"""Numerical checks only: two recorded kernel implementations, shared synthetic codes.

    inference_kernel_check.py [--versions vA vB] [--kernel qwen35_lowbit/kernels.py]
                              [--helpers /helpers/lowbit_common.py] [--out /result]

Run in the runtime image with each version's package mounted at /<version> and the packing reference at
--helpers. Both implementations get the same packed buffers and routing; each output is compared with BF16
weights unpacked by the reference, at a relative-L2 threshold fixed before measuring. This is not a full-model
benchmark, compressed-model construction, or quality evaluation; no latency is measured.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--versions", nargs=2, default=["v5", "v7"])
    p.add_argument("--kernel", default="qwen35_lowbit/kernels.py")
    p.add_argument("--helpers", default="/helpers/lowbit_common.py")
    p.add_argument("--out", default="/result")
    a = p.parse_args()
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    common = load("packing_reference", a.helpers)
    modules = {v: load(f"recorded_{v}_kernels", f"/{v}/{a.kernel}") for v in a.versions}
    threshold = 0.02  # fixed before measuring; BF16 output/activation rounding is expected
    results = []

    def matrix(n, k, bits):
        codes = torch.randint(0, 2**bits, (n, k), dtype=torch.uint8, device="cuda")
        scale = (torch.rand(k//128, n, device="cuda")*.01+.001).half()
        offset = (-scale.float()*(2**bits-1)/2).half()
        packed = common.pack_planes(codes, bits).transpose(-1, -2).contiguous()
        reference = common.dequant(codes, scale, offset).bfloat16()
        return packed, scale, offset, reference

    x = torch.randn(1, 3072, dtype=torch.bfloat16, device="cuda")
    p, s, m, w = matrix(2048, 3072, 4)
    ref = x.float() @ w.float().T
    for version, kernel in modules.items():
        y = kernel.dense_gemm(x, p, s, m, 4)
        error = float((y.float()-ref).norm()/ref.norm())
        results.append({"version": version, "test": "dense batch-one", "shape": [1, 2048, 3072], "bits": 4,
                        "relative_l2_error": error, "threshold": threshold, "passed": error < threshold})
    # Same packed buffers and routing are passed to both implementations.
    e, h, inter, topk = 16, 3072, 1024, 8
    widths13 = torch.tensor([2,3,4,5]*4)
    widths2 = torch.tensor([5,4,3,2]*4)

    def experts(n, k, widths):
        planes, offsets, scales, minima, refs = [], [], [], [], []
        offset = 0
        for bits in widths.tolist():
            p, s, m, w = matrix(n, k, bits)
            planes.append(p.flatten()); offsets.append(offset); offset += p.numel()
            scales.append(s); minima.append(m); refs.append(w)
        return {"planes": torch.cat(planes), "off": torch.tensor(offsets, dtype=torch.int64, device="cuda"),
                "nb": widths.to(device="cuda", dtype=torch.int32), "s": torch.stack(scales), "m": torch.stack(minima),
                "classes": tuple(sorted(set(widths.tolist())))}, refs

    w13, ref13 = experts(2*inter, h, widths13)
    w2, ref2 = experts(h, inter, widths2)
    x = torch.randn(1, h, device="cuda", dtype=torch.bfloat16)
    prob = torch.randn(1, e, device="cuda").softmax(-1)
    values, indices = prob.topk(topk, -1)
    values /= values.sum(-1, keepdim=True)
    reference = torch.zeros(1, h, device="cuda")
    for j in range(topk):
        expert = int(indices[0, j])
        gu = x[0].float() @ ref13[expert].float().T
        activation = (torch.nn.functional.silu(gu[:inter])*gu[inter:]).bfloat16()
        reference[0] += values[0,j]*(activation.float() @ ref2[expert].float().T)
    for version, kernel in modules.items():
        y = kernel.moe_forward(x, indices, values, w13, w2, inter)
        error = float((y.float()-reference).norm()/reference.norm())
        results.append({"version": version, "test": "MoE batch-one", "experts": e, "hidden": h, "intermediate": inter,
                        "topk": topk, "bits": [2,3,4,5], "relative_l2_error": error, "threshold": threshold, "passed": error < threshold})
    torch.cuda.synchronize()
    data = {"scope": "synthetic packed matrices, not the run's model weights", "seed": 0,
            "device": torch.cuda.get_device_name(0), "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "reference": "original bit-plane unpacking to BF16 weights, FP32 accumulation; BF16 intermediate activation",
            "latency_measured": False, "results": results, "passed": all(r["passed"] for r in results)}
    Path(a.out, "numerical_kernel_check.json").write_text(json.dumps(data, indent=2)+"\n")
    assert data["passed"], data


if __name__ == "__main__":
    main()
