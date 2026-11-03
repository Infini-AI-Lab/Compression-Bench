"""GPU numerical checks of the packed adapter against independently decoded BF16.

Run in the pinned runtime image, with an exclusively claimed GPU. This tests
cross-byte 3-bit codes, merged output shards, batch/prefill dimensions, and
vocabulary offsets beyond signed 32-bit bit addressing.
"""
import json
import sys
from pathlib import Path

import torch

import gptq_codec
from gptq_packed_runtime import embedding, matmul, packed_routed


def main(output):
    torch.manual_seed(1827)
    rows = []
    for bits in (2, 3, 4, 8):
        n, k = 192, 384
        codes = torch.randint(0, 2 ** bits, (n, k), device="cuda", dtype=torch.uint8)
        zeros = torch.randint(0, 2 ** bits, (n, 3), device="cuda", dtype=torch.uint8)
        scales = (torch.rand(n, 3, device="cuda") * .05).half()
        packed = gptq_codec.pack(codes, bits)
        independent = (scales.float().repeat_interleave(128, 1) *
                       (codes.float() - zeros.float().repeat_interleave(128, 1))).bfloat16()
        ids = torch.tensor([0, 7, 63, 128, 191], device="cuda")
        actual = embedding(ids, packed, scales, zeros, bits, k)
        assert torch.equal(actual, independent[ids]), f"{bits}-bit embedding differs"
        for m in (1, 17, 128, 129, 1024):
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            actual = matmul(x, packed, scales, zeros, bits)
            expected = x @ independent.T
            error = (actual - expected).abs()
            torch.testing.assert_close(actual, expected, rtol=.02, atol=.25)
            rows.append({"bits": bits, "input_shape": list(x.shape), "weight_shape": [n, k],
                         "max_absolute_error": error.max().item(), "mean_absolute_error": error.mean().item(),
                         "embedding_bit_exact": True})
        experts, routed_n, routed_k, m, topk = 7, 64, 384, 257, 3
        ec = torch.randint(0, 2 ** bits, (experts, routed_n, routed_k), device="cuda", dtype=torch.uint8)
        ez = torch.randint(0, 2 ** bits, (experts, routed_n, 3), device="cuda", dtype=torch.uint8)
        es = (torch.rand(experts, routed_n, 3, device="cuda") * .05).half()
        ep = gptq_codec.pack(ec, bits)
        ew = (es.float().repeat_interleave(128, -1) * (ec.float() - ez.float().repeat_interleave(128, -1))).bfloat16()
        ids = torch.randint(0, experts, (m, topk), device="cuda", dtype=torch.int32)
        rw = torch.softmax(torch.randn(m, topk, device="cuda"), -1)
        for per_route, scale_input in ((False, False), (False, True), (True, False)):
            x = torch.randn(m * topk if per_route else m, routed_k, device="cuda", dtype=torch.bfloat16)
            actual = packed_routed(x, ep, es, gptq_codec.pack(ez, bits), ids, rw, bits, routed_k, per_route, scale_input)
            expected = []
            for r, expert in enumerate(ids.flatten().tolist()):
                xx = x[r if per_route else r // topk]
                if scale_input:
                    xx = (xx.float() * rw.flatten()[r]).bfloat16()
                expected.append(xx @ ew[expert].T)
            expected = torch.stack(expected)
            torch.testing.assert_close(actual, expected, rtol=.02, atol=.25)
            rows.append({"test": "routed_packed_experts", "bits": bits, "per_route_inputs": per_route,
                         "scale_input": scale_input, "max_absolute_error": (actual - expected).abs().max().item()})
            # Exercise the single-token decode kernel using the same tensors.
            decode = packed_routed(x[:topk] if per_route else x[:1], ep, es,
                gptq_codec.pack(ez, bits), ids[:1], rw[:1], bits, routed_k, per_route, scale_input)
            torch.testing.assert_close(decode, expected[:topk], rtol=.02, atol=.25)
            rows.append({"test": "routed_packed_decode", "bits": bits, "per_route_inputs": per_route,
                         "scale_input": scale_input, "max_absolute_error": (decode - expected[:topk]).abs().max().item()})
    # Bit addresses beyond 2^31 (signed 32-bit overflow) are a failure mode of low-bit vocabulary kernels.
    n, k, bits = 131072, 4096, 8
    codes = torch.zeros((n, k), device="cuda", dtype=torch.uint8)
    codes[-1].fill_(239)
    scales = torch.full((n, k // 128), .03125, device="cuda", dtype=torch.float16)
    zeros = torch.full((n, k // 128), 127, device="cuda", dtype=torch.uint8)
    actual = embedding(torch.tensor([n - 1], device="cuda"), codes, scales, zeros, bits, k)
    assert torch.equal(actual, torch.full_like(actual, 3.5))
    rows.append({"test": "large_vocabulary_bit_offset", "last_row_bit_offset": (n - 1) * k * bits,
                 "embedding_bit_exact": True})
    routed = packed_routed(torch.ones(1, k, device="cuda", dtype=torch.bfloat16), codes,
                           scales.view(256, 512, k // 128), zeros.view(256, 512, k // 128),
                           torch.tensor([[255]], device="cuda", dtype=torch.int32),
                           torch.ones(1, 1, device="cuda"), bits, k, False, False)
    assert routed[0, -1].item() == 14336
    rows.append({"test": "large_expert_bit_offset", "last_row_bit_offset": (n - 1) * k * bits,
                 "last_output": routed[0, -1].item()})
    Path(output).write_text(json.dumps({"status": "passed", "torch_version": torch.__version__,
                                       "gpu": torch.cuda.get_device_name(), "tests": rows}, indent=2) + "\n")
    print("packed kernel numerical checks passed", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
