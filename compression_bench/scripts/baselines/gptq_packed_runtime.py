"""vLLM execution of the baseline's original row-packed codes and FP16 scales.

No weight is dequantized at load time. The Triton GEMM decodes a tile into
BF16 registers immediately before its dot product; embeddings decode only the
requested rows. The adapter supports tensor parallelism 1: dense layers, embeddings and
packed routed experts of the baseline's checkpoints. Nothing falls back to a dequantized model.
"""
import json
import math
import os
from pathlib import Path

import torch
import triton
import triton.language as tl
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig


class PackedGPTQConfig(QuantizationConfig):
    """Module-level class so vLLM's standard spawned workers can pickle it."""
    @classmethod
    def get_name(cls): return "compression_bench_packed_gptq"

    @classmethod
    def get_supported_act_dtypes(cls): return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls): return 90

    @classmethod
    def get_config_filenames(cls): return []

    @classmethod
    def from_config(cls, config): return cls()

    def get_quant_method(self, layer, prefix):
        install()
        return _quant_method_for_layer(layer, prefix)


@triton.jit
def _gemm(X, C, S, Z, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
          BITS: tl.constexpr, GROUP: tl.constexpr, BM: tl.constexpr = 16,
          BN: tl.constexpr = 64, BK: tl.constexpr = 64):
    mm = tl.program_id(0) * BM + tl.arange(0, BM)
    nn = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + kk
        x = tl.load(X + mm[:, None] * K + k[None, :], (mm[:, None] < M) & (k[None, :] < K), 0)
        # Explicit int64 avoids overflow on the vocabulary matrix.
        bit = (nn[:, None].to(tl.int64) * K + k[None, :]) * BITS
        byte, shift = bit // 8, bit % 8
        valid = (nn[:, None] < N) & (k[None, :] < K)
        lo = tl.load(C + byte, valid, 0).to(tl.int32)
        hi = tl.load(C + byte + 1, valid & (shift + BITS > 8), 0).to(tl.int32)
        codes = ((lo | (hi << 8)) >> shift) & ((1 << BITS) - 1)
        group = nn[:, None] * tl.cdiv(K, GROUP) + k[None, :] // GROUP
        scale = tl.load(S + group, valid, 0).to(tl.float32)
        zero = tl.load(Z + group, valid, 0).to(tl.float32)
        w = (scale * (codes.to(tl.float32) - zero)).to(tl.bfloat16)
        acc = tl.dot(x, tl.trans(w), acc)
    tl.store(Y + mm[:, None] * N + nn[None, :], acc, (mm[:, None] < M) & (nn[None, :] < N))


@triton.jit
def _embedding(IDS, C, S, Z, Y, T: tl.constexpr, K: tl.constexpr, BITS: tl.constexpr,
               GROUP: tl.constexpr, BLOCK: tl.constexpr = 256):
    t = tl.program_id(0)
    k = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    row = tl.load(IDS + t).to(tl.int64)
    bit = (row * K + k) * BITS
    byte, shift = bit // 8, bit % 8
    lo = tl.load(C + byte, k < K, 0).to(tl.int32)
    hi = tl.load(C + byte + 1, (k < K) & (shift + BITS > 8), 0).to(tl.int32)
    codes = ((lo | (hi << 8)) >> shift) & ((1 << BITS) - 1)
    group = row * tl.cdiv(K, GROUP) + k // GROUP
    scale = tl.load(S + group, k < K, 0).to(tl.float32)
    zero = tl.load(Z + group, k < K, 0).to(tl.float32)
    value = (scale * (codes.to(tl.float32) - zero)).to(tl.bfloat16)
    tl.store(Y + t * K + k, value, k < K)


@triton.jit
def _dequantize_matrix(C, S, Z, W, COUNT: tl.constexpr, K: tl.constexpr,
                       BITS: tl.constexpr, GROUP: tl.constexpr, BLOCK: tl.constexpr):
    pos = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, column = pos // K, pos % K
    byte = row.to(tl.int64) * (K * BITS // 8) + (column * BITS) // 8
    shift = (column * BITS) % 8
    lo = tl.load(C + byte, pos < COUNT, 0).to(tl.int32)
    hi = tl.load(C + byte + 1, (pos < COUNT) & (shift + BITS > 8), 0).to(tl.int32)
    code = ((lo | (hi << 8)) >> shift) & ((1 << BITS) - 1)
    group = row * tl.cdiv(K, GROUP) + column // GROUP
    scale = tl.load(S + group, pos < COUNT, 0).to(tl.float32)
    zero = tl.load(Z + group, pos < COUNT, 0).to(tl.float32)
    tl.store(W + pos, (scale * (code.to(tl.float32) - zero)).to(tl.bfloat16), pos < COUNT)


def matmul(x, codes, scales, zeros, bits, group=128):
    shape = x.shape
    m, k = x.numel() // shape[-1], shape[-1]
    n = scales.shape[0]
    if m > 128:
        # A per-call workspace makes large prefills efficient. This tensor is
        # discarded immediately and never attached to a layer or cached.
        # Single-request decode always executes the packed GEMM below.
        w = torch.empty((n, k), dtype=torch.bfloat16, device=x.device)
        _dequantize_matrix[(triton.cdiv(n * k, 1024),)](
            codes, scales, zeros, w, n * k, k, bits, group, BLOCK=1024, num_warps=4)
        return torch.nn.functional.linear(x, w)
    y = torch.empty((m, n), dtype=x.dtype, device=x.device)
    _gemm[(triton.cdiv(m, 16), triton.cdiv(n, 64))](
        x.contiguous(), codes, scales, zeros, y, m, n, k, bits, group, BM=16, BN=64, BK=64, num_warps=4)
    return y.reshape(*shape[:-1], n)


def embedding(ids, codes, scales, zeros, bits, k, group=128):
    y = torch.empty((*ids.shape, k), dtype=torch.bfloat16, device=ids.device)
    _embedding[(ids.numel(), triton.cdiv(k, 256))](ids.contiguous(), codes, scales, zeros, y,
                                                ids.numel(), k, bits, group, BLOCK=256, num_warps=4)
    return y


# Opaque operators preserve vLLM's dynamic prefill shapes and CUDA graph mode;
# compiling a token-count constexpr directly would specialize those shapes.
@torch.library.custom_op("compression_bench_packed::linear", mutates_args=())
def packed_linear(x: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor,
                  zeros: torch.Tensor, bits: int, group: int) -> torch.Tensor:
    return matmul(x, codes, scales, zeros, bits, group)


@packed_linear.register_fake
def _linear_fake(x, codes, scales, zeros, bits, group):
    return x.new_empty((*x.shape[:-1], scales.shape[0]))


@torch.library.custom_op("compression_bench_packed::embedding", mutates_args=())
def packed_embedding(ids: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor,
                     zeros: torch.Tensor, bits: int, k: int, group: int) -> torch.Tensor:
    return embedding(ids, codes, scales, zeros, bits, k, group)


@packed_embedding.register_fake
def _embedding_fake(ids, codes, scales, zeros, bits, k, group):
    return torch.empty((*ids.shape, k), dtype=torch.bfloat16, device=ids.device)


@triton.jit
def _routed(X, C, S, Z, IDS, ROUTE_W, Y, N: tl.constexpr, K: tl.constexpr,
            TOPK: tl.constexpr, BITS: tl.constexpr, PER_ROUTE: tl.constexpr,
            SCALE_INPUT: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    route = tl.program_id(0)
    expert = tl.load(IDS + route).to(tl.int64)
    rows = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    xrow = route if PER_ROUTE else route // TOPK
    routing_scale = tl.load(ROUTE_W + route).to(tl.float32) if SCALE_INPUT else 1.0
    acc = tl.full((BN,), 0, tl.float32)
    for block in range(tl.cdiv(K, BK)):
        k = block * BK + kk
        x = tl.load(X + xrow * K + k, k < K, 0)
        x = (x.to(tl.float32) * routing_scale).to(tl.bfloat16).to(tl.float32)
        bit = ((expert * N + rows[:, None]) * K + k[None, :]) * BITS
        byte, shift = bit // 8, bit % 8
        valid = (expert >= 0) & (rows[:, None] < N) & (k[None, :] < K)
        lo = tl.load(C + byte, valid, 0).to(tl.int32)
        hi = tl.load(C + byte + 1, valid & (shift + BITS > 8), 0).to(tl.int32)
        code = ((lo | (hi << 8)) >> shift) & ((1 << BITS) - 1)
        group = (expert * N + rows[:, None]) * tl.cdiv(K, 128) + k[None, :] // 128
        scale = tl.load(S + group, valid, 0).to(tl.float32)
        zero_bit = group * BITS
        zero_byte, zero_shift = zero_bit // 8, zero_bit % 8
        zlo = tl.load(Z + zero_byte, valid, 0).to(tl.int32)
        zhi = tl.load(Z + zero_byte + 1, valid & (zero_shift + BITS > 8), 0).to(tl.int32)
        zero = (((zlo | (zhi << 8)) >> zero_shift) & ((1 << BITS) - 1)).to(tl.float32)
        w = (scale * (code.to(tl.float32) - zero)).to(tl.bfloat16).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    tl.store(Y + route * N + rows, acc.to(tl.bfloat16), rows < N)


@triton.jit
def _routed_gemm(X, C, S, Z, SORTED, EXPERTS, COUNT, ROUTE_W, Y,
                 ROUTES: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                 TOPK: tl.constexpr, BITS: tl.constexpr, PER_ROUTE: tl.constexpr,
                 SCALE_INPUT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    block = tl.program_id(0)
    if block * BM >= tl.load(COUNT):
        return
    expert = tl.load(EXPERTS + block).to(tl.int64)
    routes = tl.load(SORTED + block * BM + tl.arange(0, BM))
    rows = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    xrows = routes if PER_ROUTE else routes // TOPK
    weights = tl.load(ROUTE_W + routes, routes < ROUTES, 0) if SCALE_INPUT else tl.full((BM,), 1, tl.float32)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + kk
        x = tl.load(X + xrows[:, None] * K + k[None, :],
                    (routes[:, None] < ROUTES) & (k[None, :] < K), 0)
        x = (x.to(tl.float32) * weights[:, None]).to(tl.bfloat16)
        bit = ((expert * N + rows[:, None]) * K + k[None, :]) * BITS
        byte, shift = bit // 8, bit % 8
        valid = (expert >= 0) & (rows[:, None] < N) & (k[None, :] < K)
        lo = tl.load(C + byte, valid, 0).to(tl.int32)
        hi = tl.load(C + byte + 1, valid & (shift + BITS > 8), 0).to(tl.int32)
        code = ((lo | (hi << 8)) >> shift) & ((1 << BITS) - 1)
        group = (expert * N + rows[:, None]) * tl.cdiv(K, 128) + k[None, :] // 128
        scale = tl.load(S + group, valid, 0).to(tl.float32)
        zero_bit = group * BITS
        zb, zs = zero_bit // 8, zero_bit % 8
        zlo = tl.load(Z + zb, valid, 0).to(tl.int32)
        zhi = tl.load(Z + zb + 1, valid & (zs + BITS > 8), 0).to(tl.int32)
        zero = (((zlo | (zhi << 8)) >> zs) & ((1 << BITS) - 1)).to(tl.float32)
        w = (scale * (code.to(tl.float32) - zero)).to(tl.bfloat16)
        acc = tl.dot(x, tl.trans(w), acc)
    tl.store(Y + routes[:, None] * N + rows[None, :], acc,
             (routes[:, None] < ROUTES) & (rows[None, :] < N))


@torch.library.custom_op("compression_bench_packed::routed", mutates_args=())
def packed_routed(x: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor,
                  zeros: torch.Tensor, ids: torch.Tensor, route_weights: torch.Tensor,
                  bits: int, k: int, per_route: bool, scale_input: bool) -> torch.Tensor:
    n = scales.shape[1]
    out = torch.empty((ids.numel(), n), device=x.device, dtype=x.dtype)
    if ids.shape[0] > 128:
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
        sorted_ids, experts, count = moe_align_block_size(ids, 16, scales.shape[0])
        _routed_gemm[(experts.numel(), triton.cdiv(n, 64))](
            x.contiguous(), codes, scales, zeros, sorted_ids, experts, count,
            route_weights.contiguous(), out, ids.numel(), n, k, ids.shape[1], bits,
            per_route, scale_input, BM=16, BN=64, BK=64, num_warps=4)
        return out
    _routed[(ids.numel(), triton.cdiv(n, 32))](x.contiguous(), codes, scales, zeros,
        ids.contiguous(), route_weights.contiguous(), out, n, k, ids.shape[1], bits,
        per_route, scale_input, BN=32, BK=128, num_warps=4)
    return out


@packed_routed.register_fake
def _routed_fake(x, codes, scales, zeros, ids, route_weights, bits, k, per_route, scale_input):
    return x.new_empty((ids.numel(), scales.shape[1]))


def _canonical(name):
    return (name.replace("model.language_model.", "model.").replace("language_model.model.", "model.")
            .removeprefix("language_model.").replace("backbone.", "model.").replace(".routed_experts", ""))


def inspect_model(model):
    totals, packed, large_bf16 = {}, [], []
    seen = set()
    for name, param in model.named_parameters():
        storage = param.untyped_storage()
        key = (str(param.device), storage.data_ptr(), storage.nbytes())
        if key not in seen:
            seen.add(key)
            dtype = str(param.dtype)
            totals[dtype] = totals.get(dtype, 0) + storage.nbytes()
        if param.dtype == torch.bfloat16 and param.numel() * param.element_size() >= 32 * 2 ** 20:
            large_bf16.append({"name": name, "shape": list(param.shape), "bytes": param.numel() * param.element_size()})
    for name, module in model.named_modules():
        if hasattr(module, "qweight") and hasattr(module, "qzeros"):
            for key, dtype in (("qweight", torch.uint8), ("scales", torch.float16), ("qzeros", torch.uint8)):
                p = getattr(module, key)
                if p.dtype != dtype or p.device.type != "cuda":
                    raise ValueError(f"{name}.{key} is not in the required packed GPU representation")
            if "weight" in dict(module.named_parameters(recurse=False)):
                raise ValueError(f"{name} retains a dequantized weight parameter")
            packed.append({"module": name, "codes_shape": list(module.qweight.shape), "scale_shape": list(module.scales.shape)})
        if hasattr(module, "w13_qweight") and hasattr(module, "w2_qweight"):
            for stem in ("w13", "w2"):
                for suffix, dtype in (("qweight", torch.uint8), ("scales", torch.float16), ("qzeros", torch.uint8)):
                    p = getattr(module, stem + "_" + suffix)
                    if p.dtype != dtype or p.device.type != "cuda":
                        raise ValueError(f"{name}.{stem}_{suffix} is not in the packed GPU representation")
            packed.append({"module": name, "expert_up_codes_shape": list(module.w13_qweight.shape),
                           "expert_down_codes_shape": list(module.w2_qweight.shape),
                           "expert_zero_points_packed": True})
    if not packed:
        raise ValueError("No packed modules were found in the loaded model")
    return {"parameter_storage_bytes_by_dtype": totals, "packed_modules": packed,
            "large_bf16_parameters": large_bf16, "fully_dequantized_checkpoint": False}

def record_residency(_llm):
    path = Path(os.environ["COMPRESSION_BENCH_PACKED_RESIDENCY_REPORT"])
    if not path.is_file() or not json.loads(path.read_text()):
        raise ValueError("No packed GPU residency report was produced by the vLLM worker")


def install():
    from vllm.model_executor.layers.quantization import register_quantization_config
    from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding, UnquantizedEmbeddingMethod
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    from vllm.model_executor.utils import set_weight_attrs
    from vllm.model_executor.layers.fused_moe import RoutedExperts, FusedMoEMethodBase, FusedMoEQuantConfig
    import gptq_codec

    if getattr(DefaultModelLoader._get_weights_iterator, "_compression_bench_packed", False):
        return
    meta = json.loads(Path(os.environ["COMPRESSION_BENCH_PACKED_META"]).read_text())
    entries = {_canonical(e["name"].removesuffix(".weight")): e for e in meta["tensors"]}

    class MoEMethod(FusedMoEMethodBase):
        def __init__(self, config, bits):
            super().__init__(config)
            self.bits = bits
            self.moe_quant_config = FusedMoEQuantConfig.make()

        def create_weights(self, layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype, **attrs):
            self.hidden, self.intermediate = hidden_size, intermediate_size_per_partition
            self.gated = self.moe.is_act_and_mul
            for stem, n, k in (("w13", intermediate_size_per_partition * (2 if self.gated else 1), hidden_size),
                               ("w2", hidden_size, intermediate_size_per_partition)):
                for suffix, columns, dtype in (("qweight", k * self.bits // 8, torch.uint8),
                                               ("scales", math.ceil(k / 128), torch.float16),
                                               ("qzeros", math.ceil(k / 128), torch.uint8)):
                    shape = ((num_experts, gptq_codec.packed_bytes(n * columns, self.bits))
                             if suffix == "qzeros" else (num_experts, n, columns))
                    p = torch.nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)
                    set_weight_attrs(p, {"weight_loader": self.loader})
                    layer.register_parameter(stem + "_" + suffix, p)

        def loader(self, param, loaded_weight, weight_name, shard_id, expert_id, return_success=False, **kwargs):
            if not 0 <= expert_id < param.shape[0]:
                return False
            target = param.data[expert_id]
            if self.gated and shard_id in ("w1", "w3"):
                count = target.numel() // 2 if target.ndim == 1 else self.intermediate
                offset = 0 if shard_id == "w1" else count
                target = target[offset:offset + count]
            if target.shape != loaded_weight.shape:
                raise ValueError(f"Expert packed shape mismatch {weight_name}: {loaded_weight.shape} != {target.shape}")
            target.copy_(loaded_weight)
            return True

        def get_fused_moe_quant_config(self, layer):
            return FusedMoEQuantConfig.make()

        def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
            # vLLM 0.32's MoERunner launches the shared branch before this
            # callback, waits for its stream, and combines its result afterward.
            on_input = layer.apply_router_weight_on_input
            up = packed_routed(x, layer.w13_qweight, layer.w13_scales, layer.w13_qzeros,
                               topk_ids, topk_weights, self.bits, self.hidden, False, on_input)
            if self.gated:
                if not any(name in str(layer.activation).lower() for name in ("silu", "swiglu")):
                    raise NotImplementedError(f"Unsupported packed gated expert activation {layer.activation}")
                gate, value = up.chunk(2, dim=-1)
                activated = (torch.nn.functional.silu(gate.float()) * value.float()).to(x.dtype)
            else:
                activation = str(layer.activation).lower()
                if "relu2" not in activation and "relu_squared" not in activation:
                    raise NotImplementedError(f"Unsupported packed expert activation {layer.activation}")
                activated = torch.relu(up.float()).square().to(x.dtype)
            down = packed_routed(activated, layer.w2_qweight, layer.w2_scales, layer.w2_qzeros,
                                 topk_ids, topk_weights, self.bits, self.intermediate, True, False)
            routed = down.view(x.shape[0], topk_ids.shape[1], self.hidden).float()
            if not on_input:
                routed = routed * topk_weights.unsqueeze(-1)
            return routed.sum(dim=1).to(x.dtype)

    class Method(LinearMethodBase):
        def __init__(self, bits):
            self.bits = bits

        def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                           input_size, output_size, params_dtype, **attrs):
            if input_size_per_partition != input_size or sum(output_partition_sizes) != output_size:
                raise NotImplementedError("The packed baseline adapter requires tensor parallelism 1")
            self.k = input_size
            self.parts = list(output_partition_sizes)
            self.rows = sum(self.parts)
            groups = math.ceil(input_size / 128)
            for name, columns, dtype in (("qweight", input_size * self.bits // 8, torch.uint8),
                                          ("scales", groups, torch.float16), ("qzeros", groups, torch.uint8)):
                param = torch.nn.Parameter(torch.zeros((self.rows, columns), dtype=dtype), requires_grad=False)
                set_weight_attrs(param, {"weight_loader": self.loader})
                layer.register_parameter(name, param)

        def loader(self, param, loaded, shard_id=None, **kwargs):
            if shard_id is None:
                if loaded.shape != param.shape:
                    raise ValueError(f"Packed parameter shape mismatch: {loaded.shape} != {param.shape}")
                param.data.copy_(loaded)
                return
            if isinstance(shard_id, str):
                shard_id = {"q": 0, "k": 1, "v": 2, "qkv": 0, "z": 1}.get(shard_id, shard_id)
            if isinstance(shard_id, tuple):
                if not shard_id or list(shard_id) != list(range(shard_id[0], shard_id[-1] + 1)):
                    raise ValueError(f"Noncontiguous packed shards {shard_id!r}")
                start = sum(self.parts[:shard_id[0]])
                count = sum(self.parts[s] for s in shard_id)
                view = param.data[start:start + count]
                if loaded.shape != view.shape:
                    raise ValueError(f"Packed tuple shard shape mismatch: {loaded.shape} != {view.shape}")
                view.copy_(loaded)
                return
            if not isinstance(shard_id, int) or not 0 <= shard_id < len(self.parts):
                raise ValueError(f"Unknown packed shard {shard_id!r}")
            start = sum(self.parts[:shard_id])
            view = param.data[start:start + self.parts[shard_id]]
            if loaded.shape != view.shape:
                raise ValueError(f"Packed shard shape mismatch: {loaded.shape} != {view.shape}")
            view.copy_(loaded)

        def apply(self, layer, x, bias=None):
            y = packed_linear(x, layer.qweight, layer.scales, layer.qzeros, self.bits, 128)
            return y if bias is None else y + bias

        def embedding(self, layer, ids):
            return packed_embedding(ids, layer.qweight, layer.scales, layer.qzeros, self.bits, self.k, 128)

    def method_for_layer(layer, prefix):
            if isinstance(layer, RoutedExperts):
                prefix = _canonical(prefix)
                found = [e for k, e in entries.items() if k.startswith(prefix + ".") and len(e["shape"]) == 3]
                if not found or len({e["bits"] for e in found}) != 1:
                    raise ValueError(f"No compatible packed expert tensors for {prefix}")
                return MoEMethod(layer.moe_config, found[0]["bits"])
            if not isinstance(layer, (LinearBase, VocabParallelEmbedding)):
                return None
            prefix = _canonical(prefix)
            keys = [prefix]
            for target, sources in (("qkv_proj", ("q_proj", "k_proj", "v_proj")),
                                    ("gate_up_proj", ("gate_proj", "up_proj")),
                                    ("in_proj_qkvz", ("in_proj_qkv", "in_proj_z"))):
                if prefix.endswith("." + target):
                    keys = [prefix.removesuffix(target) + source for source in sources]
            found = [entries[k] for k in keys if k in entries]
            if not found:
                return UnquantizedLinearMethod() if isinstance(layer, LinearBase) else UnquantizedEmbeddingMethod()
            if len(found) != len(keys) or len({e["bits"] for e in found}) != 1:
                raise ValueError(f"Incompatible packed fusion at {prefix}")
            return Method(found[0]["bits"])

    global _quant_method_for_layer
    _quant_method_for_layer = method_for_layer
    register_quantization_config("compression_bench_packed_gptq")(PackedGPTQConfig)

    # This pinned Qwen model builds its embedding without passing quant_config.
    # Inject it only for the exact vocabulary/hidden shape named by the baseline
    # metadata; otherwise the packed embedding would silently become BF16.
    import inspect
    original_embedding_init = VocabParallelEmbedding.__init__
    embedding_entries = [e for e in meta["tensors"] if e["name"].endswith(".embed_tokens.weight")]

    def embedding_init(self, num_embeddings, embedding_dim, *args, **kwargs):
        matches = [e for e in embedding_entries if e["shape"] == [num_embeddings, embedding_dim]]
        if matches and self.__class__ is VocabParallelEmbedding:
            bound = inspect.signature(original_embedding_init).bind_partial(self, num_embeddings, embedding_dim, *args, **kwargs)
            bound.arguments["quant_config"] = PackedGPTQConfig()
            bound.arguments["prefix"] = _canonical(matches[0]["name"].removesuffix(".weight"))
            return original_embedding_init(*bound.args, **bound.kwargs)
        return original_embedding_init(self, num_embeddings, embedding_dim, *args, **kwargs)

    VocabParallelEmbedding.__init__ = embedding_init

    original = DefaultModelLoader._get_weights_iterator

    def iterator(self, source):
        tensors = original(self, source)
        index = Path(source.model_or_path) / "model.safetensors.index.json"
        allowed = set(json.loads(index.read_text())["weight_map"])
        by_name = {e["name"]: e for e in meta["tensors"]}
        originals = {n for e in meta["tensors"] for n in gptq_codec.original_names(e)}
        pending, seen = {}, set()
        for name, tensor in tensors:
            if name not in allowed:
                continue
            part = next((k for k, suffix in gptq_codec.SUFFIXES.items() if name.endswith(suffix)), None)
            if part is None:
                if name not in originals:
                    yield name, tensor
                continue
            base = name.removesuffix(gptq_codec.SUFFIXES[part])
            if base not in by_name:
                raise ValueError(f"Unknown packed tensor {base}")
            parts = pending.setdefault(base, {})
            parts[part] = tensor
            if len(parts) != 3:
                continue
            entry = by_name[base]
            *lead, n, k = entry["shape"]
            stack = math.prod(lead) if lead else 1
            bits, groups = entry["bits"], math.ceil(k / entry["group_size"])
            codes = parts["codes"].reshape(stack, n, k * bits // 8)
            scales = parts["scales"].reshape(stack, n, groups)
            if lead:
                if n * groups % 8:
                    raise ValueError("An expert's packed zero-point stream must end on an eight-value boundary")
                zeros = parts["zeros"].reshape(stack, gptq_codec.packed_bytes(n * groups, bits))
            else:
                zeros = gptq_codec.unpack(parts["zeros"], bits, n * groups).reshape(1, n, groups)
            for e in range(stack):
                if entry.get("split"):
                    stem = entry["split"].format(e=e).removesuffix(".weight")
                elif lead:
                    prefix, projection = base.rsplit(".", 1)
                    stem = f"{prefix}.{e}.{projection}"
                else:
                    stem = base.removesuffix(".weight")
                yield stem + ".qweight", codes[e]
                yield stem + ".scales", scales[e]
                yield stem + ".qzeros", zeros[e]
            seen.add(base)
            del pending[base]
        if pending or seen != set(by_name):
            raise ValueError(f"Incomplete packed load: {len(seen)}/{len(by_name)} tensors")
        Path(os.environ["COMPRESSION_BENCH_PACKED_LOAD_REPORT"]).write_text(json.dumps({
            "packed_tensors": len(seen), "dequantized_tensors_at_load": 0,
            "scale_dtype": "float16", "weight_dot_dtype": "bfloat16", "tensor_parallel_size": 1}) + "\n")

    iterator._compression_bench_packed = True
    DefaultModelLoader._get_weights_iterator = iterator
    original_load_model = DefaultModelLoader.load_model

    def load_model(self, *args, **kwargs):
        model = original_load_model(self, *args, **kwargs)
        # Inspect in the loading worker, without changing the engine's normal
        # serialization policy or adding an RPC to the timed execution path.
        report = inspect_model(model)
        Path(os.environ["COMPRESSION_BENCH_PACKED_RESIDENCY_REPORT"]).write_text(json.dumps([report], indent=2) + "\n")
        return model

    DefaultModelLoader.load_model = load_model
