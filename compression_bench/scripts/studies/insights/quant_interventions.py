"""Controlled quantization interventions on Qwen3.8-27B: one fake-quantization pipeline, one factor
changed at a time, weight error, output-weighted error and model excess cross-entropy measured on the
same fixed tokens.

Pipeline (every variant): every decoder-layer linear (full attention q/k/v/o, linear attention
in_proj_qkv/in_proj_z/out_proj, MLP gate/up/down; 64 layers, 24.3 B weights) is replaced by its
dequantized reconstruction (fake quantization, stored back in BF16); embeddings, lm_head, norms, conv1d,
the 48-wide in_proj_a/in_proj_b gates and the vision tower stay BF16. Groups of 128 along the input
dimension. Factors:
  grid      U  uniform asymmetric (fp16 scale + b-bit zero per group; gptq_core.find_params)
            NF Gaussian Lloyd-Max levels (MSE-optimal for N(0,1)), symmetric, fp16 scale per group
            KM per-matrix k-means codebook (2^b fp16 entries) on w / group absmax, fp16 scale per group
            M  uniform symmetric midrise (no zero point), fp16 scale per group
  clip      none (min-max / absmax) vs per-group MSE-optimal shrink (20 factors 1.0 .. 0.525)
  rotation  none vs randomized block Hadamard (1024) on the input dimension: W' = W R, What = Q(W') R^T
            (orthogonal, folded back exactly)
  rounding  RTN vs GPTQ (official fasterquant: 1% damping, 128-column lazy blocks, natural order, group
            grid fitted lazily on the updated weights) with Hessians from BF16 activations of the FineWeb
            calibration set (non-sequential: every layer sees BF16 inputs, so every variant shares H)
  alloc     equal total bytes: attention +1 bit paid by 81 MLP matrices -1 bit, and the reverse
Measured per variant: relative weight MSE sum||W-What||^2 / sum||W||^2 (overall and per class), output-
weighted error tr(dW H dW^T)/tr(W H W^T) with H = X^T X on the calibration activations, bytes (codes +
scales + zeros + codebooks at the stated format; rotation signs from a seed, 0 bytes), and the per-token
NLL on the Qwen3.8 task's PUBLIC PPL blocks (environment/public-checker/evaluator-assets/ppl_inputs.json,
256 x 2048 tokens; the 1536 continuation tokens per block the Checker scores, positions 512..2047).
excess CE = mean(NLL_q - NLL_bf16) over the same 393,216 tokens (nats/token = ln PPL ratio).

Run inside gpu_worker.py (jobs call setup(), then run_variant(name) for each name in VARIANTS).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import QUANT_ERROR_DIR, REPO  # noqa: E402

MODEL = REPO / "compression_bench/models/Qwen3.8-27B-bf16"
TASK = REPO / "compression_tasks/qwen3_8_27b_8g_standard_1xH200NVL"
PUBLIC = TASK / "environment/public-checker/evaluator-assets"
FINEWEB = REPO / "compression_bench/environments/agent-base/datasets/fineweb/000_00000.parquet"
OUT = QUANT_ERROR_DIR / "interventions"
G = 128
CAL_N, CAL_LEN, SEED = 128, 2048, 0
PREFIX = 512
SHRINK = [1.0 - 0.025 * i for i in range(20)]
HBLOCK = 1024


# ----------------------------------------------------------------------------- grids
def lloyd_max_gauss(bits: int) -> torch.Tensor:
    """MSE-optimal 2^b levels for N(0,1) (Lloyd-Max), by fixed-point iteration on the exact conditions."""
    from scipy.stats import norm
    n = 1 << bits
    lv = np.array(norm.ppf((np.arange(n) + 0.5) / n))
    for _ in range(500):
        t = np.concatenate([[-np.inf], (lv[1:] + lv[:-1]) / 2, [np.inf]])
        a, b = t[:-1], t[1:]
        lv = (norm.pdf(a) - norm.pdf(b)) / (norm.cdf(b) - norm.cdf(a))
    return torch.tensor(lv, dtype=torch.float32)


class Grid:
    """fit(Wg [R, ng, G]) -> params; qdq(w, params) -> dequantized (same shape as w, broadcast params)."""
    zero_bits = 0
    codebook = 0

    def __init__(self, bits: int, clip: bool):
        self.bits, self.clip = bits, clip

    def search(self, Wg, cand):
        """Per group, the candidate params with the least squared error (first wins ties)."""
        best = bp = None
        for p in cand(Wg):
            e = ((self.qdq(Wg, p) - Wg) ** 2).sum(-1, keepdim=True)
            if best is None:
                best, bp = e, list(p)
            else:
                m = e < best
                best = torch.where(m, e, best)
                bp = [torch.where(m, x, y) for x, y in zip(p, bp)]
        return bp


class Uniform(Grid):
    """Asymmetric uniform, scale fp16, integer zero in [0, 2^b-1] (stored with b bits)."""

    def __init__(self, bits, clip):
        super().__init__(bits, clip)
        self.maxq = (1 << bits) - 1
        self.zero_bits = bits

    def _params(self, lo, hi):
        lo = torch.minimum(lo, torch.zeros_like(lo)); hi = torch.maximum(hi, torch.zeros_like(hi))
        both = (lo == 0) & (hi == 0)
        lo = torch.where(both, -torch.ones_like(lo), lo); hi = torch.where(both, torch.ones_like(hi), hi)
        s = ((hi - lo) / self.maxq).half().float().clamp_min(2.0 ** -24)
        z = torch.round(-lo / s).clamp(0, self.maxq)
        return s, z

    def fit(self, Wg):
        lo, hi = Wg.amin(-1, keepdim=True), Wg.amax(-1, keepdim=True)
        if not self.clip:
            return list(self._params(lo, hi))
        return self.search(Wg, lambda _: (self._params(lo * f, hi * f) for f in SHRINK))

    def qdq(self, w, p):
        s, z = p
        return s * (torch.clamp(torch.round(w / s) + z, 0, self.maxq) - z)


class Levels(Grid):
    """Symmetric level set times a per-group fp16 scale; nearest level by midpoints."""

    def __init__(self, bits, clip, levels):
        super().__init__(bits, clip)
        self.set_levels(levels)

    def set_levels(self, levels):
        self.lv = levels.float()
        self.mid = ((self.lv[1:] + self.lv[:-1]) / 2).contiguous()
        self.lmax = self.lv.abs().max()

    def fit(self, Wg):
        amax = Wg.abs().amax(-1, keepdim=True).clamp_min(1e-12)
        mk = lambda f: [(amax * f / self.lmax).half().float().clamp_min(2.0 ** -24)]
        if not self.clip:
            return mk(1.0)
        return self.search(Wg, lambda _: (mk(f) for f in SHRINK))

    def qdq(self, w, p):
        (s,) = p
        lv, mid = self.lv.to(w.device), self.mid.to(w.device)
        return s * lv[torch.bucketize((w / s).contiguous(), mid)]


class NF(Levels):
    def __init__(self, bits, clip):
        super().__init__(bits, clip, lloyd_max_gauss(bits))


class KMeans(Levels):
    """Per-matrix 2^b-entry codebook: Lloyd (15 iterations, quantile init) on 65,536 seeded samples of
    w / group absmax; the per-group scale then searched like the other grids (clip=True)."""
    codebook = 1

    def __init__(self, bits, clip=True):
        super().__init__(bits, clip, lloyd_max_gauss(bits))

    def prepare(self, W):
        R, K = W.shape
        Wg = W.reshape(R, K // G, G)
        x = (Wg / Wg.abs().amax(-1, keepdim=True).clamp_min(1e-12)).reshape(-1)
        gen = torch.Generator(device="cpu").manual_seed(SEED)
        idx = torch.randint(0, x.numel(), (65536,), generator=gen).to(x.device)
        smp = x[idx].float()
        n = 1 << self.bits
        cb = torch.quantile(smp, (torch.arange(n, device=smp.device) + 0.5) / n)
        for _ in range(15):
            a = torch.bucketize(smp, (cb[1:] + cb[:-1]) / 2)
            cb = torch.stack([smp[a == j].mean() if (a == j).any() else cb[j] for j in range(n)])
            cb, _ = cb.sort()
        self.set_levels(cb.half().float())


class Midrise(Levels):
    """Uniform symmetric midrise levels +-0.5, +-1.5, ... (no zero point), fp16 scale per group: the uniform grid with
    NF's storage and symmetry, to separate level shape from the zero point."""
    def __init__(self, bits, clip):
        n = 1 << bits
        super().__init__(bits, clip, torch.arange(n, dtype=torch.float32) - (n - 1) / 2)


def make_grid(kind: str, bits: int, clip: bool) -> Grid:
    return {"U": Uniform, "NF": NF, "KM": KMeans, "M": Midrise}[kind](bits, clip)


# ----------------------------------------------------------------------------- RTN / GPTQ / rotation
def rtn(W, grid: Grid):
    R, K = W.shape
    Wg = W.reshape(R, K // G, G)
    out = torch.empty_like(Wg)
    step = max(1, (1 << 26) // K)                  # row chunks bound the search's working set
    for a in range(0, R, step):
        w = Wg[a:a + step]
        out[a:a + step] = grid.qdq(w, grid.fit(w))
    return out.reshape(R, K)


def gptq(W, H, grid: Grid, block=128, percdamp=0.01):
    """IST-DASLab GPTQ.fasterquant (no act-order), group grid fitted lazily on the updated weights."""
    W = W.float().clone()
    R, K = W.shape
    H = H.clone()
    diag = torch.diagonal(H)
    dead = diag == 0
    diag[dead] = 1.0
    diag += percdamp * diag.mean()
    L = torch.linalg.cholesky(H)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
    del L, H
    Q = torch.zeros_like(W)
    p = None
    for i1 in range(0, K, block):
        i2 = min(i1 + block, K)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            c = i1 + i
            if c % G == 0:                          # block == G: the group is W1[:, i:i+G], already updated
                p = [x[:, 0, 0] for x in grid.fit(W1[:, i:i + G][:, None, :])]
            w = W1[:, i]
            q = grid.qdq(w, p)
            Q1[:, i] = q
            err = (w - q) / Hi[i, i]
            W1[:, i:] -= err[:, None] * Hi[i, i:][None, :]
            E1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q


def hadamard(n):
    Hm = torch.ones(1, 1)
    while Hm.shape[0] < n:
        Hm = torch.cat([torch.cat([Hm, Hm], 1), torch.cat([Hm, -Hm], 1)], 0)
    return Hm / math.sqrt(n)


_SIGNS = {}


def rot_signs(K, device):
    if K not in _SIGNS:
        g = torch.Generator().manual_seed(SEED + K)
        _SIGNS[K] = (torch.randint(0, 2, (K,), generator=g).float() * 2 - 1)
    return _SIGNS[K].to(device)


def rotate(W, inverse=False):
    """W [R, K] -> W R with R = diag(signs) blockdiag(H_1024); inverse: W R^T."""
    R_, K = W.shape
    Hm = hadamard(HBLOCK).to(W.device)
    s = rot_signs(K, W.device)
    if not inverse:
        return ((W * s).reshape(R_, K // HBLOCK, HBLOCK) @ Hm).reshape(R_, K)
    return ((W.reshape(R_, K // HBLOCK, HBLOCK) @ Hm.t()).reshape(R_, K)) * s


def rotate_H(H):
    """R^T H R for the same R."""
    return rotate(rotate(H).t().contiguous()).t().contiguous()


def quantize_matrix(W, H, spec, bits):
    """One matrix under a variant spec: grid kind, clip, rot, gptq."""
    grid = make_grid(spec["grid"], bits, spec.get("clip", False))
    Wq_in = rotate(W) if spec.get("rot") else W
    if isinstance(grid, KMeans):
        grid.prepare(Wq_in)
    if spec.get("gptq"):
        Hq = rotate_H(H) if spec.get("rot") else H
        Q = gptq(Wq_in, Hq, grid)
    else:
        Q = rtn(Wq_in, grid)
    if spec.get("rot"):
        Q = rotate(Q, inverse=True)
    nbits = Wq_in.numel() * bits + (Wq_in.numel() // G) * (16 + grid.zero_bits) + grid.codebook * (1 << bits) * 16
    return Q, nbits


# ----------------------------------------------------------------------------- model plumbing
def linears(layer):
    """(class, name, module, hessian key) for every quantized linear of one decoder layer."""
    out = []
    if layer.block_type == "linear_attention":
        a = layer.linear_attn
        out += [("linattn", "in_proj_qkv", a.in_proj_qkv, "in"), ("linattn", "in_proj_z", a.in_proj_z, "in"),
                ("linattn", "out_proj", a.out_proj, "o")]
    else:
        a = layer.self_attn
        out += [("attn", n, getattr(a, n), "in") for n in ("q_proj", "k_proj", "v_proj")]
        out += [("attn", "o_proj", a.o_proj, "o")]
    m = layer.mlp
    out += [("mlp_gate", "gate_proj", m.gate_proj, "mlp"), ("mlp_up", "up_proj", m.up_proj, "mlp"),
            ("mlp_down", "down_proj", m.down_proj, "down")]
    return out


def hook_modules(layer):
    mods = {}
    for cls, n, mod, key in linears(layer):
        mods.setdefault(key, mod)
    return mods


def calib_ids(tok):
    """128 x 2048 tokens of the FineWeb sample the agents had (agent image /opt/datasets/fineweb):
    documents in a seed-0 permutation, each followed by a blank line, packed back to back."""
    import pyarrow.parquet as pq
    texts = pq.read_table(FINEWEB, columns=["text"]).column("text").to_pylist()
    order = np.random.default_rng(SEED).permutation(len(texts))
    ids, used = [], 0
    for i in order:
        ids += tok(texts[i] + "\n\n", add_special_tokens=False)["input_ids"]
        used += 1
        if len(ids) >= CAL_N * CAL_LEN:
            break
    return torch.tensor(ids[:CAL_N * CAL_LEN]).view(CAL_N, CAL_LEN), used


class State:
    pass


S = State()


def setup(model):
    """BF16 originals to CPU, eval/calib tokens, BF16 NLL, Hessians (GPU for 5120/6144 inputs, CPU for 17408)."""
    from transformers import AutoTokenizer
    OUT.mkdir(parents=True, exist_ok=True)
    S.model = model
    S.lm = model.model.language_model
    S.dev = next(model.parameters()).device
    S.orig = {}
    for L, layer in enumerate(S.lm.layers):
        for cls, n, mod, key in linears(layer):
            S.orig[(L, n)] = mod.weight.data.to("cpu", copy=True)
    S.eval_ids = torch.tensor(json.load(open(PUBLIC / "ppl_inputs.json")))
    assert S.eval_ids.shape == (256, 2048)
    tok = AutoTokenizer.from_pretrained(str(PUBLIC / "tokenizer"))
    S.cal_ids, S.cal_docs = calib_ids(tok)
    t0 = time.time()
    S.nll_bf16 = eval_nll()
    S.t_eval = time.time() - t0
    np.save(OUT / "nll_bf16.npy", S.nll_bf16)
    print(f"BF16 ppl {math.exp(S.nll_bf16.mean()):.6f} over {S.nll_bf16.size} tokens in {S.t_eval:.0f}s", flush=True)
    t0 = time.time()
    collect_hessians()
    print(f"hessians in {time.time() - t0:.0f}s", flush=True)


@torch.no_grad()
def eval_nll(bs=4):
    out = []
    for a in range(0, S.eval_ids.shape[0], bs):
        ids = S.eval_ids[a:a + bs].to(S.dev)
        h = S.lm(input_ids=ids).last_hidden_state[:, PREFIX - 1:-1]
        logits = S.model.lm_head(h).float()
        lp = torch.log_softmax(logits, -1)
        out.append((-lp.gather(-1, ids[:, PREFIX:, None])[..., 0]).cpu())
        del h, logits, lp
    return torch.cat(out).numpy().astype(np.float64)


@torch.no_grad()
def collect_hessians(bs=8):
    """Layer by layer on BF16 activations of the calibration set: H_key = sum x x^T (float32)."""
    lm = S.lm
    S.H = {}
    hs = lm.embed_tokens(S.cal_ids.to(S.dev))          # [128, 2048, 5120] bf16
    pos = torch.arange(CAL_LEN, device=S.dev).view(1, 1, -1).expand(4, bs, -1)
    pe = lm.rotary_emb(hs[:bs], pos[1:])
    for L, layer in enumerate(lm.layers):
        acc, handles = {}, []
        for key, mod in hook_modules(layer).items():
            def pre(m, args, key=key):
                x = args[0].reshape(-1, args[0].shape[-1]).float()
                acc[key] = acc.get(key, 0) + x.t() @ x
            handles.append(mod.register_forward_pre_hook(pre))
        for a in range(0, hs.shape[0], bs):
            hs[a:a + bs] = layer(hs[a:a + bs], position_embeddings=pe, attention_mask=None, position_ids=pos[0])
        for hd in handles:
            hd.remove()
        for key, H in acc.items():
            S.H[(L, key)] = H.cpu() if key == "down" else H
        del acc
    del hs
    torch.cuda.empty_cache()


def restore():
    for L, layer in enumerate(S.lm.layers):
        for cls, n, mod, key in linears(layer):
            mod.weight.data.copy_(S.orig[(L, n)].to(S.dev))


# ----------------------------------------------------------------------------- variants
ATTN = ("attn", "linattn")


def alloc_set():
    """The 81 MLP matrices that move by one bit in the equal-byte allocation variants: gate_proj of all 64
    layers + up_proj of 17 evenly spaced layers (81 x 89.1 M = attention's 7.21 G weights, within 0.1%)."""
    ups = set(np.linspace(0, 63, 17).round().astype(int).tolist())
    return {(L, "gate_proj") for L in range(64)} | {(L, "up_proj") for L in ups}


def bits_for(spec, L, n, cls):
    b = spec["bits"]
    a = spec.get("alloc")
    if a is None:
        return b
    if cls in ATTN:
        return b + a
    return b - a if (L, n) in alloc_set() else b


VARIANTS = {
    # 3-bit, RTN: grid x clip
    "U3": dict(grid="U", bits=3),
    "U3c": dict(grid="U", bits=3, clip=True),
    "NF3": dict(grid="NF", bits=3),
    "NF3c": dict(grid="NF", bits=3, clip=True),
    "KM3c": dict(grid="KM", bits=3, clip=True),
    # rotation, rounding
    "U3c_rot": dict(grid="U", bits=3, clip=True, rot=True),
    "U3c_gptq": dict(grid="U", bits=3, clip=True, gptq=True),
    "U3_gptq": dict(grid="U", bits=3, gptq=True),
    "U3c_rot_gptq": dict(grid="U", bits=3, clip=True, rot=True, gptq=True),
    "NF3c_gptq": dict(grid="NF", bits=3, clip=True, gptq=True),
    # equal-byte allocation (RTN, U clip)
    "U3c_attn+1": dict(grid="U", bits=3, clip=True, alloc=+1),
    "U3c_attn-1": dict(grid="U", bits=3, clip=True, alloc=-1),
    # 2-bit and 4-bit references
    "U2c": dict(grid="U", bits=2, clip=True),
    "U2c_rot": dict(grid="U", bits=2, clip=True, rot=True),
    "U2c_gptq": dict(grid="U", bits=2, clip=True, gptq=True),
    "U2c_rot_gptq": dict(grid="U", bits=2, clip=True, rot=True, gptq=True),
    "U4c": dict(grid="U", bits=4, clip=True),
    # level shape vs zero point: symmetric uniform (no zero point); NF after rotation
    "M3c": dict(grid="M", bits=3, clip=True),
    "NF3c_rot": dict(grid="NF", bits=3, clip=True, rot=True),
}


@torch.no_grad()
def run_variant(name):
    spec = VARIANTS[name]
    path = OUT / f"{name}.json"
    t0 = time.time()
    per_cls = {}
    per_linear = []
    total_bits = 0
    for L, layer in enumerate(S.lm.layers):
        for cls, n, mod, key in linears(layer):
            W = S.orig[(L, n)].to(S.dev).float()
            H = S.H[(L, key)].to(S.dev)
            b = bits_for(spec, L, n, cls)
            Q, nb = quantize_matrix(W, H, spec, b)
            dW = W - Q
            w2, e2 = (W * W).sum().item(), (dW * dW).sum().item()
            o_num = ((dW @ H) * dW).sum().item()
            o_den = ((W @ H) * W).sum().item()
            mod.weight.data.copy_(Q.to(mod.weight.dtype))
            total_bits += nb
            c = per_cls.setdefault(cls, dict(w2=0.0, e2=0.0, o_num=0.0, o_den=0.0, bits=0, params=0))
            c["w2"] += w2; c["e2"] += e2; c["o_num"] += o_num; c["o_den"] += o_den; c["bits"] += nb; c["params"] += W.numel()
            per_linear.append([L, n, b, e2 / w2, o_num / o_den])
            del W, H, Q, dW
    t_q = time.time() - t0
    nll = eval_nll()
    restore()
    d = nll - S.nll_bf16
    blk = d.reshape(256, -1).mean(1)
    rng = np.random.default_rng(SEED)
    boot = [blk[rng.integers(0, 256, 256)].mean() for _ in range(2000)]
    agg = lambda keys: {k: sum(per_cls[c][k] for c in keys if c in per_cls) for k in ("w2", "e2", "o_num", "o_den", "bits", "params")}
    def summ(x):
        return dict(rel_mse=x["e2"] / x["w2"], out_err=x["o_num"] / x["o_den"], bits_per_weight=x["bits"] / x["params"],
                    bytes=x["bits"] / 8, params=x["params"])
    res = dict(
        variant=name, spec=spec,
        overall=summ(agg(per_cls)),
        attention=summ(agg(ATTN)), mlp=summ(agg(("mlp_gate", "mlp_up", "mlp_down"))),
        per_class={c: summ(v) for c, v in per_cls.items()},
        mean_linear_out_err=float(np.mean([x[4] for x in per_linear])),
        mean_linear_rel_mse=float(np.mean([x[3] for x in per_linear])),
        excess_ce=float(d.mean()), excess_ce_ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        ppl=float(math.exp(nll.mean())), bf16_ppl=float(math.exp(S.nll_bf16.mean())),
        scored_tokens=int(d.size), quantize_s=round(t_q, 1), eval_s=round(time.time() - t0 - t_q, 1),
        decoder_linear_bytes=total_bits / 8,
        per_linear=per_linear,
    )
    np.save(OUT / f"nll_{name}.npy", nll.astype(np.float32))
    json.dump(res, open(path, "w"), indent=1)
    print(f"{name}: bytes {total_bits/8/2**30:.3f} GiB rel_mse {res['overall']['rel_mse']:.5f} out_err {res['overall']['out_err']:.5f} "
          f"excess {res['excess_ce']:.4f} [{res['excess_ce_ci95'][0]:.4f},{res['excess_ce_ci95'][1]:.4f}] ({t_q:.0f}s q)", flush=True)
    return res
