"""Offline, CPU-only: relative weight MSE of the five agents' 3-bit routed-expert grids (RTN form, no data)
on the same real Nemotron-3.5-Super expert matrices (layers LAYERS, experts EXPERTS; fetch_sampled_tensors.py samples
the same). Read-only on the checkpoint: nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 at the 48 GiB task's
pinned revision, located by model_for_task.py. Writes grid_mse.json to common.QUANT_ERROR_DIR
(compression_bench/evolve_space/analysis/quant_error).

Each quantizer re-implements the agent's final expert grid from its submitted code:
  DS  : DeepSeek nemoq/format.py quantize_groupwise: signed symmetric, codes in [-3,3] (7 levels), g128,
        clip ratio linspace(0.55,1,7), fp16 scale (codes chosen with fp32 scale, dequant with fp16 scale)
  MM  : MiniMax build_package.py quantize_bitplane: 8 levels, zero=round(3.5)=4, range +-c*max|w|,
        up g256 c=0.70, down g128 c=0.76, uint8 scale relative to per-row fp16 super-scale
  OPu : Opus lineage v1 grid: symmetric midrise 8 levels (q-3.5), g128, alpha linspace(0.5,1,21), fp16 scale
  OPf : Opus final grid: midrise + cubic companding t+0.0244 t^3, log8 scales (2^-1/16 steps below a
        per-matrix fp32 base), alpha search; optionally with the latent Hadamard rotation
  SN1 : Sonnet v1: asymmetric uniform, integer zero, clip search (20 shrink factors 1..0.5), fp16 scale + uint8 zero
  SNf : Sonnet final: companded w=s*t*(1+0.025 t^2), t=code-zp/2 (half-step zp), Lloyd fit, scales re-encoded
        on a per-tensor uint8 log2 grid (16 octaves max)
  GLM : GLM nq/common.py quant_rtn_3_perrow: midrise 8 levels, ONE fp16 scale per output row, golden-section
        MSE scale; with and without the latent Hadamard rotation (GLM shipped it with rotation + GPTQ)
  KIMI: Kimi K3.1 work/compress_n3g_opt.py quant3 (its final v3): one 8-entry codebook per expert matrix, k-means
        (12 Lloyd iterations from quantiles) on 65,536 samples of w / group max, groups of 104 input columns, scale
        factor searched in {0.7..1.2} x group max, FP8 E4M3 scale under a per-matrix fp32 base, FP16 codebook; the
        agent drew the k-means sample unseeded on CUDA, here seeded (seed 0)
"""
import json
import math
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import QUANT_ERROR_DIR as OUT, model_dir  # noqa: E402

torch.set_num_threads(int(os.environ.get("NT", "16")))
TASK = "nemotron3_5_super_vl_120b_a12b_48g_standard_1xH200NVL"
SOURCE = None   # (checkpoint dir, weight map), found on first get(): importing this module (run_grids.py) needs no checkpoint
LAYERS = [1, 12, 45, 72, 87]
EXPERTS = [0, 64, 128, 192, 256, 320, 384, 448]


def get(name):
    global SOURCE
    if SOURCE is None:
        src = model_dir(TASK)
        SOURCE = src, json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]
    src, index = SOURCE
    with safe_open(f"{src}/{index[name]}", framework="pt", device="cpu") as f:
        return f.get_tensor(name).float()


def grp(W, G):
    n, k = W.shape[-2:]
    return W.reshape(*W.shape[:-1], k // G, G)


# ---------------------------------------------------------------- DeepSeek
def q_ds(W, G=128, bits=3, steps=7):
    w = grp(W, G)
    amax = w.abs().amax(-1, keepdim=True).clamp_min(1e-8)
    qmax = (1 << (bits - 1)) - 1
    best = bc = bs = None
    for r in torch.linspace(0.55, 1.0, steps):
        s = amax * r / qmax
        c = torch.clamp(torch.round(w / s), -qmax, qmax)
        e = ((c * s - w) ** 2).sum(-1, keepdim=True)
        if best is None:
            best, bc, bs = e, c, s
        else:
            m = e < best
            best = torch.where(m, e, best); bc = torch.where(m, c, bc); bs = torch.where(m, s, bs)
    return (bc * bs.half().float()).reshape(W.shape)


# ---------------------------------------------------------------- MiniMax
def q_mm(W, G, clip):
    n, k = W.shape
    w = grp(W, G)
    peak = w.abs().amax(-1, keepdim=True)
    lo, hi = -clip * peak, clip * peak
    s = ((hi - lo) / 7.0).clamp(min=1e-12)
    z = torch.round(-lo / s).clamp(0, 7)
    q = torch.clamp(torch.round(w / s) + z, 0, 7)
    fs = s.reshape(n, -1)
    row = fs.amax(1)
    s8 = torch.round(fs / row[:, None].clamp(min=1e-30) * 255.0).clamp(1, 255)
    seff = (s8 / 255.0 * row.half().float()[:, None]).reshape(s.shape)
    return ((q - z) * seff).reshape(W.shape)


# ---------------------------------------------------------------- Opus
ALPH3 = torch.linspace(0.50, 1.0, 21).tolist()


def op_vals(q, g):
    t = q - 3.5
    return t + g * t ** 3 if g else t


def op_nearest(u, g):
    if not g:
        return torch.clamp(torch.round(u + 3.5), 0, 7)
    lv = op_vals(torch.arange(8, dtype=torch.float32), g)
    mids = (lv[1:] + lv[:-1]) / 2
    return torch.bucketize(u.contiguous(), mids).float()


def q_op(W, g=0.0, log8=False, G=128):
    w = grp(W, G)
    lmax = 3.5 + g * 3.5 ** 3
    amax = w.abs().amax(-1, keepdim=True).clamp(min=1e-8)
    base = W.abs().amax() / lmax if log8 else None
    best = bs = None
    for a in ALPH3:
        s = amax * (a / lmax)
        if log8:
            code = torch.clamp(torch.round(torch.log2(base / s.clamp(min=1e-30)) * 16), 0, 255)
            s = torch.exp2(-code / 16.0) * base
        else:
            s = s.half().float().clamp(min=1e-7)
        q = op_nearest(w / s, g)
        e = ((op_vals(q, g) * s - w) ** 2).sum(-1, keepdim=True)
        if best is None:
            best, bs = e, s
        else:
            m = e < best
            best = torch.where(m, e, best); bs = torch.where(m, s, bs)
    q = op_nearest(w / bs, g)
    return (op_vals(q, g) * bs).reshape(W.shape)


# ---------------------------------------------------------------- Sonnet
def q_sn1(W, G=128, n_grid=20, shrink_min=0.5, bits=3):
    qmax = (1 << bits) - 1
    wg = grp(W, G)
    wmin = wg.amin(-1, keepdim=True).clamp(max=0)
    wmax = wg.amax(-1, keepdim=True).clamp(min=0)

    def quant(a, b):
        sc = ((b - a) / qmax).clamp(min=1e-8).half().float().clamp(min=1e-8)
        z = torch.round(-a / sc).clamp(0, qmax)
        q = torch.clamp(torch.round(wg / sc) + z, 0, qmax)
        return (q - z) * sc

    deq = quant(wmin, wmax)
    best = ((deq - wg) ** 2).sum(-1, keepdim=True)
    for i in range(1, n_grid):
        f = 1.0 - (1.0 - shrink_min) * i / n_grid
        d2 = quant(wmin * f, wmax * f)
        e = ((d2 - wg) ** 2).sum(-1, keepdim=True)
        m = e < best
        best = torch.where(m, e, best); deq = torch.where(m, d2, deq)
    return deq.reshape(W.shape)


def _levels(s, zp, beta, bits=3):
    t = torch.arange(1 << bits, dtype=torch.float32)[None, :] - zp[:, None] * 0.5
    return s[:, None] * (t + beta * t ** 3), t


def _lloyd(wg, beta, init_frac, iters=3, bits=3):
    n = wg.shape[0]
    qmax = (1 << bits) - 1
    rng = (wg.amax(1) - wg.amin(1)).clamp(min=1e-12)
    t0 = torch.arange(1 << bits, dtype=torch.float32) - qmax / 2
    span = t0 + beta * t0 ** 3
    s = rng * init_frac / (span.max() - span.min())
    zp = torch.full((n,), float(qmax))
    best_err = torch.full((n,), float("inf"))
    best = (s.clone(), zp.clone())
    for _ in range(iters):
        cand = []
        for dz in (-1.0, 0.0, 1.0):
            zc = (zp + dz).clamp(0, 15)
            lev, t = _levels(s, zc, beta)
            code = (wg[:, :, None] > ((lev[:, 1:] + lev[:, :-1]) / 2)[:, None, :]).sum(-1)
            gg = (t + beta * t ** 3).gather(1, code)
            s_new = ((wg * gg).sum(1) / (gg * gg).sum(1).clamp(min=1e-20)).clamp(min=1e-12)
            lev2, _ = _levels(s_new, zc, beta)
            err = ((lev2.gather(1, code) - wg) ** 2).sum(1)
            cand.append((err, zc, s_new))
        errs = torch.stack([c[0] for c in cand]); k = errs.argmin(0)[None]
        e_best = errs.gather(0, k)[0]
        zp = torch.stack([c[1] for c in cand]).gather(0, k)[0]
        s = torch.stack([c[2] for c in cand]).gather(0, k)[0]
        better = e_best < best_err
        best_err = torch.where(better, e_best, best_err)
        best = (torch.where(better, s, best[0]), torch.where(better, zp, best[1]))
    return best_err, best[0], best[1]


def q_snf(W, G=128, beta=0.025, chunk=1 << 17):
    """W [E, N, K] stack (one stored class tensor) -> dequantized."""
    wg_all = W.reshape(-1, G)
    S_ = torch.empty(wg_all.shape[0]); Z_ = torch.empty(wg_all.shape[0])
    for a in range(0, wg_all.shape[0], chunk):
        wg = wg_all[a:a + chunk]
        res = [_lloyd(wg, beta, f) for f in (0.8, 0.5)]
        errs = torch.stack([r[0] for r in res]); k = errs.argmin(0)[None]
        S_[a:a + chunk] = torch.stack([r[1] for r in res]).gather(0, k)[0]
        Z_[a:a + chunk] = torch.stack([r[2] for r in res]).gather(0, k)[0]
    # codes chosen with the fitted scales, then scales re-encoded on the per-tensor uint8 log2 grid
    out = torch.empty_like(wg_all)
    ls = torch.log2(S_.clamp(min=1e-12)); lmax = ls.max(); lmin = torch.maximum(ls.min(), lmax - 16.0)
    step = ((lmax - lmin) / 255.0).clamp(min=1e-6)
    sdec = torch.exp2(lmin + torch.round((ls - lmin) / step).clamp(0, 255) * step)
    for a in range(0, wg_all.shape[0], chunk):
        wg = wg_all[a:a + chunk]
        lev, t = _levels(S_[a:a + chunk], Z_[a:a + chunk], beta)
        code = (wg[:, :, None] > ((lev[:, 1:] + lev[:, :-1]) / 2)[:, None, :]).sum(-1)
        tt = t.gather(1, code)
        out[a:a + chunk] = sdec[a:a + chunk, None] * tt * (1 + beta * tt * tt)
    return out.reshape(W.shape)


# ---------------------------------------------------------------- GLM
def q_glm(W, iters=20):
    amax = W.abs().amax(-1).clamp_min(1e-12)
    lo, hi = amax / 7.0, amax / 3.0

    def mse(s):
        q = torch.round(W / s.unsqueeze(-1) + 3.5).clamp_(0, 7) - 3.5
        return ((q * s.unsqueeze(-1) - W) ** 2).sum(-1)
    for _ in range(iters):
        m1 = lo + 0.382 * (hi - lo); m2 = lo + 0.618 * (hi - lo)
        e1, e2 = mse(m1), mse(m2)
        lo = torch.where(e1 < e2, lo, m1); hi = torch.where(e1 < e2, m2, hi)
    s = ((lo + hi) / 2).half().float()
    q = torch.round(W / s.unsqueeze(-1) + 3.5).clamp(0, 7) - 3.5
    return q * s.unsqueeze(-1)


FACTORS = (.7, .8, .9, 1.0, 1.1, 1.2)
PROBS = (.005, .05, .2, .4, .6, .8, .95, .995)


def _lloyd_1d(sample, cb, iters):
    for _ in range(iters):
        assignment = (sample[:, None] - cb).abs().argmin(dim=1)
        updated = cb.clone()
        for j in range(cb.numel()):
            mask = assignment == j
            if mask.any():
                updated[j] = sample[mask].mean()
        cb = updated
    return cb


def q_kimi(W, group=104, seed=0, n_sample=65536, iters=12):
    n, k = W.shape
    groups = (k + group - 1) // group
    padded = groups * group
    w = torch.nn.functional.pad(W.float(), (0, padded - k)).reshape(n, groups, group)
    magnitude = w.abs().amax(dim=2)
    flat = (w / magnitude.clamp_min(1e-30)[:, :, None]).reshape(-1)
    sample = flat[torch.randint(0, flat.numel(), (n_sample,), generator=torch.Generator().manual_seed(seed))]
    cb = _lloyd_1d(sample, torch.quantile(sample, torch.tensor(PROBS)), iters)
    best_error = torch.full_like(magnitude, float("inf"))
    best_factor = torch.ones_like(magnitude)
    for factor in FACTORS:                    # per group, the scale factor with the lowest MSE (first wins ties)
        raw = magnitude * factor
        codes = ((w / raw.clamp_min(1e-30)[:, :, None])[:, :, :, None] - cb).abs().argmin(dim=3)
        error = (cb[codes] * raw[:, :, None] - w).square().mean(dim=2)
        take = error < best_error
        best_error = torch.where(take, error, best_error)
        best_factor = torch.where(take, torch.full_like(magnitude, factor), best_factor)
    raw = magnitude * best_factor
    base = (raw.max() / 256).clamp_min(1e-30)
    scale = (raw / base).to(torch.float8_e4m3fn).float()
    codebook = (cb * base).half().float()
    sf = scale * base
    norm = torch.where((sf > 0)[:, :, None], w / sf.clamp_min(1e-30)[:, :, None], torch.zeros_like(w))
    codes = (norm[:, :, :, None] - codebook / base).abs().argmin(dim=3)   # against the stored values
    return (codebook[codes] * scale[:, :, None]).reshape(n, padded)[:, :k].contiguous()


def hadamard(n, seed):
    Hm = torch.ones(1, 1, dtype=torch.float64)
    while Hm.shape[0] < n:
        Hm = torch.cat([torch.cat([Hm, Hm], 1), torch.cat([Hm, -Hm], 1)], 0)
    gen = torch.Generator().manual_seed(seed)
    s = torch.randint(0, 2, (n,), generator=gen).double() * 2 - 1
    return ((Hm * s[None, :]) / math.sqrt(n)).float()


R = hadamard(1024, 7)


def main():
    rows = []
    tot = {}
    for L in LAYERS:
        for proj in ("up", "down"):
            Ws = torch.stack([get(f"language_model.backbone.layers.{L}.mixer.experts.{e}.{proj}_proj.weight") for e in EXPERTS])
            Wr = Ws @ R if proj == "up" else R.t() @ Ws          # latent-space rotation (orthogonal: MSE comparable)
            den = (Ws ** 2).sum().item()
            r = {}
            r["DS"] = sum(((q_ds(w) - w) ** 2).sum().item() for w in Ws) / den
            r["MM"] = sum(((q_mm(w, 256 if proj == "up" else 128, 0.70 if proj == "up" else 0.76) - w) ** 2).sum().item() for w in Ws) / den
            r["OPu"] = sum(((q_op(w) - w) ** 2).sum().item() for w in Ws) / den
            r["OPf"] = sum(((q_op(w, 0.0244, True) - w) ** 2).sum().item() for w in Ws) / den
            r["OPf_rot"] = sum(((q_op(w, 0.0244, True) - w) ** 2).sum().item() for w in Wr) / den
            r["SN1"] = sum(((q_sn1(w) - w) ** 2).sum().item() for w in Ws) / den
            r["SNf"] = ((q_snf(Ws) - Ws) ** 2).sum().item() / den
            r["GLM_norot"] = sum(((q_glm(w) - w) ** 2).sum().item() for w in Ws) / den
            r["GLM_rot"] = sum(((q_glm(w) - w) ** 2).sum().item() for w in Wr) / den
            r["KIMI"] = sum(((q_kimi(w) - w) ** 2).sum().item() for w in Ws) / den
            for k, v in r.items():
                tot.setdefault((proj, k), []).append(v)
            print(L, proj, " ".join(f"{k}={v:.5f}" for k, v in r.items()), flush=True)
            rows.append({"layer": L, "proj": proj, **r})
    print("MEAN over layers")
    for proj in ("up", "down"):
        print(proj, " ".join(f"{k}={sum(v)/len(v):.5f}" for (p, k), v in tot.items() if p == proj))
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(OUT / "grid_mse.json", "w"), indent=1)


if __name__ == "__main__":
    main()
