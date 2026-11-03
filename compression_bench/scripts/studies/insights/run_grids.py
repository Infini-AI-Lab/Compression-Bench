"""RTN forms of each analysed run's FINAL main-class grid (routed experts; dense MLP for Qwen3.8-27B), for
grid_mse_all.py. Each entry re-implements the stored format from the run's own code (source paths in
`source`, relative to the run directory, or `lineage vN: files/...` inside that version of its lineage):
bits, grouping, level set, scale/zero precision, clip search, data-free rotations. Data-dependent steps
(GPTQ/LDLQ error feedback, calibration-chosen bits or experts, fine-tuning, learned codebooks trained on
calibration activations) are dropped and listed in `omitted`; a per-layer static rule is kept. Rotations are
re-drawn with a fixed seed (the MSE of an orthogonal rotation does not depend on the particular signs).
fn(W [out, in] float32, proj, layer) -> What in the original basis.

Grids are keyed `<agent>/<model>-<cap>` (e.g. `glm/qwen8-12`). The run behind a key -- whose final
quantization_config and work files some grids read -- is looked up in the version ledger
(evaluation_progress/trajectory_analysis/version_ledger.json, compression_bench/scripts/version_ledger.py) by agent,
model and capacity.
The six Nemotron-48 grids are grid_mse.py's (q_ds, q_mm, q_op, q_snf, q_glm, q_kimi), registered here too.
"""
import math

import torch

import grid_mse as g48


# ----------------------------------------------------------------------------- primitives
def had(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


def signs(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (n,), generator=g).float() * 2 - 1


def rot_in(W, block, seed=0, inverse=False):
    """W diag(s) blockdiag(H_block) on the input dim (inverse: W R^T)."""
    n, k = W.shape
    H, s = had(block), signs(k, seed)
    if not inverse:
        return ((W * s).reshape(n, k // block, block) @ H).reshape(n, k)
    return (W.reshape(n, k // block, block) @ H.t()).reshape(n, k) * s


def rot_out(W, block, seed=0, inverse=False):
    """blockdiag(H) diag(s) W on the output dim."""
    return rot_in(W.t().contiguous(), block, seed, inverse).t().contiguous()


def with_rot(q, block_in=None, block_out=None, seed=0):
    def f(W, *a):
        X = W
        if block_in:
            X = rot_in(X, block_in, seed)
        if block_out:
            X = rot_out(X, block_out, seed + 1)
        Q = q(X, *a)
        if block_out:
            Q = rot_out(Q, block_out, seed + 1, inverse=True)
        if block_in:
            Q = rot_in(Q, block_in, seed, inverse=True)
        return Q
    return f


def f16(x):
    return x.half().float()


def grp(W, G):
    n, k = W.shape
    G = k if G is None else G
    pad = (-k) % G
    if pad:
        W = torch.nn.functional.pad(W, (0, pad))
    return W.reshape(n, -1, G), k


def ungrp(Wg, k):
    return Wg.reshape(Wg.shape[0], -1)[:, :k]


def best_of(wg, cands, scope="group"):
    """cands: iterable of dequantized candidates (same shape as wg [n, ng, G]); pick per group (or per matrix)."""
    best = be = None
    for c in cands:
        e = ((c - wg) ** 2).sum(-1, keepdim=True)
        if scope == "matrix":
            e = e.sum().expand_as(e)
        if best is None:
            best, be = c, e
        else:
            m = e < be
            be = torch.where(m, e, be)
            best = torch.where(m, c, best)
    return best


def levels_q(wg, s, lv):
    """nearest level of the (sorted) level set lv for w/s, times s."""
    mid = ((lv[1:] + lv[:-1]) / 2).contiguous()
    return s * lv[torch.bucketize((wg / s).contiguous(), mid)]


def sym_levels(wg, lv, clips, scale_fmt=f16, scope="group"):
    """s = clip * amax / max|lv| (scale stored by scale_fmt), nearest level."""
    amax = wg.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    lmax = lv.abs().max()
    return best_of(wg, (levels_q(wg, scale_fmt(amax * c / lmax).clamp_min(1e-12), lv) for c in clips), scope)


def midrise(bits):
    m = (1 << bits) - 1
    return (2 * torch.arange(1 << bits).float() - m) / 2          # +-0.5, +-1.5, ...


def intsym(qmax):
    return torch.arange(-qmax, qmax + 1).float()


def asym_minmax(wg, bits, clips, zero="int", scale_fmt=f16, scope="group", lo_hi=None):
    """uniform asymmetric: s = (hi-lo)/(2^b-1) with lo, hi = clip*(min, max); zero int or fp16."""
    L = (1 << bits) - 1
    lo0, hi0 = wg.amin(-1, keepdim=True), wg.amax(-1, keepdim=True)

    def one(c):
        lo, hi = lo0 * c, hi0 * c
        s = scale_fmt(((hi - lo) / L).clamp_min(2.0 ** -14))
        if zero == "int":
            z = torch.round(-lo / s).clamp(0, L)
            return s * (torch.clamp(torch.round(wg / s) + z, 0, L) - z)
        z = f16(-lo / s)
        return s * (torch.clamp(torch.round(wg / s + z), 0, L) - z)
    return best_of(wg, (one(c) for c in clips), scope)


def grid_fn(kind, bits, G, clips=(1.0,), scope="group", scale_fmt=f16, zero="int", lv=None):
    """A row-wise grouped scalar grid as fn(W, proj, layer)."""
    def f(W, proj=None, layer=None):
        wg, k = grp(W, G)
        if kind == "asym":
            Q = asym_minmax(wg, bits, clips, zero, scale_fmt, scope)
        else:
            levels = {"midrise": midrise(bits), "levels": lv}.get(kind)
            if kind == "intsym":
                levels = intsym((1 << (bits - 1)) - 1)
            Q = sym_levels(wg, levels, clips, scale_fmt, scope)
        return ungrp(Q, k)
    return f


def per_proj(**fns):
    """dispatch on proj ('up','gate','down'); 'gate' falls back to 'up'."""
    def f(W, proj, layer):
        fn = fns.get(proj) or fns.get("up")
        return fn(W, proj, layer)
    return f


def per_layer(rule, fns):
    """rule(proj, layer) -> key into fns (a static per-layer bit rule)."""
    def f(W, proj, layer):
        return fns[rule(proj, layer)](W, proj, layer)
    return f


def lin(a, b, n):
    return torch.linspace(a, b, n).tolist()


GRIDS = {}


def reg(agent, model, task, fn, bits, bpw, desc, omitted, source, coverage="all experts"):
    GRIDS[f"{agent}/{task}"] = dict(agent=agent, model=model, task=task, fn=fn, bits=bits, bpw_stored=bpw, desc=desc,
                                    omitted=omitted, coverage=coverage, source=source)


# ----------------------------------------------------------------------------- Nemotron-48 (grid_mse.py)
def _stackwise(q):
    return lambda W, proj, layer: q(W[None])[0]


reg("deepseek", "nemotron", "nemotron-48", lambda W, p, L: g48.q_ds(W), 3, None,
    "int sym [-3,3] g128, clip 0.55..1 (7), fp16 scale", ["layers 63,72 at 4 bit not modelled"], ["grid_mse.py:q_ds"])
reg("minimax", "nemotron", "nemotron-48",
    lambda W, p, L: g48.q_mm(W, 256 if p == "up" else 128, 0.70 if p == "up" else 0.76), 3, None,
    "bit-plane asym 8 levels, up g256 c0.70 / down g128 c0.76, uint8 scale under fp16 row scale", [], ["grid_mse.py:q_mm"])
reg("opus", "nemotron", "nemotron-48", lambda W, p, L: g48.q_op(W, 0.0244, True), 3, None,
    "midrise + cubic companding, log8 scales, alpha search", ["4-bit upgrades (calibration-chosen)"], ["grid_mse.py:q_op"])
reg("sonnet", "nemotron", "nemotron-48", _stackwise(g48.q_snf), 3, None,
    "companded half-step-zp Lloyd fit, uint8 log2 scales", [], ["grid_mse.py:q_snf"])
reg("glm", "nemotron", "nemotron-48",
    lambda W, p, L: (g48.R.t() @ g48.q_glm(g48.R @ W) if p == "down" else g48.q_glm(W @ g48.R) @ g48.R.t()), 3, None,
    "midrise 8 levels, one fp16 scale per row, golden-section MSE scale, latent Hadamard", ["GPTQ"], ["grid_mse.py:q_glm"])
reg("kimi", "nemotron", "nemotron-48", lambda W, p, L: g48.q_kimi(W), 3, None,
    "per-matrix 8-entry k-means codebook, g104, FP8 scale", [], ["grid_mse.py:q_kimi"])


# ----------------------------------------------------------------------------- the run behind a grid
import json as _json
from pathlib import Path as _Path
import sys as _sys

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from common import QUANT_ERROR_DIR as SCRATCH, REPO as _REPO, TRAJECTORY_ANALYSIS, run_dir  # noqa: E402

LEDGER = TRAJECTORY_ANALYSIS / "version_ledger.json"
_LEDGER = None


def run_record(key):
    """The version-ledger entry of the run behind grid `key` (`<agent>/<model>-<cap>`)."""
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = _json.load(open(LEDGER))["runs"]
    agent, task = key.split("/")
    hits = [r for r in _LEDGER if r["agent"] == agent and f"{r['model']}-{r['cap']}" == task]
    if len(hits) != 1:
        raise KeyError(f"{key}: {len(hits)} runs in {LEDGER} (expected one)")
    return hits[0]


def run_file(key, rel):
    """A file of the run behind grid `key`, relative to its run directory (work/... as the agent left it)."""
    run = run_record(key)["run"]
    return run_dir(run) / rel


def qcfg(key):
    """The final version's model/config.json quantization_config of the run behind grid `key`."""
    r = run_record(key)
    v = r["versions"][-1]["version"]
    c = _json.load(open(_REPO / f"compression_output/{r['lineage']}/{r['lineage']}_{v}/files/model/config.json"))
    return c.get("quantization_config") or c.get("text_config", {}).get("quantization_config")


def fp8(x):
    return x.to(torch.float8_e4m3fn).float()


# ----------------------------------------------------------------------------- GLM
_r = "glm/nemotron-32"
reg("glm", "nemotron", "nemotron-32",
    per_proj(up=with_rot(grid_fn("midrise", 2, None, lin(1.0, 0.4, 25), scope="matrix"), block_in=1024),
             down=with_rot(grid_fn("midrise", 2, None, lin(1.0, 0.4, 25), scope="matrix"), block_out=1024)),
    2, None, "midrise 4 levels (2c-3)/3, one fp16 scale per row, one clip per matrix (1.0..0.4, 25), latent Hadamard (up inputs, down outputs)",
    ["GPTQ refine (nq/refine.py)"], ["work/nq/nqcore.py:74-110",
                                     "work/nq/build_package.py:219-240"])

_r = "glm/qwen122-32"
reg("glm", "qwen122", "qwen122-32", grid_fn("asym", 2, 256, lin(0.55, 1.0, 19), zero="fp16"),
    2, None, "asymmetric 4 levels, g256, fp16 scale + fp16 zero, per-group clip linspace(0.55,1,19)",
    ["expert pruning (moe_keep, calibration-chosen)"],
    ["lineage v4: files/sqmoe/pack.py:46-160"],
    coverage="kept experts only (moe_keep per layer)")


def _glm_rowlut(W, group, iters=4):
    """qfmt/lut.py fit_row_lut: per-row 4-level additive grid {C, C+B, C+A, C+A+B} (Lloyd), per-group LS scale."""
    N, K = W.shape
    Wg = W.reshape(N, K // group, group)
    A = torch.full((N,), 1.963 / 1.51); B = torch.full((N,), 1.057 / 1.51); C = torch.full((N,), -1.0)
    lev = lambda a, b, c: torch.stack([c, c + b, c + a, c + a + b], 1)
    s = Wg.pow(2).mean(-1).sqrt().clamp_min(1e-12)

    def assign(lv, s):
        best = bc = None
        for j in range(4):
            e = (Wg - s[:, :, None] * lv[:, j, None, None]) ** 2
            if best is None:
                best, bc = e, torch.zeros_like(e, dtype=torch.long)
            else:
                m = e < best; best = torch.where(m, e, best); bc = torch.where(m, torch.full_like(bc, j), bc)
        return bc
    for _ in range(iters + 1):
        lv = lev(A, B, C)
        code = assign(lv, s)
        L_ = torch.gather(lv[:, None, :].expand(N, K // group, 4), 2, code)
        s = (Wg * L_).sum(-1) / L_.pow(2).sum(-1).clamp_min(1e-12)
        if _ == iters:
            break
        t = (Wg / s[:, :, None]).reshape(N, K); cf = code.reshape(N, K)
        X = torch.stack([(cf >> 1).float(), (cf & 1).float(), torch.ones(N, K)], 2)
        abc = torch.linalg.solve(torch.einsum("nki,nkj->nij", X, X) + 1e-5 * torch.eye(3)[None], torch.einsum("nki,nk->ni", X, t))
        n2 = lev(abc[:, 0], abc[:, 1], abc[:, 2]).abs().amax(1).clamp_min(1e-12)
        abc = abc / n2[:, None]; s = s * n2[:, None]
        A, B, C = abc[:, 0], abc[:, 1], abc[:, 2]
    s = f16(s)
    return (s[:, :, None] * torch.gather(lv[:, None, :].expand(N, K // group, 4), 2, code)).reshape(N, K)


_r = "glm/qwen8-8"
reg("glm", "qwen8", "qwen8-8",
    per_proj(up=with_rot(lambda W, p, L: _glm_rowlut(W, 256), block_in=1024),
             down=with_rot(lambda W, p, L: _glm_rowlut(W, 128), block_in=1024)),
    2, None, "2-bit per-row 4-level additive LUT (Lloyd, 4 iters), per-group LS fp16 scale (gate/up g256, down g128), Hadamard-1024 on inputs",
    ["GPTQ (gptq_lut)", "H-weighted rank-64 residual on down_proj"],
    ["work/qfmt/lut.py:13-87", "work/qfmt/config.py:28-60"])

_r = "glm/qwen122-48"


def _glm_tq48(W, proj, layer, _c={}):
    if not _c:
        q = qcfg("glm/qwen122-48"); _c["w13"], _c["w2"] = q["w13_exp"], q["w2_exp"]
    G, e = (64, _c["w13"][layer]) if proj in ("gate", "up") else (128, _c["w2"][layer])
    wg, k = grp(W, G)
    s = fp8((0.78 * wg.abs().amax(-1, keepdim=True) / 3.5 / 2.0 ** e).clamp_min(2 ** -9)) * 2.0 ** e
    return ungrp(levels_q(wg, s, midrise(3)), k)


reg("glm", "qwen122", "qwen122-48", _glm_tq48, 3, None,
    "midrise 8 levels (c-3.5)s, s = fp8-e4m3(0.78 amax/3.5 / 2^e) 2^e (per-layer exponent), gate/up g64, down g128",
    ["GPTQ (act-order variant gptq2.py not in final)"],
    ["work/tq_pack.py:27-53", "work/gptq.py:1-12,65"])

_r = "glm/qwen8-12"


def _glm_qq(W, proj, layer, _c={}):
    if not _c:
        _c.update(qcfg("glm/qwen8-12")["tensors"])
    t = _c[f"model.layers.{layer}.mlp.{'down_proj' if proj == 'down' else 'gate_up_proj'}"]
    qmax = (1 << (t["bits"] - 1)) - 1
    wg, k = grp(W, t["group"])
    amax = wg.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    cands = []
    for p in (1.0, 0.95, 0.9, 0.85, 0.8):
        s = (amax * p / qmax).bfloat16().float()
        cands.append(s * torch.clamp(torch.round(wg / s), -qmax - 1, qmax))
    return ungrp(best_of(wg, cands), k)


reg("glm", "qwen8", "qwen8-12", _glm_qq, 3, None,
    "int sym full range [-2^(b-1), 2^(b-1)-1], g64 (some g256), bf16 scale, shrink {1..0.8}; bits per tensor as stored (3 or 4)",
    ["GPTQ (qqcore.gptq)"], ["work/scripts/qqcore.py:14-48"],
    coverage="per-tensor bits from the final config")


def asym_mid_fn(bits, G, fracs, zero="int", scope="group", scale_fmt=f16):
    """asymmetric uniform on [mid - f half, mid + f half] (MiniMax's clip family)."""
    L = (1 << bits) - 1

    def f(W, proj=None, layer=None):
        wg, k = grp(W, G)
        lo0, hi0 = wg.amin(-1, keepdim=True), wg.amax(-1, keepdim=True)
        mid, half = (lo0 + hi0) / 2, (hi0 - lo0) / 2

        def one(fr):
            lo, hi = mid - fr * half, mid + fr * half
            s = scale_fmt(((hi - lo) / L).clamp_min(1e-12)).clamp_min(2.0 ** -24)   # all-zero groups exist (qwen122)
            z = torch.round(-lo / s).clamp(0, L) if zero == "int" else -lo / s
            return s * (torch.clamp(torch.round(wg / s + z), 0, L) - z)
        return ungrp(best_of(wg, (one(x) for x in fracs), scope), k)
    return f


# ----------------------------------------------------------------------------- MiniMax
_r = "minimax/nemotron-32"
reg("minimax", "nemotron", "nemotron-32", asym_mid_fn(2, 128, (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)),
    2, None, "asymmetric 4 levels on mid +- f half, f in {1..0.3} (8) per group, integer zero, g128 up and down",
    ["activation-weighted clip error (AWQ statistics)", "expert width pruning 2688->2432 (importance pass)"],
    ["lineage v10: files/nemo_compact/kernels.py:22-202",
     "work/build_final/build_package.py:225-243"],
    coverage="kept intermediate channels (2432/2688)")
_r = "minimax/qwen122-32"
reg("minimax", "qwen122", "qwen122-32", grid_fn("asym", 3, 128), 3, None,
    "asymmetric min-max 8 levels (no clip), integer zero, fp16 scale, g128", ["GPTQ", "REAP-style pruning to 7920/12288 experts"],
    ["work/src/build_gptq.py:169-185"], coverage="kept experts only")
_r = "minimax/qwen8-8"
reg("minimax", "qwen8", "qwen8-8",
    with_rot(asym_mid_fn(3, 128, (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7), zero="float", scope="matrix"), block_in=1024),
    3, None, "asymmetric 8 levels on mid +- r half (one r per matrix from {1..0.7}), float zero, g128, Hadamard-1024 on inputs",
    ["GPTQ"], ["lineage v1: files/qlm/format.py:227-256",
               "work/build/gptq.py:40-80"])
_r = "minimax/qwen122-48"
reg("minimax", "qwen122", "qwen122-48", asym_mid_fn(3, 256, (1.0, 0.9, 0.8, 0.7, 0.6)), 3, None,
    "asymmetric 8 levels on mid +- f half, f in {1,.9,.8,.7,.6} per group, integer zero, fp16 scale, g256", ["GPTQ"],
    ["work/build/build_submission.py:36-102"])
_r = "minimax/qwen8-12"
reg("minimax", "qwen8", "qwen8-12", per_proj(up=grid_fn("asym", 3, 128), down=grid_fn("asym", 4, 128)), 3, None,
    "asymmetric min-max incl. 0, integer zero, g128; gate/up 3-bit, down 4-bit", ["GPTQ"],
    ["work/scripts/common.py:161-172"])


# ----------------------------------------------------------------------------- DeepSeek
def _ds_sym2(W, G):
    """nvquant/fmt.py quantize_sym2: levels {-3,-1,1,3} s, s = amax/c over _SYM2_GRID, then 2 LS refits, fp16."""
    wg, k = grp(W, G)
    amax = wg.abs().amax(-1, keepdim=True)
    qf = lambda s: torch.round(wg / (2 * s) + 1.5).clamp(0, 3) * 2 - 3
    best = be = None
    for c in (4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 9.0, 11.0, 14.0):
        s = (amax / c).clamp_min(1e-8)
        e = ((wg - qf(s) * s) ** 2).sum(-1, keepdim=True)
        if best is None:
            best, be = s, e
        else:
            m = e < be; best = torch.where(m, s, best); be = torch.where(m, e, be)
    s = best
    for _ in range(2):
        lv = qf(s)
        s = ((wg * lv).sum(-1, keepdim=True) / (lv * lv).sum(-1, keepdim=True).clamp_min(1e-8)).clamp_min(1e-8)
    s = f16(s)
    return ungrp(qf(s) * s, k)


def _hph(W, block=128, rounds=2, seed=0, inverse=False):
    """two rounds of (block-128 Hadamard, fixed permutation) on the input dim: an orthogonal 'H.P.H' mix."""
    n, k = W.shape
    perm = torch.randperm(k, generator=torch.Generator().manual_seed(seed))
    inv = torch.argsort(perm)
    X = W
    if not inverse:
        for r in range(rounds):
            X = rot_in(X, block, seed + r)
            if r + 1 < rounds:
                X = X[:, perm]
    else:
        for r in reversed(range(rounds)):
            if r + 1 < rounds:
                X = X[:, inv]
            X = rot_in(X, block, seed + r, inverse=True)
    return X


_r = "deepseek/nemotron-32"
reg("deepseek", "nemotron", "nemotron-32",
    per_proj(up=with_rot(lambda W, p, L: _ds_sym2(W, 512), block_in=1024),
             down=lambda W, p, L: _hph(_ds_sym2(_hph(W), 896), inverse=True)),
    2, None, "sym2 levels {-3,-1,1,3}s, s from amax/c grid + 2 LS refits, fp16; up g512 + Hadamard-1024 on inputs, down g896 + 2-round block-128 H.P.H",
    ["GPTQ (impl/gptq.py gptq_sym2)"], ["work/impl/code_v2r/nvquant/fmt.py:68-105"])


def _ds_midrise2_ls(W, p=None, L=None):
    """gptq_lib.py _ls_scale + rtn_quantize: (q-1.5)s, s init 0.5 amax/1.5, 2 LS refits, fp16, g128."""
    wg, k = grp(W, 128)
    amax = wg.abs().amax(-1, keepdim=True)
    s = amax / 1.5 * 0.5
    for _ in range(2):
        v = torch.clamp(torch.round(wg / s + 1.5), 0, 3) - 1.5
        s = (wg * v).sum(-1, keepdim=True) / (v * v).sum(-1, keepdim=True).clamp_min(1e-20)
    s = torch.where(torch.isfinite(s) & (s > 0), s, amax.clamp_min(1e-6) / 1.5)
    s = f16(s).clamp_min(6e-8)
    return ungrp((torch.clamp(torch.round(wg / s + 1.5), 0, 3) - 1.5) * s, k)


_r = "deepseek/qwen122-32"
reg("deepseek", "qwen122", "qwen122-32", with_rot(_ds_midrise2_ls, block_in=1024), 2, None,
    "midrise 4 levels (q-1.5)s, LS-fitted fp16 scale (2 iterations), g128, Hadamard-1024 on inputs (gate_up 3072, down 1024)",
    ["GPTQ"], ["work/gptq/gptq_lib.py:4-83"])


def _ds_rvq(W, ncb, seed=0, G=8, CB=256, SAMPLE=262144, iters=10):
    """dev/quantize.py quantize_matrix: per-row RMS scale, ncb residual k-means codebooks of 256 8-d vectors, 1 refit round."""
    N, K = W.shape
    s = ((W * W).sum(1).sqrt() / math.sqrt(K)).clamp_min(1e-12)
    flat = (W / s[:, None]).reshape(-1, G)

    def nearest(v, c):
        return (2 * v @ c.T - (c * c).sum(1)[None]).argmax(1)

    def kmeans(v, sd):
        g = torch.Generator().manual_seed(sd)
        vs = v[torch.randperm(v.shape[0], generator=g)[:SAMPLE]] if v.shape[0] > SAMPLE else v
        c = vs[torch.randperm(vs.shape[0], generator=g)[:CB]].clone()
        for _ in range(iters):
            a = nearest(vs, c)
            sm = torch.zeros_like(c).index_add_(0, a, vs)
            cnt = torch.zeros(CB).index_add_(0, a, torch.ones(a.shape[0]))
            emp = cnt < 1
            c = sm / cnt.clamp_min(1)[:, None]
            if emp.any():
                c[emp] = vs[torch.randperm(vs.shape[0], generator=g)[:int(emp.sum())]]
        return c
    cbs, codes, res = [], [], flat.clone()
    for c in range(ncb):
        cb = kmeans(res, seed + 17 * c)
        idx = nearest(res, cb)
        res = res - cb[idx]
        cbs.append(cb); codes.append(idx)
    for c in range(ncb):                                  # one refit round
        tgt = flat.clone()
        for c2 in range(ncb):
            if c2 != c:
                tgt -= cbs[c2][codes[c2]]
        a = nearest(tgt, cbs[c])
        sm = torch.zeros_like(cbs[c]).index_add_(0, a, tgt)
        cnt = torch.zeros(CB).index_add_(0, a, torch.ones(a.shape[0]))
        cbs[c] = torch.where((cnt > 0)[:, None], sm / cnt.clamp_min(1)[:, None], cbs[c])
        codes[c] = nearest(tgt, cbs[c])
    rec = sum(f16(cbs[c])[codes[c]] for c in range(ncb))
    return rec.reshape(N, K) * f16(s)[:, None]


def _ds_aq8(W, proj, layer, _c={}):
    if not _c:
        _c.update({k: v for k, v in qcfg("deepseek/qwen8-8")["rules"]})
    ncb = _c.get(f"model.layers.{layer}.mlp.{'down_proj' if proj == 'down' else 'gate_up_proj'}", 2)
    return _ds_rvq(W, ncb)


_r = "deepseek/qwen8-8"
reg("deepseek", "qwen8", "qwen8-8", _ds_aq8, 2, None,
    "additive VQ: per-row RMS fp16 scale, 2-4 residual k-means codebooks of 256 8-d fp16 vectors per matrix (ncb per tensor as stored), 1 refit round",
    ["calibration-aware sequential encoding (inverse-Hessian-weighted search + error feedback)", "per-row scale refit on activations"],
    ["work/dev/quantize.py:18-110"], coverage="per-tensor codebook count from the final config")


def _ds_oq3(W, p=None, L=None):
    wg, k = grp(W, 128)
    amax = wg.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    return ungrp(best_of(wg, (levels_q(wg, f16(amax * r / 3.5), midrise(3)) for r in (1.0, 0.97, 0.94, 0.90, 0.86, 0.82))), k)


_r = "deepseek/qwen122-48"
reg("deepseek", "qwen122", "qwen122-48", with_rot(_ds_oq3, block_in=128), 3, None,
    "midrise 8 levels (q-3.5)s, s = r amax/3.5, r in {1..0.82} per group, fp16, g128, random signed block-Hadamard-128 on inputs",
    ["GPTQ (FineWeb + self-continuation Hessians)"], ["work/dev/gptq_lib.py:5-21",
                                                     "work/dev/convert3.py:49-60"])


def _ds_qmix(W, proj, layer, _c={}):
    if not _c:
        _c.update(qcfg("deepseek/qwen8-12")["overrides"])
    bits, G = _c.get(f"model.layers.{layer}.mlp.{'down_proj' if proj == 'down' else 'gate_up_proj'}", [4, 128])
    mid = 1 << (bits - 1)
    wg, k = grp(W, G)
    amax = wg.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    rs = (1.0, 0.98, 0.96, 0.94, 0.92, 0.90, 0.87, 0.84, 0.80, 0.75) if bits <= 4 else (1.0,)
    cands = []
    for r in rs:
        s = f16(r * amax / (mid - 1))
        cands.append(s * torch.clamp(torch.round(wg / s), -mid, mid - 1))
    return ungrp(best_of(wg, cands), k)


_r = "deepseek/qwen8-12"
reg("deepseek", "qwen8", "qwen8-12", _ds_qmix, 3, None,
    "int sym full range [-2^(b-1), 2^(b-1)-1], s = r amax/(2^(b-1)-1), r in {1..0.75}, fp16; bits/group per tensor as stored (3/g128 mostly, 4/g64)",
    ["GPTQ", "Hessian-diagonal-weighted clip search", "measured per-module bit allocation (kept: as stored)"],
    ["work/tools/gptq_quantize.py:30-54"], coverage="per-tensor bits from the final config")


def set_rows(key, rows):
    GRIDS[key]["rows"] = rows


# ----------------------------------------------------------------------------- shared encoders
def viterbi(x, lut, L, k, zero_start=False, chunk=2048):
    """Bitshift-trellis Viterbi (Sonnet tcq_core.encode_viterbi / ref/tcq.py viterbi; Opus qtc/trellis.py for V>1):
    x [B, n, V] targets, lut [2^L, V]; each step emits lut[state], state_{t+1} = (state_t >> k) | (bits << (L-k)) in
    Opus's convention, equivalently the window shift; free (or zero) initial state. Returns recon [B, n, V]."""
    B, n, V = x.shape
    S, Gs, R = 1 << L, 1 << (L - k), 1 << k
    lut = lut.float().reshape(S, V)
    l2 = (lut * lut).sum(1)
    out = torch.empty_like(x)
    for a in range(0, B, chunk):
        xb = x[a:a + chunk]
        b = xb.shape[0]
        d = lambda t: l2[None] - 2 * xb[:, t] @ lut.t() + (xb[:, t] ** 2).sum(1, keepdim=True)
        cost = d(0)
        if zero_start:
            ok = torch.zeros(S, dtype=torch.bool); ok[torch.arange(R) * Gs] = True
            cost = cost.masked_fill(~ok[None], float("inf"))
        bp = torch.empty(n, b, Gs, dtype=torch.uint8)
        for t in range(1, n):
            m, arg = cost.view(b, Gs, R).min(2)          # predecessors of state (u << (L-k)) | v share v
            bp[t] = arg.to(torch.uint8)
            cost = (d(t).view(b, R, Gs) + m[:, None, :]).view(b, S)
        s = cost.argmin(1)
        st = torch.empty(b, n, dtype=torch.long)
        st[:, -1] = s
        ar = torch.arange(b)
        for t in range(n - 1, 0, -1):
            j = s & (Gs - 1)
            s = (j << k) | bp[t][ar, j].long()
            st[:, t - 1] = s
        out[a:a + chunk] = lut[st]
    return out


def tcq_rows(W, lut, k, L=12, V=1, T=256, zero_start=False, refit=True):
    """Row-RMS-normalised rows cut into T-weight trellis sequences (V weights per step), Viterbi, LS row-scale refit, fp16 scale."""
    N, K = W.shape
    sc = W.pow(2).mean(1).sqrt().clamp_min(1e-8)
    x = (W / sc[:, None]).reshape(-1, T // V, V)
    rec = viterbi(x, lut, L, k, zero_start).reshape(N, K)
    if refit:
        sc = ((rec * W).sum(1) / (rec * rec).sum(1).clamp_min(1e-20)).clamp_min(1e-8)
    return rec * f16(sc)[:, None]


_E8P = {}


def e8p_quantize(x):
    """E8P (QuIP#-style) 16-bit codebook for 8-d vectors: 256 abs patterns x 7 sign bits (8th by parity) x +-1/4 shift,
    nearest point; a copy of Opus's E8PCodebook._quantize (nemotron-32 work/scripts/e8p.py:29-110) with its abs table."""
    if not _E8P:
        A = torch.load(run_file("opus/nemotron-32", "work/scripts/e8p_abs_table.pt"),
                       weights_only=True).float()
        c = torch.remainder((2 * A).round().long(), 4)
        _E8P.update(A=A, p=((c.sum(-1) % 4) // 2) % 2, A2=(A ** 2).sum(-1))
    A, p, A2 = _E8P["A"], _E8P["p"], _E8P["A2"]
    out = torch.empty_like(x)
    for a0 in range(0, x.shape[0], 1 << 15):
        xc = x[a0:a0 + (1 << 15)]
        best = None
        for shift in (0.25, -0.25):
            y = xc - shift
            ay, neg = y.abs(), y < 0
            P = ay @ A.t()
            mval, marg = (ay[:, None, :] * A[None]).min(-1)
            mism = (neg.sum(-1)[:, None] % 2) != p[None]
            dd = A2[None] - 2 * (P - torch.where(mism, 2 * mval, torch.zeros_like(mval)))
            dmin, idx = dd.min(-1)
            dmin = dmin + (y * y).sum(-1)
            sg = torch.where(neg, -1.0, 1.0)
            fl = torch.where(mism.gather(1, idx[:, None])[:, 0], marg.gather(1, idx[:, None])[:, 0], torch.full_like(idx, -1))
            has = fl >= 0
            ar = torch.arange(xc.shape[0])
            sg[ar[has], fl[has]] *= -1
            q = sg * A[idx] + shift
            if best is None:
                best, bq = dmin, q
            else:
                m = dmin < best
                best = torch.where(m, dmin, best); bq = torch.where(m[:, None], q, bq)
        out[a0:a0 + xc.shape[0]] = bq
    return out


_CB16 = {}


def vq16(x, chunk=2048):
    """nearest codeword of the universal 2^16 x 8 Gaussian k-means codebook (bf16, as stored)."""
    if not _CB16:
        C = torch.load(SCRATCH / "cb8_16_gauss_retrained.pt", weights_only=True).float()
        C = C.bfloat16().float()
        _CB16.update(C=C, n=(C * C).sum(1))
    C, cn = _CB16["C"], _CB16["n"]
    return torch.cat([C[(2 * x[i:i + chunk] @ C.t() - cn).argmax(1)] for i in range(0, x.shape[0], chunk)])


def rowscaled_vq(W, enc, scale_mult=1.0, refit=False):
    N, K = W.shape
    s = (W.pow(2).mean(1).sqrt() * scale_mult).clamp_min(1e-12)
    q = enc((W / s[:, None]).reshape(-1, 8)).reshape(N, K)
    if refit:
        s = ((W * q).sum(1) / (q * q).sum(1).clamp_min(1e-20)).clamp_min(1e-12)
    return q * f16(s)[:, None]


# ----------------------------------------------------------------------------- Opus
_r = "opus/nemotron-32"
reg("opus", "nemotron", "nemotron-32",
    per_proj(up=with_rot(lambda W, p, L: rowscaled_vq(W, e8p_quantize, 0.97), block_in=1024),
             down=lambda W, p, L: _hph(rowscaled_vq(_hph(W), e8p_quantize, 0.97), inverse=True)),
    2, None, "E8P lattice VQ (16-bit code per 8 weights), per-row fp16 scale 0.97 x RMS, rotated input basis (H1024 up; Paley84xH32 down, modelled as 2-round block-128 H.P.H)",
    ["BlockLDLQ (Hessian feedback)", "row-scale multiplier search on the H proxy", "3-bit tier (E8P + 1-bit E8 residual) on calibration-chosen experts"],
    ["work/scripts/e8p.py:29-110",
     "work/scripts/quantize.py:44,273-290,531-535"],
    coverage="2-bit tier; t3 experts (listed in expert_tiers) also modelled at 2 bit")
set_rows(_r, 512)


def _op_qx(W, proj, layer, expert, _c={}):
    """qx/quantize4.py: midrise (q-c)s, one scale per 1024-input block, s = t amax/c, t in linspace(lo,1,36)."""
    if not _c:
        _c.update(qcfg("opus/qwen122-32")["experts"])
    bits = _c[str(layer)]["gu" if proj in ("gate", "up") else "dn"][expert]
    lo = {2: 0.30, 3: 0.40}[bits]
    wg, k = grp(W, 1024)
    amax = wg.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    c = ((1 << bits) - 1) / 2
    return ungrp(best_of(wg, (levels_q(wg, f16(amax * t / c), midrise(bits)) for t in lin(lo, 1.0, 36))), k)


_r = "opus/qwen122-32"
reg("opus", "qwen122", "qwen122-32", with_rot(_op_qx, block_in=1024), 3, None,
    "midrise, one fp16 scale per 1024-input block of a row, scale search t in linspace(lo,1,36); bits per expert as stored (gate_up 3, down 2; 3 in 3 layers); Hadamard-1024 on inputs",
    ["GPTQ", "H-weighted scale refit"], ["work/qx/quantize4.py:1-20,121-195"],
    coverage="per-expert bits from the final config")
GRIDS[_r]["needs_expert"] = True


def _op_tcq(W, proj, layer, _c={}):
    if not _c:
        q = qcfg("opus/qwen8-8"); _c["bits"] = q["bits"]; _c["luts"] = q["luts"]
    s = _c["bits"].get(f"{layer}.{'mlp_down' if proj == 'down' else 'mlp_in'}", 6)
    return tcq_rows(W, torch.tensor(_c["luts"][str(s)]), s, L=12, V=2)


_r = "opus/qwen8-8"
reg("opus", "qwen8", "qwen8-8", with_rot(_op_tcq, block_in=1024), 2, None,
    "2-D bitshift-trellis TCQ (L=12, s bits per 2 weights: s=4 -> 2 bpw, 5 -> 2.5 bpw per tensor as stored), LUT [4096,2] as stored, per-row RMS scale + LS refit (fp16), randomized Hadamard on inputs",
    ["LDLQ (Hessian feedback)", "H-weighted row-scale refit", "KD fine-tuning", "tail-biting (free start used)"],
    ["work/qtc/trellis.py:1-62", "work/qtc/quantize.py:134-173"],
    coverage="per-tensor rate from the final config")
set_rows(_r, 128)


def _op_lowbit(W, proj, layer, expert, _c={}):
    if not _c:
        _c.update(qcfg("opus/qwen122-48")["expert_bits"])
    bits = _c[str(layer)]["w13" if proj in ("gate", "up") else "w2"][expert]
    maxq = (1 << bits) - 1
    wg, k = grp(W, 128)
    xmin, xmax = wg.amin(-1, keepdim=True), wg.amax(-1, keepdim=True)
    cands = []
    for p in (1.0, 0.96, 0.92, 0.88, 0.84, 0.80, 0.76, 0.72):
        s = f16(((xmax - xmin) * p / maxq).clamp_min(1e-6)); m = f16(xmin * p)
        cands.append(torch.clamp(torch.round((wg - m) / s), 0, maxq) * s + m)
    return ungrp(best_of(wg, cands), k)


_r = "opus/qwen122-48"
reg("opus", "qwen122", "qwen122-48", _op_lowbit, 2, None,
    "asymmetric, fp16 scale + fp16 offset (= clipped min), shrink {1..0.72} per group, g128; per-expert bits 2-5 as stored",
    ["GPTQ", "H-diagonal-weighted clip search"], ["work/scripts/lowbit_common.py:39-70"],
    coverage="per-expert bits from the final config")
GRIDS[_r]["needs_expert"] = True


def _op_hgq(W, proj, layer, _c={}):
    """hgq: rotated (block 256), groups of 256, pairs (w_i, w_{i+128}) on a 2-D Lloyd grid of 2^kb points, scale = rms x mult (6), log8 row-base scales."""
    if not _c:
        q = qcfg("opus/qwen8-12"); _c["grids"] = q["grids"]; _c["layers"] = q["layers"]
    key = f"model.layers.{layer}.mlp.{'down_proj' if proj == 'down' else 'gate_up_proj'}"
    sh = [x for x in _c["layers"][key]["shards"] if x["name"] == proj or (proj == "down")][0]
    grid = torch.tensor(_c["grids"][str(sh["kb"])]).float()
    N, K = W.shape
    G, h = 256, 128
    Wg = W.reshape(N, K // G, G)
    pairs = torch.stack([Wg[..., :h], Wg[..., h:]], -1)
    rms = Wg.pow(2).mean(-1).sqrt().clamp_min(1e-12)
    g2 = (grid * grid).sum(1)
    near = lambda z: grid[torch.cat([(2 * z[i:i + 65536] @ grid.t() - g2).argmax(1) for i in range(0, z.shape[0], 65536)])]
    best = be = None
    for m in (0.88, 0.94, 1.0, 1.06, 1.12, 1.18):
        s = (rms * m)
        base = s.amax(1, keepdim=True)                       # log8 row-base: s = base * 2^(-c/16), c in 0..255
        cq = torch.clamp(torch.round(-16 * torch.log2(s / base)), 0, 255)
        s = (f16(base) * torch.exp2(-cq / 16))[..., None, None]
        q = near((pairs / s).reshape(-1, 2)).view_as(pairs) * s
        e = ((pairs - q) ** 2).sum((-1, -2))
        if best is None:
            best, be = q, e
        else:
            u = e < be
            be = torch.where(u, e, be); best = torch.where(u[..., None, None], q, best)
    return torch.cat([best[..., 0], best[..., 1]], -1).reshape(N, K)


_r = "opus/qwen8-12"
reg("opus", "qwen8", "qwen8-12", with_rot(_op_hgq, block_in=256), 3.5, None,
    "2-D VQ: pairs (w_i, w_i+128) of each 256-group on a stored 2-D Lloyd grid of 2^kb points (kb 7 gate/up -> 3.5 bpw, 8 down -> 4 bpw), scale = rms x mult (6), log8 row-base scales, RHT block 256",
    ["LDLQ (Hessian feedback)", "H-weighted scale refit"], ["work/hgq/hgq_quant.py:84-203"],
    coverage="per-shard kb from the final config")


# ----------------------------------------------------------------------------- Kimi
def _mix_in(W, inverse=False):
    """the input-side orthogonal mix: Hadamard-1024 if the dim is a multiple of 1024, else 2-round block-128 H.P.H."""
    return rot_in(W, 1024, 5, inverse) if W.shape[1] % 1024 == 0 else _hph(W, seed=5, inverse=inverse)


def _kimi_e8p_rvq(W):
    """pack_model.py quantize_rvq (1 level): RMS row scale, E8P encode, LS refit, re-encode, LS refit, bf16 scale."""
    s = W.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-20)
    for _ in range(2):
        q = e8p_quantize((W / s).reshape(-1, 8)).reshape(W.shape)
        s = ((W * q).sum(-1, keepdim=True) / q.square().sum(-1, keepdim=True).clamp_min(1e-20)).clamp_min(1e-20)
    return q * s.bfloat16().float()


def _kimi_n32(W, proj, layer):
    X = _mix_in(_mix_in(W).t().contiguous()).t().contiguous()          # left and right rotations
    Q = _kimi_e8p_rvq(X)
    return _mix_in(_mix_in(Q.t().contiguous(), True).t().contiguous(), True)


_r = "kimi/nemotron-32"
reg("kimi", "nemotron", "nemotron-32", _kimi_n32, 2, None,
    "E8P lattice VQ (16-bit code / 8 weights) per row, RMS scale + 2 LS refits (bf16), two-sided rotation (FWHT x 21-block mix modelled as H1024 / block-128 H.P.H)",
    [], ["work/pack_model.py:22-78,157"],
    coverage="all experts; E8P abs table from Opus's run (same E8P construction)")
set_rows(_r, None)

_r = "kimi/qwen122-32"
reg("kimi", "qwen122", "qwen122-32",
    grid_fn("midrise", 2, 128, (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75), scope="matrix", scale_fmt=lambda x: x.bfloat16().float()),
    2, None, "midrise 4 levels (q-1.5)s, s = f amax/1.5 with one factor f per matrix from {0.45..0.75}, bf16 scale, g128",
    ["GPTQ", "per-matrix factor chosen on an H proxy (RTN form: weight MSE)"],
    ["work/compress.py:32-53", "work/recompress_adaptive.py:45-60"])

_LM2 = torch.tensor([-1.5104, -0.4528, 0.4528, 1.5104])


def _kimi_q8(W, proj=None, layer=None):
    wg, k = grp(W, 128)
    rms = wg.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-12)
    return ungrp(best_of(wg, (levels_q(wg, f16(rms * m), _LM2) for m in (0.60, 0.72, 0.84, 0.94, 1.04, 1.16, 1.32, 1.55))), k)


_r = "kimi/qwen8-8"
reg("kimi", "qwen8", "qwen8-8", _kimi_q8, 2, None,
    "fixed Gaussian Lloyd-Max 4 levels, s = rms x m (8 multipliers) per group, fp16, g128",
    ["GPTQ (H-weighted)", "distilled rank-16 residual"], ["work/quantize.py:21-107"])

_K3C = None


def _kimi_k3(W, proj, layer):
    global _K3C
    if _K3C is None:
        _K3C = torch.tensor(_json.load(open(run_file("kimi/qwen122-48", "work/calibration/nonuniform_codebook.json")))).float()
    four = (proj in ("gate", "up") and layer in (44, 45, 46, 47)) or (proj == "down" and layer in (46, 47))
    bits = 4 if four else 3
    qmax = (1 << bits) - 1
    centers = torch.arange(qmax + 1).float() if four else _K3C
    bnd = ((centers[:-1] + centers[1:]) / 2).contiguous()
    wg, k = grp(W, 128)
    lo0, hi0 = wg.amin(-1, keepdim=True), wg.amax(-1, keepdim=True)
    cands = []
    for sh in (1.0, 0.975, 0.95, 0.925, 0.9, 0.85, 0.8):
        s = f16((hi0 - lo0) * sh / qmax); s = torch.where(s > 0, s, torch.ones_like(s))
        z = torch.round(-lo0 * sh / s).clamp(0, qmax)
        cands.append((centers[torch.bucketize((wg / s + z).contiguous(), bnd)] - z) * s)
    return ungrp(best_of(wg, cands), k)


_r = "kimi/qwen122-48"
reg("kimi", "qwen122", "qwen122-48", _kimi_k3, 3, None,
    "asymmetric with a shared nonuniform 8-level codebook on the [0,7] code axis, fp16 scale + integer zero, shrink {1..0.8} per group, g128; gate_up 4-bit in layers 44-47, down in 46-47 (uniform)",
    ["expert-diagonal-weighted shrink choice", "GPTQ"], ["work/requant_mse_expert.py:21-99",
                                                     "work/calibration/nonuniform_codebook.json"])

_SEL = None


def _kimi_s6(W, proj, layer):
    global _SEL
    if _SEL is None:
        _SEL = set(_json.load(open(run_file("kimi/qwen8-12", "work/calib/selected_allocation_scale6.json")))["selected_weights"])
    name = f"model.language_model.layers.{layer}.mlp.{proj}_proj.weight"
    bits = 4 if (proj == "down" or name in _SEL) else 3
    L = (1 << bits) - 1
    wg, k = grp(W, 64)
    mn, mx = wg.amin(-1, keepdim=True), wg.amax(-1, keepdim=True)
    s = ((mx - mn) / L).clamp_min(1e-8)
    z = torch.round(-mn / s).clamp(0, L)
    q = torch.clamp(torch.round(wg / s + z), 0, L)
    s16 = f16(s)[..., 0]                                            # scale6: 6-bit mantissa under a per-row power of two
    row = torch.exp2(torch.ceil(torch.log2(s16.amax(1, keepdim=True))))
    s6 = (torch.round(s16 / row * 63).clamp(0, 63) / 63 * row)[..., None]
    return ungrp((q - z) * s6, k)


_r = "kimi/qwen8-12"
reg("kimi", "qwen8", "qwen8-12", _kimi_s6, 3, None,
    "asymmetric min-max, g64, 4-bit zero, 6-bit linear scale mantissa under a per-row power of two; down 4-bit, gate/up 3-bit unless in the selected 4-bit set",
    ["GPTQ", "covariance-weighted knapsack choice of the 4-bit set (kept: as stored)"],
    ["work/quantize_scale6.py:21-106"], coverage="per-matrix bits from the final allocation")


# ----------------------------------------------------------------------------- Sonnet
_r = "sonnet/nemotron-32"
reg("sonnet", "nemotron", "nemotron-32", lambda W, p, L: rowscaled_vq(W, vq16), 2, None,
    "8-d VQ, 16-bit code into a universal 2^16 x 8 Gaussian k-means codebook (bf16; re-trained by train_vq_codebook.py), per-row RMS fp16 scale",
    ["latent-output refit folded into fc2", "8-bit residual VQ on layers 74,76,79 (not sampled)"],
    ["work/scripts/qdriver.py:22,88-96,300-332",
     "work/scripts/train_cb.py"])
set_rows(_r, 256)


def _sn_q122(W, proj, layer):
    b = 1024 if W.shape[1] % 1024 == 0 and W.shape[1] > 1024 else 64
    return rot_in(rowscaled_vq(rot_in(W, b, 3), vq16), b, 3, inverse=True)


_r = "sonnet/qwen122-32"
reg("sonnet", "qwen122", "qwen122-32", _sn_q122, 2, None,
    "8-d VQ with the universal 2^16 x 8 Gaussian codebook, per-row fp16 RMS scale, randomized Hadamard on inputs (block 1024 for 3072, 64 for 1024)",
    ["residual VQ stage on the ~16% experts carrying most routed energy (calibration-chosen)"],
    ["lineage v4: files/README.md"])
set_rows(_r, 256)

_SNL = {}


def _sn_tcq8(W, proj, layer):
    if not _SNL:
        for K in (1, 2, 3, 4):
            _SNL[K] = torch.load(run_file("sonnet/qwen8-8", f"work/ckpt/lut_K{K}_L12.pt"), weights_only=True).float()
        _SNL["words"] = qcfg("sonnet/qwen8-8")["layer_words"]
    key = f"layers.{layer}.mlp.{'down_proj' if proj == 'down' else 'gate_up_proj'}"
    m, n = (5120, 17408) if proj == "down" else (34816, 5120)
    kbar = _SNL["words"][key] * 32 / (m * n)
    lo = max(1, int(math.floor(kbar))); hi = min(4, lo + 1)
    nhi = int(round((kbar - lo) * W.shape[0] / 16)) * 16 if hi > lo else 0
    out = torch.empty_like(W)
    if nhi:
        out[:nhi] = tcq_rows(W[:nhi], _SNL[hi], hi, zero_start=True)
    if nhi < W.shape[0]:
        out[nhi:] = tcq_rows(W[nhi:], _SNL[lo], lo, zero_start=True)
    return out


_r = "sonnet/qwen8-8"
reg("sonnet", "qwen8", "qwen8-8", with_rot(_sn_tcq8, block_in=1024), 2, None,
    "bitshift-trellis TCQ (L=12, zero start, 256-weight sequences, stored 4096-entry LUT per K), K in {1..4} per 16-row tile-row with the module's stored mean K, row RMS scale + LS refit (fp16), randomized Hadamard on inputs",
    ["LDLQ (Hessian feedback)", "Fisher-chosen K per tile-row (kept: module mean K)", "distillation of scales/signs/norms"],
    ["work/src/tcq_core.py:13-44", "work/src/ldlq2.py:9-39",
     "work/SPEC_V2.md"], coverage="mean K per module from the final config's layer_words")
set_rows(_r, 128)


def _sn_bpq(W, proj, layer, expert, _c={}):
    if not _c:
        _c.update(qcfg("sonnet/qwen122-48")["experts"])
    ent = [e for e in _c[str(layer)]["gate_up" if proj in ("gate", "up") else "down"] if expert in e["ids"]][0]
    bits, G, c = ent["bits"], ent["group"], ent["c"]
    v = torch.arange(1 << bits).float() - ((1 << bits) - 1) / 2
    lv = v * (1 + c * v * v)
    wg, k = grp(W, G)
    amax = wg.abs().amax(-1, keepdim=True).clamp_min(1e-8)
    sel = None
    for r in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6):  # pick r, then fp16 scale and re-encode (bpq_ref.py)
        sr = amax * r / lv[-1]
        e = ((levels_q(wg, sr, lv) - wg) ** 2).sum(-1, keepdim=True)
        sel = (e, sr) if sel is None else (torch.where(e < sel[0], e, sel[0]), torch.where(e < sel[0], sr, sel[1]))
    s = f16(sel[1]).clamp_min(6.2e-5)
    return ungrp(levels_q(wg, s, lv), k)


_r = "sonnet/qwen122-48"
reg("sonnet", "qwen122", "qwen122-48", _sn_bpq, 3, None,
    "companded midrise levels v(1+c v^2), s = r amax/level_max (9 r), fp16; bits 2-5, group, c per expert as stored",
    ["GPTQ", "calibration-chosen per-expert bits (kept: as stored)"],
    ["work/dev/bpq_ref.py:11-46"], coverage="per-expert bits from the final config")
GRIDS[_r]["needs_expert"] = True

_SNG = {}


def _sn_tgq(W, proj, layer):
    if not _SNG:
        for k in range(2, 9):
            _SNG[k] = torch.load(run_file("sonnet/qwen8-12", f"work/calib/tcq_cb_L12_k{k}.pt"), weights_only=True).float()
        _SNG["layout"] = qcfg("sonnet/qwen8-12")["layout"]
    blocks = _SNG["layout"][f"{layer}.mlp.{'down_proj' if proj == 'down' else 'gate_up_proj'}"]["blocks"]
    off = 0 if proj != "up" else 17408                       # gate rows first in the fused gate_up
    k = [b["k"] for b in blocks if b["rows"][0] <= off < b["rows"][1]][0]
    N, K = W.shape
    sig = f16(W.reshape(N, K // 1024, 1024).pow(2).mean(-1).sqrt().clamp_min(1e-8))
    x = (W.reshape(N, K // 1024, 1024) / sig[..., None]).reshape(-1, 256, 1)
    rec = viterbi(x, _SNG[k], 12, k).reshape(N, K // 1024, 1024)
    return (rec * sig[..., None]).reshape(N, K)


_r = "sonnet/qwen8-12"
reg("sonnet", "qwen8", "qwen8-12", with_rot(_sn_tgq, block_in=1024), 3, None,
    "bitshift-trellis TCQ (L=12, free start, 256-weight sequences, stored 4096-entry codebook per k), k per layer/row block as stored (2..8), RMS scale per row and 1024-block (fp16), RHT block 1024",
    ["TCQ-GPTQ (Hessian feedback)"], ["work/ref/tcq.py:6-32,99-110"],
    coverage="k per row block from the final layout (first block covering the sampled rows)")
set_rows(_r, 128)
