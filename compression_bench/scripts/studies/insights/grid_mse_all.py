"""Offline, CPU-only: relative weight MSE ||W-What||^2/||W||^2 on sampled real matrices of the three source
models, for (a) a common set of controlled grids (quant_interventions.py's RTN grids, groups of 128) on
every sampled tensor class, and (b) each analysed run's final main-class grid (routed experts / dense MLP)
in RTN form (run_grids.py; Nemotron-48 grids from grid_mse.py).

Sampled tensors:
  nemotron, qwen122: fetch_sampled_tensors.py (HTTP range requests at the contract revisions; MANIFEST.json)
  qwen8: the local BF16 copy compression_bench/models/Qwen3.8-27B-bf16 (revision 1d4bf0f2), row subsets below
Class aggregates are energy-weighted: sum ||dW||^2 / sum ||W||^2 over the class's sampled matrices.

    NT=24 python3 grid_mse_all.py controlled                  # -> quant_error_grid_controlled.json (scratch)
    NT=24 python3 grid_mse_all.py runs [AGENT|KEY,...]        # -> quant_error_grid_runs.json (scratch)
    NT=24 python3 grid_mse_all.py pruning KEY:KEPT [KEY:KEPT ...]

Scratch is common.QUANT_ERROR_DIR (compression_bench/evolve_space/analysis/quant_error). `runs` keys its
results by run id, resolving each grid key (`<agent>/<model>-<cap>`) through the version ledger (run_grids.py).
`pruning` adds a width-pruning lower bound to a run that removed intermediate channels: KEY is its grid key,
KEPT the number of intermediate channels its final config keeps per MLP (qwen8) or routed expert (nemotron).
"""
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from common import QUANT_ERROR_DIR as SCRATCH, REPO  # noqa: E402
import quant_interventions as qi  # noqa: E402

torch.set_num_threads(int(os.environ.get("NT", "24")))
CACHE = SCRATCH / "tensors"
Q8 = REPO / "compression_bench/models/Qwen3.8-27B-bf16"

# qwen8 samples: (name, class, layer, rows) -- rows: a leading row subset (row-wise grids: unbiased)
Q8_SAMPLES = []
for L in (0, 16, 32, 48, 63):
    for p, rows in (("gate", 2048), ("up", 2048), ("down", 1024)):
        Q8_SAMPLES.append((f"model.language_model.layers.{L}.mlp.{p}_proj.weight", f"mlp_{p}", L, rows))
for L in (0, 16, 32, 48):
    for p in ("in_proj_qkv", "in_proj_z", "out_proj"):
        Q8_SAMPLES.append((f"model.language_model.layers.{L}.linear_attn.{p}.weight", f"lin_{p}", L, 2048))
for L in (3, 31, 63):
    for p in ("q", "k", "v", "o"):
        Q8_SAMPLES.append((f"model.language_model.layers.{L}.self_attn.{p}_proj.weight", f"attn_{p}", L, 2048))
for r in (0, 32768, 65536, 98304):
    Q8_SAMPLES.append((f"model.language_model.embed_tokens.weight@{r}", "embed", None, 1024))
    Q8_SAMPLES.append((f"lm_head.weight@{r}", "lm_head", None, 1024))


def bf16(a):
    return torch.from_numpy((a.astype(np.uint32) << 16).view(np.float32))


def load(model):
    """[(class, layer, expert, W float32 [out, in], source)] for one model."""
    out = []
    if model == "qwen8":
        from safetensors import safe_open
        wm = json.load(open(Q8 / "model.safetensors.index.json"))["weight_map"]
        for name, cls, L, rows in Q8_SAMPLES:
            n, _, r0 = name.partition("@")
            r0 = int(r0 or 0)
            with safe_open(str(Q8 / wm[n]), "pt") as f:
                W = f.get_slice(n)[r0:r0 + rows].float()
            out.append((cls, L, None, W, f"{n}[{r0}:{r0 + W.shape[0]}]"))
        return out
    man = json.load(open(CACHE / model / "MANIFEST.json"))
    for t in man["tensors"]:
        W = bf16(np.load(CACHE / model / t["file"]))
        cls = t["cls"]
        if cls == "routed_gate_up":                     # fused rows: gate then up
            I = W.shape[0] // 2
            out.append(("routed_gate", t["layer"], t["expert"], W[:I].contiguous(), t["file"] + "[:I]"))
            out.append(("routed_up", t["layer"], t["expert"], W[I:].contiguous(), t["file"] + "[I:]"))
        else:
            out.append((cls, t.get("layer"), t.get("expert"), W, t["file"]))
    return out


def rot_block(K):
    b = 1
    while b < 1024 and K % (2 * b) == 0:
        b *= 2
    return b


CONTROLLED = {
    "U3": dict(grid="U", bits=3), "U3c": dict(grid="U", bits=3, clip=True), "NF3c": dict(grid="NF", bits=3, clip=True),
    "KM3c": dict(grid="KM", bits=3, clip=True), "U3c_rot": dict(grid="U", bits=3, clip=True, rot=True),
    "U2c": dict(grid="U", bits=2, clip=True), "NF2c": dict(grid="NF", bits=2, clip=True),
    "U2c_rot": dict(grid="U", bits=2, clip=True, rot=True), "U4c": dict(grid="U", bits=4, clip=True),
    "NF4c": dict(grid="NF", bits=4, clip=True),
}


def controlled():
    res = {}
    for model in ("qwen8", "nemotron", "qwen122"):
        mats = load(model)
        acc = {}
        for cls, L, e, W, src in mats:
            qi.HBLOCK = rot_block(W.shape[1])
            w2 = (W * W).sum().item()
            for g, spec in CONTROLLED.items():
                Q, nb = qi.quantize_matrix(W, None, spec, spec["bits"])
                a = acc.setdefault(cls, {}).setdefault(g, [0.0, 0.0, 0.0, 0])
                a[0] += ((W - Q) ** 2).sum().item(); a[1] += w2; a[2] += nb; a[3] += W.numel()
        qi.HBLOCK = 1024
        res[model] = {cls: {g: dict(rel_mse=v[0] / v[1], bpw=v[2] / v[3]) for g, v in d.items()} for cls, d in acc.items()}
        res[model + "_sampled"] = [dict(cls=c, layer=L, expert=e, shape=list(W.shape), source=s) for c, L, e, W, s in mats]
        print(model, json.dumps({c: {g: round(v["rel_mse"], 5) for g, v in d.items()} for c, d in res[model].items()}), flush=True)
    json.dump(res, open(SCRATCH / "quant_error_grid_controlled.json", "w"), indent=1)


def runs(only=None):
    """only: comma list of agents or grid keys; results merge into the existing file, keyed by run id."""
    import run_grids
    out = SCRATCH / "quant_error_grid_runs.json"
    res = json.load(open(out)) if out.exists() else {}
    only = set(only.split(",")) if only else None
    for model in ("qwen8", "nemotron", "qwen122"):
        mats = [m for m in load(model) if m[0].startswith(("routed", "mlp_"))]
        for key, g in run_grids.GRIDS.items():
            if g["model"] != model or (only and key not in only and g["agent"] not in only):
                continue
            try:
                run = run_grids.run_record(key)["run"]
            except KeyError as e:
                print("skip", e, flush=True)
                continue
            acc = {}
            rows = g.get("rows", 1024)
            for cls, L, e, W, src in mats:
                proj = cls.split("_")[-1]
                if rows:
                    W = W[:rows]                       # a leading row subset: unbiased for row-wise grids
                Q = g["fn"](W, proj, L, e) if g.get("needs_expert") else g["fn"](W, proj, L)
                a = acc.setdefault(proj, [0.0, 0.0])
                a[0] += ((W - Q) ** 2).sum().item(); a[1] += (W * W).sum().item()
            tot = [sum(v[0] for v in acc.values()), sum(v[1] for v in acc.values())]
            res[run] = dict({k: v for k, v in g.items() if k != "fn"}, grid=key, rows_per_matrix=rows, rel_mse=tot[0] / tot[1],
                            rel_mse_by_proj={p: v[0] / v[1] for p, v in acc.items()})
            print(run, key, f"{tot[0] / tot[1]:.5f}", {p: round(v[0] / v[1], 5) for p, v in acc.items()}, flush=True)
            json.dump(res, open(out, "w"), indent=1)


def _channel_energy(model):
    """Per intermediate channel, ||gate/up row||^2 + ||down column||^2, for each sampled MLP / routed expert.
    qwen8: full matrices of the sampled layers; nemotron: the sampled experts' full matrices."""
    if model == "qwen8":
        from safetensors import safe_open
        wm = json.load(open(Q8 / "model.safetensors.index.json"))["weight_map"]
        for L in (0, 16, 32, 48, 63):
            en = 0
            for p in ("gate", "up", "down"):
                n = f"model.language_model.layers.{L}.mlp.{p}_proj.weight"
                with safe_open(str(Q8 / wm[n]), "pt") as f:
                    W = f.get_tensor(n).float()
                en = en + (W.pow(2).sum(1) if p != "down" else W.pow(2).sum(0))
            yield en
    elif model == "nemotron":
        mats = {(c, L, e): W for c, L, e, W, s in load("nemotron") if c in ("routed_up", "routed_down")}
        for (c, L, e), W in mats.items():
            if c == "routed_up":
                yield W.pow(2).sum(1) + mats[("routed_down", L, e)].pow(2).sum(0)
    else:
        raise SystemExit(f"width pruning is modelled for qwen8 and nemotron, not {model}")


def pruning(*specs):
    """Width-pruning runs: a lower bound on the main class's error including the removed intermediate channels.
    A removed channel j takes all of gate/up row j and down column j; the least energy any choice of the same number
    of channels can remove is that of the lowest-energy channels, so
        rel_err_lb = [E_drop_min + mse_kept * (E - E_drop_min)] / E   (mse_kept: the run's grid on the kept weights).
    specs: KEY:KEPT, the run's grid key and the intermediate channels its final config keeps. (Whole-expert removal
    is not a reconstruction error; quant_error_assemble.py reports it as a share.)"""
    import run_grids
    if not specs:
        raise SystemExit("usage: grid_mse_all.py pruning KEY:KEPT [KEY:KEPT ...]")
    out = SCRATCH / "quant_error_grid_runs.json"
    res = json.load(open(out))
    for spec in specs:
        key, _, kept = spec.rpartition(":")
        kept = int(kept)
        run = run_grids.run_record(key)["run"]
        E = Ed = 0.0
        of = None
        for en in _channel_energy(run_grids.GRIDS[key]["model"]):
            of = en.numel()
            E += en.sum().item(); Ed += en.sort().values[:of - kept].sum().item()
        m = res[run]["rel_mse"]
        res[run]["width_pruning"] = dict(kept=kept, of=of, dropped_energy_share_lb=Ed / E,
                                         rel_err_incl_pruning_lb=(Ed + m * (E - Ed)) / E)
        print(run, key, res[run]["width_pruning"])
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    {"controlled": controlled, "runs": runs, "pruning": pruning}[sys.argv[1]](*sys.argv[2:])
