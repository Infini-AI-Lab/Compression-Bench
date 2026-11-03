"""Assemble quant_error.json from the three measurements:

  controlled grids x sampled real matrices (grid_mse_all.py controlled)   -> weight rel. MSE per tensor class
  each analysed run's final grid in RTN form (grid_mse_all.py runs)       -> weight rel. MSE of its main class
  Qwen3.8-27B interventions (quant_interventions.py in gpu_worker.py)     -> bytes, weight MSE, output error, excess CE

plus, for the analysed runs, the verdict's excess (ln PPL/PPL_bf16, version_ledger.json) and the stored bytes per
tensor class (byte_allocation.json). Runs: those of the run-selection file (--runs: {"runs": [{"run", "agent",
"model", "cap", ...}]}). No hidden input is read: model loss here is on the task's public PPL blocks; the verdict
numbers are the ledger's.

    python3 quant_error_assemble.py [--runs F] [--ledger F] [--byte-allocation F] [--out F]
                                    [--experts-removed RUN=SHARE ...]

--experts-removed gives, for a run whose final config removes whole routed experts, the removed share of experts
(it is not a reconstruction error and is reported beside the grid error). Defaults read from and write to
evaluation_progress/trajectory_analysis/ (the outputs of version_ledger.py and byte_allocation.py); the
measurements from common.QUANT_ERROR_DIR (compression_bench/evolve_space/analysis/quant_error).
"""
import argparse
import json
import math
from itertools import combinations
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import QUANT_ERROR_DIR as SCR, REPO, TRAJECTORY_ANALYSIS as TA  # noqa: E402
IV = SCR / "interventions"


def spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for k, i in enumerate(o):
            r[i] = k
        return r
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=str(TA / "selected_runs.json"))
    ap.add_argument("--ledger", default=str(TA / "version_ledger.json"))
    ap.add_argument("--byte-allocation", default=str(TA / "byte_allocation.json"))
    ap.add_argument("--out", default=str(TA / "quant_error.json"))
    ap.add_argument("--experts-removed", action="append", default=[], metavar="RUN=SHARE")
    a = ap.parse_args()
    experts_removed = {k: float(v) for k, _, v in (x.partition("=") for x in a.experts_removed)}
    sel = json.load(open(a.runs))["runs"]
    ledger = {r["run"]: r for r in json.load(open(a.ledger))["runs"]}
    alloc = json.load(open(a.byte_allocation))
    ctrl = json.load(open(SCR / "quant_error_grid_controlled.json"))
    rg = json.load(open(SCR / "quant_error_grid_runs.json"))
    here = HERE.relative_to(REPO)
    out = {"about": __doc__.strip(), "scripts": {
        "fetch": f"{here}/fetch_sampled_tensors.py",
        "weight_grids": f"{here}/grid_mse_all.py (+ run_grids.py, grid_mse.py)",
        "vq_codebook": f"{here}/train_vq_codebook.py",
        "interventions": f"{here}/quant_interventions.py (run by gpu_worker.py)",
        "assemble": f"{here}/quant_error_assemble.py"}}

    # ---------------------------------------------------------------- sampled tensors
    samp = {}
    for m in ("nemotron", "qwen122"):
        man = json.load(open(SCR / "tensors" / m / "MANIFEST.json"))
        samp[m] = dict(repo=man["repo"], revision=man["revision"], contract=man["contract"],
                       fetched_by="HTTP range requests (released_identity.get)",
                       tensors=[{k: t[k] for k in ("name", "cls", "layer", "expert", "row0", "rows", "shape", "shard", "byte_range") if k in t}
                                for t in man["tensors"]])
    samp["qwen8"] = dict(repo="Qwen/Qwen3.8-27B", revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
                         local="compression_bench/models/Qwen3.8-27B-bf16", tensors=ctrl["qwen8_sampled"])
    out["sampled_tensors"] = samp

    # ---------------------------------------------------------------- controlled grids
    out["controlled_grids"] = {
        "grids": {"U3": "uniform asym min-max 3b g128 fp16 scale + 3b zero", "U3c": "+ per-group MSE clip (20 shrink 1..0.525)",
                  "NF3c": "Gaussian Lloyd-Max 8 levels, sym, fp16 scale, clip search", "KM3c": "per-matrix k-means 8-entry codebook, scale search",
                  "U3c_rot": "U3c after randomized block Hadamard on inputs (block = largest power of 2 <= 1024 dividing K)",
                  "U2c / NF2c / U2c_rot / U4c / NF4c": "same at 2 / 4 bits"},
        "rel_mse": {m: {c: {g: round(v["rel_mse"], 6) for g, v in d.items()} for c, d in ctrl[m].items()} for m in ("qwen8", "nemotron", "qwen122")},
        "bpw": {g: v["bpw"] for g, v in next(iter(ctrl["qwen8"].values())).items()},
        "aggregation": "energy-weighted per class: sum ||dW||^2 / sum ||W||^2 over the sampled matrices",
    }

    # ---------------------------------------------------------------- run grids vs verdict
    runs = {}
    for r in sel:
        g = rg.get(r["run"])
        L = ledger[r["run"]]
        f = L["final"]
        ex = math.log(f["ppl"] / f["bf16_ppl"]) if f.get("ppl") and f.get("bf16_ppl") else None
        a = alloc.get(r["run"], {})
        bpw = a.get("bits_per_weight", {})
        gib = a.get("gib", {})
        main_cls = "dense_mlp" if r["model"] == "qwen8" else "routed"
        nonmain = sum(v for k, v in gib.items() if k not in (main_cls, "vision")) if gib else None
        runs[r["run"]] = dict(
            task=f"{r['model']}-{r['cap']}", agent=r["agent"], status=f["status"],
            verdict_excess_nats=ex, verdict_ppl=f.get("ppl"), verdict_bf16_ppl=f.get("bf16_ppl"),
            grid_rel_mse=g and g["rel_mse"], grid_rel_mse_by_proj=g and g["rel_mse_by_proj"],
            width_pruning=g and g.get("width_pruning"),
            grid_effective_rel_err=g and (g["width_pruning"]["rel_err_incl_pruning_lb"] if g.get("width_pruning") else g["rel_mse"]),
            experts_removed_share=experts_removed.get(r["run"]),
            grid=g and {k: g[k] for k in ("desc", "omitted", "coverage", "source", "rows_per_matrix") if k in g},
            main_class_bpw_stored=bpw.get(main_cls), nonmain_bpw={k: v for k, v in bpw.items() if k != main_cls and v},
            nonmain_gib=nonmain, main_gib=gib.get(main_cls), resident_gib=a.get("resident_gib"))
    out["run_grids"] = runs

    per_task = {}
    for t in sorted({v["task"] for v in runs.values()}):
        rs = [v for v in runs.values() if v["task"] == t and v["grid_rel_mse"] is not None and v["verdict_excess_nats"] is not None]
        if len(rs) < 3:
            continue
        x = [v["grid_rel_mse"] for v in rs]; y = [v["verdict_excess_nats"] for v in rs]
        inv = []                                              # pairs where lower weight MSE did NOT give lower excess
        for a, b in combinations(rs, 2):
            ka, kb = a["grid_effective_rel_err"], b["grid_effective_rel_err"]
            if (ka - kb) * (a["verdict_excess_nats"] - b["verdict_excess_nats"]) < 0:
                lo, hi = (a, b) if ka < kb else (b, a)
                inv.append(dict(lower_mse=lo["agent"], higher_mse=hi["agent"], effective_err=[lo["grid_effective_rel_err"], hi["grid_effective_rel_err"]],
                                excess=[lo["verdict_excess_nats"], hi["verdict_excess_nats"]],
                                nonmain_gib=[lo["nonmain_gib"], hi["nonmain_gib"]]))
        per_task[t] = dict(n=len(rs), spearman_mse_vs_excess=spearman(x, y),
                           spearman_effective_err_vs_excess=spearman([v["grid_effective_rel_err"] for v in rs], y),
                           spearman_nonmain_gib_vs_excess=spearman([v["nonmain_gib"] or 0 for v in rs], y),
                           inversions=inv, n_pairs=len(rs) * (len(rs) - 1) // 2)
    out["run_grids_vs_verdict"] = per_task

    # ---------------------------------------------------------------- interventions
    iv = {}
    for p in sorted(IV.glob("*.json")):
        d = json.load(open(p))
        iv[d["variant"]] = {k: d[k] for k in ("spec", "overall", "attention", "mlp", "per_class", "mean_linear_out_err",
                                               "mean_linear_rel_mse", "excess_ce", "excess_ce_ci95", "ppl", "bf16_ppl",
                                               "scored_tokens", "decoder_linear_bytes", "quantize_s", "eval_s")}
    out["interventions"] = dict(
        model="Qwen/Qwen3.8-27B @ 1d4bf0f2 (compression_bench/models/Qwen3.8-27B-bf16), HF transformers 5.14 BF16 forward, one H200",
        tokens="public PPL blocks of qwen3_8_27b_8g_standard_1xH200NVL (environment/public-checker/evaluator-assets/ppl_inputs.json, "
               "sha256 89a766d8...): 256 x 2048, scored positions 512..2047 = 393,216 tokens, as ppl_benchmark.py scores them",
        calibration="128 x 2048 FineWeb tokens (agent-image copy compression_bench/environments/agent-base/datasets/fineweb/000_00000.parquet), "
                    "documents in a seed-0 permutation, blank-line separated, packed; H = X^T X per linear on BF16 activations",
        quantized="all decoder linears (attention q/k/v/o, linear-attention in_proj_qkv/in_proj_z/out_proj, MLP gate/up/down), 64 layers; "
                  "embeddings, lm_head, norms, conv1d, in_proj_a/b, vision BF16",
        seeds="calibration permutation 0; Hadamard signs seed 0+K; k-means sample seed 0; bootstrap over the 256 blocks (2000, seed 0)",
        excess="mean over scored tokens of NLL_quant - NLL_bf16 (nats/token = ln PPL ratio); CI: block bootstrap",
        variants=iv)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out, len(runs), "runs,", len(iv), "variants")


if __name__ == "__main__":
    main()
