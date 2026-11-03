#!/usr/bin/env python3
"""Compare two inference implementations of the same compressed weights within one run, and draw the comparison.

    inference_implementation.py extract --lineage ID --run RUN --baseline vA --variant vB [--out DIR]
                                        [--kernel-file PATH] [--label-baseline TEXT] [--label-variant TEXT]
                                        [--subtitle TEXT]
    inference_implementation.py figure [--out DIR]
    inference_implementation.py [all] ...              # extract, then figure (the default)

The two versions must store identical weight files (SHA-256 from their MANIFEST.json); every file that differs is
listed. Timings are the design-time public full checks whose utc_start equals each version's capture time (not
the fastest observed), full-model batch-one decode, not matrix microbenchmarks. Quality is each version's
checkpoint verdict (CHECKPOINT_VERDICTS/<version>). --kernel-file (default qwen35_lowbit/kernels.py, relative to
the package) is scanned for its module-level *SLOTS* / DENSE_GEMV_MAX_M dispatch constants. Writes to --out
(default evaluation_progress/trajectory_analysis/inference_implementation). No model is quantized, trained, or searched.
"""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import statistics
import textwrap

from common import REPO, TRAJECTORY_ANALYSIS, file_sha256 as sha, read_json as read, run_dir, write_json

DEFAULT_OUT = TRAJECTORY_ANALYSIS / "inference_implementation"


def validate_timing(d):
    """Check one public check's decode record against the evaluator's protocol and recompute its statistics."""
    assert d["valid"] and d["complete"]
    assert d["measurement_protocol"] == "submitted-decode-single-token-3x3-v3"
    assert (d["input_tokens"], d["output_tokens"], d["concurrency"], d["warmup_requests"]) == (2048, 128, 1, 1)
    rows = d["requests"]
    assert len(rows) == 9 and {(r["round"], r["request"]) for r in rows} == {(i, j) for i in range(3) for j in range(3)}
    for r in rows:
        assert len(r["arrival_seconds"]) == 128 and r["chunk_sizes"] == [1]*128
        assert r["decode_intervals"] == 127
        assert abs(r["decode_seconds"] - (r["arrival_seconds"][-1] - r["arrival_seconds"][0])) < 1e-9
    means = [statistics.mean(r["decode_seconds"] for r in rows if r["round"] == i) for i in range(3)]
    assert all(abs(a-b) < 1e-9 for a, b in zip(means, d["round_mean_seconds"]))
    assert abs(statistics.median(means) - d["median_round_mean_seconds"]) < 1e-9
    return {"decode_seconds": statistics.median(means), "tps": 127/statistics.median(means),
            "round_mean_seconds": means, "round_tps": [127/m for m in means],
            "min_round_tps": min(127/m for m in means), "max_round_tps": max(127/m for m in means)}


def extract(a):
    out = Path(a.out)
    lineage = REPO / "compression_output" / a.lineage
    run = run_dir(a.run)
    identity = read(run / "identity.json")
    version_records = list(map(json.loads, (lineage/"versions.jsonl").read_text().splitlines()))
    versions = {v["version"]: lineage/v["name"] for v in version_records}
    captured = {v["version"]: v["captured_utc"] for v in version_records}
    pair = (a.baseline, a.variant)
    manifests = {v: {e["path"]: {"size": e["size"], "sha256": e["sha256"]} for e in read(versions[v]/"MANIFEST.json") if e.get("type") == "file"}
                 for v in pair}
    weights = {k: v for k, v in manifests[a.baseline].items() if k.endswith(".safetensors")}
    assert weights and {k: d for k, d in manifests[a.variant].items() if k.endswith(".safetensors")} == weights, "weights differ"
    changed = sorted(k for k in set(manifests[a.baseline]) | set(manifests[a.variant])
                     if manifests[a.baseline].get(k) != manifests[a.variant].get(k))
    config = read(versions[a.baseline]/"files/model/config.json")
    write_json(out / "identity.json", {"run": a.run, "lineage": a.lineage, "task": identity["task_id"], "identity_basis": "SHA-256 manifests recorded at version capture",
          "weight_files": weights, "weight_bytes": sum(w["size"] for w in weights.values()),
          "changed_files": changed,
          "versions": {v: {"manifest": str(versions[v].relative_to(REPO)/"MANIFEST.json"), "manifest_sha256": sha(versions[v]/"MANIFEST.json")} for v in pair},
          "precision_allocation_sha256": hashlib.sha256(json.dumps(config.get("quantization_config"), sort_keys=True).encode()).hexdigest()})
    checks = [json.loads(line) for line in (run/"public-checks/checks.jsonl").read_text().splitlines()]
    labels = {a.baseline: a.label_baseline or f"{a.baseline} implementation", a.variant: a.label_variant or f"{a.variant} implementation"}
    timings, archived = [], []
    common_device = None
    for v in pair:
        utc = captured[v]
        i, check = next((i, c) for i, c in enumerate(checks) if c["utc_start"] == utc)
        sizes = dict(check["submission"])
        assert all(sizes[k] == manifests[v][k]["size"] for k in sizes if k in manifests[v])
        d = check["response"]["decode"]
        common_device = common_device or d["device"]
        assert d["device"] == common_device
        timing = {"version": v, "label": labels[v], "check_utc": utc, **validate_timing(d)}
        timings.append(timing)
        archived.append({"version": v, "check_utc": utc, "source_line_1_based": i+1,
                         "source": str((run/"public-checks/checks.jsonl").relative_to(REPO)),
                         "decode": {k: value for k, value in d.items() if k != "model_memory"},
                         "resident_storage_gate": {k: d["model_memory"][k] for k in ("protocol", "passed", "max_resident_bytes")},
                         "public_ppl": check["response"]["public_ppl"]})
    write_json(out / "selected_timings.json", archived)
    correctness = {}
    for v in pair:
        vd = lineage/"CHECKPOINT_VERDICTS"/v
        ppl, accuracy = read(vd/"ppl.json"), read(vd/"accuracy.json")
        correctness[v] = {"ppl": ppl["ppl"], "nll": ppl["nll"], "predicted_tokens": ppl["predicted_tokens"],
            "benchmarks": accuracy["benchmarks"], "accuracy": accuracy["accuracy"],
            "generation_hashes": [hashlib.sha256(s.encode()).hexdigest() for s in accuracy["generations"]],
            "sources": {n: {"path": str((vd/n).relative_to(REPO)), "sha256": sha(vd/n)} for n in ("ppl.json", "accuracy.json")}}
    x, y = correctness[a.baseline], correctness[a.variant]
    write_json(out / "correctness.json", {"versions": correctness,
          "same_nll": x["nll"] == y["nll"], "same_benchmark_counts": x["benchmarks"] == y["benchmarks"],
          "identical_generated_strings": sum(p == q for p, q in zip(x["generation_hashes"], y["generation_hashes"])),
          "generation_items": len(x["generation_hashes"]),
          "note": "equal PPL and benchmark counts do not imply bitwise generation equivalence; raw hidden generations are not copied"})
    task = REPO/"compression_tasks"/identity["task_id"]
    assets = task/"environment/public-checker/evaluator-assets"
    write_json(out / "controls.json", {"timing_source": "design-time public full checks, decode excluded from reward",
          "selection_rule": "the public full check whose utc_start equals each version's capture time; not selected by fastest observed time",
          "hardware": common_device,
          "decode_config": read(assets/"decode_config.json"),
          "decode_config_sha256": sha(assets/"decode_config.json"),
          "decode_inputs_sha256": sha(assets/"decode_inputs.json"),
          "limitations": ["one session per implementation; round range is descriptive variability, not a confidence interval",
                          "physical KV allocation is auto-sized by the runtime and not recorded by public checks; cache-budget and runtime settings, not physical allocation, are matched",
                          "verdict timings on other cards are excluded"]})
    dispatch = {}
    for v in pair:
        path = versions[v]/"files"/a.kernel_file
        constants = {}
        if path.exists():
            for node in ast.parse(path.read_text()).body:
                if isinstance(node, ast.Assign):
                    for name in node.targets:
                        if isinstance(name, ast.Name) and ("SLOTS" in name.id or name.id == "DENSE_GEMV_MAX_M"):
                            constants[name.id] = ast.literal_eval(node.value)
        dispatch[v] = {"kernel": a.kernel_file, "kernel_sha256": sha(path) if path.exists() else None, "constants": constants}
    write_json(out / "dispatch.json", {"implementations": dispatch,
          "batch_one_routed_slots": config.get("num_experts_per_tok") or config.get("text_config", {}).get("num_experts_per_tok")})
    gain = 100*(timings[1]["tps"] / timings[0]["tps"] - 1)
    within = timings[1]["min_round_tps"] <= timings[0]["max_round_tps"] and timings[0]["min_round_tps"] <= timings[1]["max_round_tps"]
    write_json(out / "summary.json", {"timings": timings, "tps_change_percent": gain, "round_ranges_overlap": within,
          "weight_files": len(weights), "task": identity["task_id"], "subtitle": a.subtitle or identity["task_id"]})
    with (out/"timings.csv").open("w", newline="") as f:
        keys = ("version", "label", "check_utc", "decode_seconds", "tps", "min_round_tps", "max_round_tps")
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore", lineterminator="\n")
        w.writeheader(); w.writerows(timings)


def figure(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    out = Path(a.out)
    data = read(out/"summary.json")
    rows = data["timings"]
    quality = read(out/"correctness.json")
    families = {f.name for f in font_manager.fontManager.ttflist}
    family = "Arial" if "Arial" in families else "Liberation Sans"
    plt.rcParams.update({"font.family": family, "font.size": 7, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False})
    fig = plt.figure(figsize=(3.25, 3.2), facecolor="white")
    fig.text(.04, .955, "Same compressed weights, different kernels", fontsize=8.4, weight="bold", color="#222222")
    fig.text(.04, .898, data["subtitle"], fontsize=7.2, color="#666666")
    ax = fig.add_axes([.31, .605, .63, .23])
    top = max(r["max_round_tps"] for r in rows) * 1.25
    for i, (row, color) in enumerate(zip(rows, ("#3775BA", "#0F4D92"))):
        y = 1-i
        ax.hlines(y, 0, row["tps"], color=color, lw=4, alpha=.22)
        ax.errorbar(row["tps"], y, xerr=[[row["tps"]-row["min_round_tps"]], [row["max_round_tps"]-row["tps"]]], fmt="o", ms=4.3,
                    color=color, elinewidth=.9, capsize=2.5, zorder=3)
        ax.text(row["tps"]+top*.02, y, f"{row['tps']:.2f}", va="center", fontsize=7.2, weight="bold", color=color)
        ax.text(top/2, y-.29, f"rounds {row['min_round_tps']:.2f}–{row['max_round_tps']:.2f}", va="center", ha="center", fontsize=6.2, color="#666666")
    ax.set_yticks([1, 0], [f"{r['version']}\n{r['label']}" for r in rows], fontsize=7)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.set_xlim(0, top); ax.set_ylim(-.55, 1.55)
    ax.tick_params(axis="x", labelsize=6.5, length=2)
    ax.set_xlabel("Full-model decode (tokens/s)", fontsize=7, labelpad=2)
    ax.spines['bottom'].set_color('#B4B4B4');ax.spines['bottom'].set_linewidth(.6)
    v = [quality["versions"][r["version"]] for r in rows]
    items = quality["generation_items"]
    correct = [sum(b["correct"] for b in q["benchmarks"].values()) for q in v]
    fig.text(.04, .455, f"{data['weight_files']}/{data['weight_files']} weight hashes match", fontsize=6.7, color="#5F4596")
    fig.text(.04, .414, f"Verdict PPL: {v[0]['ppl']:.5f} vs {v[1]['ppl']:.5f} · accuracy: {correct[0]}/{items} vs {correct[1]}/{items}",
             fontsize=6.7, color="#3E4C61")
    caption = ("Public full checks on one card: batch 1, 2,048 input and 128 output tokens; one warmup then three rounds of three requests. "
               "Dots are 127/median round-mean latency; whiskers span round TPS (not confidence intervals). "
               f"TPS change {data['tps_change_percent']:+.2f}%; round ranges {'overlap' if data['round_ranges_overlap'] else 'do not overlap'}. "
               "Matching quality metrics do not imply identical generated text. Matrix microbenchmarks and batched throughput are excluded.")
    fig.text(.04, .36, textwrap.fill(caption, width=65), va="top", fontsize=6.5, linespacing=1.15, color="#222222")
    for ext in ("pdf", "svg", "png"):
        path = out/f"inference_implementation.{ext}"
        fig.savefig(path, dpi=300, pad_inches=.02)
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
    plt.close(fig)
    (out/"caption.txt").write_text(caption+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("extract", "figure", "all"), default="all", nargs="?")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--lineage")
    p.add_argument("--run")
    p.add_argument("--baseline")
    p.add_argument("--variant")
    p.add_argument("--kernel-file", default="qwen35_lowbit/kernels.py")
    p.add_argument("--label-baseline")
    p.add_argument("--label-variant")
    p.add_argument("--subtitle", help="figure subtitle (default: the task id)")
    a = p.parse_args()
    if a.command in ("extract", "all"):
        missing = [n for n in ("lineage", "run", "baseline", "variant") if not getattr(a, n)]
        if missing:
            p.error("extract needs --" + ", --".join(missing))
        extract(a)
    if a.command in ("figure", "all"):
        figure(a)
