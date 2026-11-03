#!/usr/bin/env python3
"""Figures of design changes, quantization error, generation failures and reward-feedback search.

    python3 compression_bench/scripts/findings_figures.py [--data DIR] [--out DIR] [--chain RUN]
                                                          [--feedback-runs RUN,RUN,...] [FIGURE ...]   # default: all

Every figure uses the selected runs of the analysis directory DIR (default evaluation_progress/trajectory_analysis/)
and labels the measurement stage it shows: reward feedback (the public Checker's full check during design: PPL on the
public BF16-continued blocks) or final verification (the hidden evaluation after design). Stages are never joined
into one curve. Figures (names as given on the command line, and of the files written to --out, default DIR/figures):

  design_changes       within-run design changes by factor: the change in excess cross-entropy each one measured,
                       against the bytes it moved (DIR/design_pairs.json, design_pairs.py)
  quantization_error   PPL against weight and output error of the 3-bit variants of the controlled quantization
                       experiment (DIR/quant_error.json, studies/insights/quant_error_assemble.py)
  generation_failures  one run's successive final-verified versions (--chain, default the run with the most), the
                       items gained and lost between consecutive versions, and the behaviour of the wrong HumanEval
                       answers of every run that did not pass, beside its BF16 model's (DIR/generation_failures.json)
  feedback_search      the reward feedback of a few runs (--feedback-runs, default the four with the most checks),
                       every run's calls by design time, and what followed a worse call (DIR/version_ledger.json)

Figures are drawn at print size (COL_W one column, PAPER_W two).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_figures as A  # noqa: E402

AG_SHORT = {"sonnet": "Sonnet", "opus": "Opus", "glm": "GLM", "minimax": "MiniMax", "deepseek": "DeepSeek", "kimi": "Kimi",
            "glm-flash": "GLM Flash"}
OPTIONS = {"chain": None, "feedback_runs": None}


def load(name):
    path = A.DATA / name
    if not path.is_file():
        raise SystemExit(f"{path} is missing")
    return json.loads(path.read_text())


def utc(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def style():
    A.style()
    plt.rcParams.update({"font.size": 6.5, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
                         "legend.fontsize": 6, "axes.titlesize": 7, "axes.linewidth": 0.8, "xtick.major.width": 0.7,
                         "ytick.major.width": 0.7, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 0.9})


def save(fig, name):
    A.OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(A.OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", A.OUT / f"{name}.pdf")


def panel(ax, tag, size=7):
    """Panel letter only; what the panel shows is the caption's job."""
    ax.set_title(f"({tag})", loc="left", fontsize=size, fontweight="bold", pad=3)


def colour(agent):
    return A.C.get(agent, A.C["gray"])


def short(agent):
    return AG_SHORT.get(agent, agent)


def agents_of(runs):
    present = {r["agent"] for r in runs}
    return [a for a in A.AGENTS if a in present] + sorted(present - set(A.AGENTS))


# ---------------------------------------------------------------------------------------------------------------------
# Which design changes lower PPL under the same size limit (within-run pairs, reward feedback)
FACTOR_ORDER = ["levels & clipping", "rotation", "rounding", "scale storage", "bit allocation", "calibration data",
                "recovery", "parameter removal"]
FACTOR_SHORT = {"levels & clipping": "levels,\nclipping", "rotation": "rotation", "rounding": "rounding", "scale storage": "scale\nbits",
                "bit allocation": "bit\nalloc.", "calibration data": "calib.\ndata", "recovery": "recovery", "parameter removal": "removal"}


def pair_marker(ax, x, y, p, s=16):
    if p["decision"] == "reverted":
        ax.scatter(x, y, s=s * 1.3, marker="X", color=A.C["red"], linewidth=0, zorder=4)
    else:
        ax.scatter(x, y, s=s, marker="o" if p["same_measured_size"] else "D", color=colour(p["agent"]), edgecolor=A.C["ink"],
                   linewidth=0.35, zorder=4)


def fig_design_changes():
    pairs = load("design_pairs.json")["pairs"]
    factors = [f for f in FACTOR_ORDER if any(p["factor"] == f for p in pairs)]
    factors += sorted({p["factor"] for p in pairs} - set(factors))
    style()
    fig = plt.figure(figsize=(A.COL_W, 2.75))
    a = fig.add_axes([0.15, 0.53, 0.83, 0.405])
    b = fig.add_axes([0.15, 0.215, 0.83, 0.20])
    ys = [p["excess_ce_change_pct"] for p in pairs]
    lim = max([100] + [abs(y) * 1.05 for y in ys])
    for i, f in enumerate(factors):
        P = sorted((p for p in pairs if p["factor"] == f), key=lambda p: p["excess_ce_change_pct"])
        xs = np.linspace(i - 0.22, i + 0.22, len(P)) if len(P) > 1 else [i]
        for x, p in zip(xs, P):
            y = p["excess_ce_change_pct"]
            pair_marker(a, x, y, p)
            if p.get("label"):
                a.annotate(p["label"], (x, y), xytext=(5, 0 if y < 0 else -1), textcoords="offset points", fontsize=5.2,
                           va="center", ha="left", color=A.C["ink"], linespacing=0.95)
        n_low = sum(p["excess_ce_change_pct"] < 0 for p in P)
        a.text(i, lim * 1.04, f"{n_low}/{len(P)}", ha="center", va="bottom", fontsize=5.6, color=A.C["gray"])
    a.axhline(0, color=A.C["ink"], lw=0.6, zorder=1)
    a.set_xlim(-0.6, len(factors) - 0.4)
    a.set_ylim(-lim, lim)
    a.set_xticks(range(len(factors)))
    a.set_xticklabels([FACTOR_SHORT.get(f, f) for f in factors], fontsize=5.6, linespacing=0.95)
    a.tick_params(axis="x", length=0, pad=2)
    a.set_ylabel("excess-CE change (%)", labelpad=1)
    a.grid(axis="x", visible=False)
    a.set_title("(a)", loc="left", fontsize=6.8, fontweight="bold", pad=9)
    # (b) the same changes against the bytes they added or freed under the limit
    ds = [p["after"]["package_gib"] - p["before"]["package_gib"] for p in pairs]
    for d, p in zip(ds, pairs):
        pair_marker(b, d, p["excess_ce_change_pct"], p, s=13)
    b.axhline(0, color=A.C["ink"], lw=0.6, zorder=1)
    b.axvspan(-0.01, 0.01, color=A.C["panel"], zorder=0)
    span = max([0.2] + [abs(d) for d in ds])
    b.set_xlim(min([-0.15] + [d - 0.1 * span for d in ds]), max(d + 0.1 * span for d in ds + [0.0]))
    b.set_ylim(-lim, lim)
    b.set_xlabel("package size change (GiB)", labelpad=1)
    b.set_ylabel("excess-CE change (%)", labelpad=1)
    b.text(0.03, lim * 0.92, "same measured size", fontsize=5.2, color=A.C["gray"], va="top")
    b.set_title("(b)", loc="left", fontsize=6.8, fontweight="bold", pad=3)
    h = [Line2D([], [], marker="o", ls="", color=colour(k), markeredgecolor=A.C["ink"], markeredgewidth=0.3, markersize=3.4,
                label=short(k)) for k in agents_of(pairs)]
    h += [Line2D([], [], marker="D", ls="", color="white", markeredgecolor=A.C["ink"], markersize=2.9, markeredgewidth=0.5, label="size changed"),
          Line2D([], [], marker="X", ls="", color=A.C["red"], markersize=4, markeredgewidth=0, label="reverted")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.53, -0.01), ncol=4, frameon=False, fontsize=5.6,
               handletextpad=0.15, columnspacing=0.8)
    save(fig, "design_changes")


# ---------------------------------------------------------------------------------------------------------------------
# How reward feedback shapes the sequence of candidates
TOL = 0.002   # reward units: below the Checker's run-to-run spread on one unchanged package


def feedback_calls(run):
    t0 = utc(run["agent_window_utc"][0])
    calls, best = [], None
    for c in run["checks"]:
        r = c.get("public_reward")
        if not r or r < 0.5:       # a failed or degenerate check (the package did not run)
            continue
        h = (utc(c["utc"]) - t0).total_seconds() / 3600
        if best is None or r > best + TOL:
            kind = "new best"
            best = r
        elif r < best - TOL:
            kind = "regression"
        else:
            kind = "re-validation"
        calls.append({"h": h, "reward": r, "pct": 100 * (10 / r - 1), "best_pct": 100 * (10 / best - 1), "kind": kind,
                      "utc": c["utc"], "ppl": c["public_ppl"], "best_ppl": c["public_bf16_ppl"] * 10 / best})
    return calls


def budget_hours(run):
    start, end = run["agent_window_utc"]
    return (utc(end) - utc(start)).total_seconds() / 3600 if start and end else 8.0


KIND_STYLE = {"new best": dict(marker="o", s=11), "re-validation": dict(marker="o", s=7), "regression": dict(marker="v", s=12)}
KIND_COL = {"new best": "#0F4D92", "re-validation": "#CFCECE", "regression": "#B64342"}


def fig_feedback_search():
    led = load("version_ledger.json")
    runs = {r["run"]: r for r in led["runs"]}
    chosen = OPTIONS["feedback_runs"] or sorted(runs, key=lambda k: -len(feedback_calls(runs[k])))[:4]
    chosen = [k for k in chosen if k in runs and feedback_calls(runs[k])]
    hours = max([budget_hours(r) for r in led["runs"]] or [8.0])
    style()
    fig = plt.figure(figsize=(A.PAPER_W, 2.0))
    gs = fig.add_gridspec(1, 3, wspace=0.42, width_ratios=[1.25, 1.0, 0.85], left=0.065, right=0.99, top=0.89, bottom=0.27)
    # (a) the reward feedback of a few runs, each on its own linear PPL axis
    sub = gs[0, 0].subgridspec(max(len(chosen), 1), 1, hspace=0.25)
    ax = None
    for j, key in enumerate(chosen):
        ax = fig.add_subplot(sub[j])
        run = runs[key]
        calls = feedback_calls(run)
        col = colour(run["agent"])
        ax.step([c["h"] for c in calls] + [hours], [c["best_ppl"] for c in calls] + [calls[-1]["best_ppl"]], where="post",
                color=col, lw=0.9)
        for c in calls:
            face = col if c["kind"] != "re-validation" else "white"
            edge = A.C["red"] if c["kind"] == "regression" else (col if c["kind"] == "re-validation" else A.C["ink"])
            ax.scatter(c["h"], c["ppl"], s=KIND_STYLE[c["kind"]]["s"] * 0.8, marker=KIND_STYLE[c["kind"]]["marker"], facecolor=face,
                       edgecolor=edge, linewidth=0.6 if c["kind"] != "new best" else 0.3, zorder=3)
        lo, hi = min(c["ppl"] for c in calls), max(c["ppl"] for c in calls)
        pad = (hi - lo) * 0.22 + 1e-3
        ax.set_ylim(lo - pad, hi + pad * 2.2)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(2, steps=[1, 2, 2.5, 5, 10]))
        ax.tick_params(labelsize=5.2, pad=1)
        ax.set_xlim(0, hours)
        if j < len(chosen) - 1:
            ax.set_xticklabels([])
        ax.text(0.99, 0.93, f"{short(run['agent'])} {A.task_tag(run['model'], run['cap'])}", transform=ax.transAxes, fontsize=5.2,
                ha="right", va="top", color=col)
        if j == 0:
            panel(ax, "a", size=6.8)
    if ax is not None:
        ax.set_xlabel("design time (h)", labelpad=1)
    fig.text(0.012, 0.58, "PPL", rotation=90, ha="left", va="center", fontsize=6.5)
    # (b) every run: what reward feedback found, by quarter of the design time
    calls_all = [c for r in led["runs"] for c in feedback_calls(r)]
    ax = fig.add_subplot(gs[0, 1])
    edges = list(np.linspace(0, hours, 5)); edges[-1] += 0.01
    for k in range(4):
        Cs = [c for c in calls_all if edges[k] <= c["h"] < edges[k + 1]]
        left = 0
        for kind in ("new best", "re-validation", "regression"):
            n = sum(c["kind"] == kind for c in Cs)
            ax.barh(k, n, left=left, color=KIND_COL[kind], height=0.62, edgecolor="white", linewidth=0.4)
            if n >= 6:
                ax.text(left + n / 2, k, str(n), ha="center", va="center", fontsize=5.6,
                        color="white" if kind != "re-validation" else A.C["ink"])
            left += n
    ax.set_yticks(range(4))
    ax.set_yticklabels([f"{edges[k]:.0f}–{edges[k + 1]:.0f} h" for k in range(4)])
    ax.set_ylim(3.55, -0.55)
    ax.set_xlabel("calls", labelpad=1)
    ax.grid(axis="y", visible=False)
    panel(ax, "b", size=6.8)
    # (c) what followed a reward feedback worse than the best so far
    ax = fig.add_subplot(gs[0, 2])
    nxt = {"new best": 0, "restored best": 0, "worse again": 0, "run ended": 0}
    for r in led["runs"]:
        calls = feedback_calls(r)
        for i, c in enumerate(calls):
            if c["kind"] == "regression":
                n = calls[i + 1]["kind"] if i + 1 < len(calls) else None
                nxt[{"new best": "new best", "re-validation": "restored best", "regression": "worse again", None: "run ended"}[n]] += 1
    names = list(nxt)
    ax.barh(range(4), [nxt[k] for k in names], color=[KIND_COL["new best"], "#6E9BC8", KIND_COL["regression"], A.C["light"]], height=0.62)
    for i, k in enumerate(names):
        ax.text(nxt[k] + 0.3, i, str(nxt[k]), va="center", fontsize=5.8)
    ax.set_yticks(range(4))
    ax.set_yticklabels(["new best", "best restored", "worse again", "run ended"])
    ax.set_ylim(3.55, -0.55)
    ax.set_xlim(0, max(max(nxt.values()), 1) * 1.25)
    ax.set_xlabel("calls", labelpad=1)
    ax.grid(axis="y", visible=False)
    panel(ax, "c", size=6.8)
    h = [Line2D([], [], marker="o", ls="", color=KIND_COL["new best"], markeredgecolor=A.C["ink"], markeredgewidth=0.3, markersize=3.3, label="new best"),
         Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=A.C["gray"], markeredgewidth=0.6, markersize=3, label="same PPL as the best"),
         Line2D([], [], marker="v", ls="", color=KIND_COL["regression"], markeredgecolor=A.C["red"], markeredgewidth=0.6, markersize=3.5, label="worse than the best"),
         Line2D([], [], color=A.C["gray"], lw=0.9, label="best so far")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=False, fontsize=5.8,
               handletextpad=0.25, columnspacing=1.0)
    save(fig, "feedback_search")


# ---------------------------------------------------------------------------------------------------------------------
# Final verification of successive versions, item flips, and the behaviour of wrong HumanEval answers
BCOL = {b: c for b, _, c in A.BENCH}
BNAME = {b: n for b, n, _ in A.BENCH}
BEHAVIOR = {   # merged behaviour classes of generation_failures.json categories
    "code": [("repetition loop", ["repetition"], "#B64342"), ("truncated or empty", ["truncated", "empty", "early_exit", "timeout"], "#E9A6A1"),
             ("syntax or runtime error", ["syntax_error", "runtime_error"], "#9A4D8E"), ("format", ["tag_leak"], "#C0943A"),
             ("tests fail", ["tests_fail", "other_fail"], "#CFCECE")],
    "math": [("repetition loop", ["repetition"], "#B64342"), ("truncated or empty", ["truncated", "empty", "no_number", "unmarked_stop"], "#E9A6A1"),
             ("format", ["tag_leak"], "#C0943A"), ("wrong number", ["wrong_number"], "#CFCECE")],
}


def measured(run):
    return sorted((v for v in run["verdicts"] if v.get("measured") and v.get("benchmarks") and v["versions"][0] != "final"),
                  key=lambda v: int(v["versions"][0][1:]))


def fig_generation_failures():
    g = load("generation_failures.json")
    runs = {r["run"]: r for r in g["runs"]}
    if not runs:
        raise SystemExit("generation_failures.json holds no run")
    chain = OPTIONS["chain"] or max(runs, key=lambda k: len(measured(runs[k])))
    if chain not in runs:
        raise SystemExit(f"--chain {chain} is not in generation_failures.json")
    style()
    fig = plt.figure(figsize=(A.PAPER_W, 1.85))
    gs = fig.add_gridspec(1, 3, wspace=0.55, width_ratios=[1.0, 1.0, 1.05], left=0.06, right=0.99, top=0.88, bottom=0.30)
    # (a) one run's successive final-verified versions: PPL and the three benchmarks
    run = runs[chain]
    V = measured(run)
    bf = g["bf16"][run["model"]]["benchmarks"]
    a2 = fig.add_subplot(gs[0, 0])
    xs = range(len(V))
    for b_ in ("math", "code", "knowledge"):
        a2.plot(xs, [100 * v["benchmarks"][b_]["correct"] / v["benchmarks"][b_]["items"] for v in V], color=BCOL[b_],
                marker="o", ms=2.6, lw=1.0, zorder=3)
        a2.axhline(100 * bf[b_]["correct"] / bf[b_]["items"], color=BCOL[b_], lw=0.7, ls=(0, (3, 2)), alpha=0.9, zorder=2)
    a2.set_xticks(list(xs))
    a2.set_xticklabels(["=".join(v["versions"]) for v in V], fontsize=5.6)
    a2.set_xlim(-0.4, max(len(V) - 0.6, 0.4))
    a2.set_ylabel("accuracy (%)", labelpad=1)
    a2.set_xlabel(f"final-verified version ({short(run['agent'])} {A.task_tag(run['model'], run['cap'])})", labelpad=1)
    a2.set_ylim(0, 100)
    a1 = a2.twinx()
    a1.plot(xs, [v["hidden_ppl"] for v in V], color=A.C["ink"], marker="s", ms=2.4, lw=0.9, ls=(0, (1, 1)), zorder=4)
    a1.set_ylabel("PPL", labelpad=1)
    a1.spines["right"].set_visible(True)
    a1.grid(False)
    a1.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    a1.tick_params(labelsize=5.6, pad=1)
    a2.set_title("(a)", loc="left", fontsize=6.8, fontweight="bold", pad=3)
    # (b) every consecutive pair of final-verified versions: item flips vs PPL change
    ax = fig.add_subplot(gs[0, 1])
    dps = [t["hidden_ppl_to"] - t["hidden_ppl_from"] for r in g["runs"] for t in r["transitions"]
           if t.get("hidden_ppl_to") is not None and t.get("hidden_ppl_from") is not None]
    width = (max(dps) - min(dps)) if len(dps) > 1 else 0.1
    off = {"math": -0.007 * width, "code": 0.0, "knowledge": 0.007 * width}
    for r in g["runs"]:
        for t in r["transitions"]:
            if t.get("hidden_ppl_to") is None or t.get("hidden_ppl_from") is None:
                continue
            dp = t["hidden_ppl_to"] - t["hidden_ppl_from"]
            for b_, x in t["benchmarks"].items():
                up, dn = len(x["wrong_to_right"]), len(x["right_to_wrong"])
                ax.plot([dp + off[b_], dp + off[b_]], [-dn, up], color=BCOL[b_], lw=1.5, solid_capstyle="butt", alpha=0.85)
                ax.scatter(dp + off[b_], up - dn, s=7, color=BCOL[b_], edgecolor=A.C["ink"], linewidth=0.3, zorder=4)
    ax.axhline(0, color=A.C["ink"], lw=0.6)
    ax.set_xlabel("PPL difference between versions", labelpad=1)
    ax.set_ylabel("items lost (−) / gained (+)", labelpad=1)
    panel(ax, "b", size=6.8)
    # (c) behaviour of the wrong HumanEval answers: every run that did not pass, under its BF16 model
    ax = fig.add_subplot(gs[0, 2])
    failed = [r for r in g["runs"] if r.get("final_status") != "SCORED"
              and any(v["is_final"] and v.get("measured") for v in r["verdicts"])]
    rows = []
    for m in [m for m in A.MODEL_ORDER if any(r["model"] == m for r in failed)] + \
             sorted({r["model"] for r in failed} - set(A.MODEL_ORDER)):
        if m in g["bf16"] and g["bf16"][m].get("benchmarks"):
            rows.append(("bf16", m, g["bf16"][m]["benchmarks"]["code"]["categories"]))
        for r in sorted((r for r in failed if r["model"] == m), key=lambda r: (r["cap"], r["agent"])):
            final = next(v for v in r["verdicts"] if v["is_final"] and v.get("measured"))
            rows.append((r["agent"], r, final["benchmarks"]["code"]["categories"]))
    labels = []
    for i, (who, what, cats) in enumerate(rows):
        labels.append(f"BF16 {A.MODEL_SHORT.get(what, what)}" if who == "bf16" else f"{short(who)} {A.task_tag(what['model'], what['cap'])}")
        left = 0
        for name, keys, col in BEHAVIOR["code"]:
            n = sum(cats.get(c, 0) for c in keys)
            if n:
                ax.barh(i, n, left=left, color=col, height=0.66, edgecolor="white", linewidth=0.3)
                if n >= 5:
                    ax.text(left + n / 2, i, str(n), ha="center", va="center", fontsize=5.0,
                            color="white" if col in ("#B64342", "#9A4D8E") else A.C["ink"])
            left += n
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=5.6)
    for i, row in enumerate(rows):
        if row[0] == "bf16":
            ax.get_yticklabels()[i].set_color(A.C["gray"])
    ax.set_ylim(len(rows) - 0.45, -0.55)
    ax.set_xlim(0, 50)
    ax.set_xlabel("wrong HumanEval items of 50", labelpad=1)
    ax.grid(axis="y", visible=False)
    panel(ax, "c", size=6.8)
    h = [Line2D([], [], color=BCOL[b_], marker="o", ms=2.6, lw=1.0, label=BNAME[b_]) for b_ in ("math", "code", "knowledge")]
    h.append(Line2D([], [], color=A.C["gray"], lw=0.7, ls=(0, (3, 2)), label="BF16, same items"))
    h.append(Line2D([], [], color=A.C["ink"], marker="s", ms=2.4, lw=0.9, ls=(0, (1, 1)), label="PPL (right axis)"))
    h += [matplotlib.patches.Patch(color=c, label=n) for n, _, c in BEHAVIOR["code"]]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=len(h), frameon=False, fontsize=5.5,
               handlelength=1.1, handletextpad=0.25, columnspacing=0.6)
    save(fig, "generation_failures")


# ---------------------------------------------------------------------------------------------------------------------
# Direct evidence about quantization error (quant_error.json, the controlled quantization experiment)
CHANGE_KIND = [   # (label, colour, marker, predicate on a variant's spec), first match wins
    ("rotation + GPTQ", "#0B2F5E", "*", lambda sp: sp.get("rot") and sp.get("gptq")),
    ("GPTQ rounding", "#0F4D92", "o", lambda sp: sp.get("gptq")),
    ("Hadamard rotation", "#6E9BC8", "o", lambda sp: sp.get("rot")),
    ("bits moved, attention ↔ MLP", "#D9822B", "D", lambda sp: sp.get("alloc")),
    ("round-to-nearest grids", "#A8A8A8", "o", lambda sp: True),
]


def change_kind(spec):
    return next(k for k in CHANGE_KIND if k[3](spec))


def fig_quantization_error():
    q = load("quant_error.json")["interventions"]
    V = {k: v for k, v in q["variants"].items() if v["spec"]["bits"] == 3}
    if not V:
        raise SystemExit("quant_error.json holds no 3-bit variant")
    bf = next(iter(V.values()))["bf16_ppl"]
    style()
    fig, axes = plt.subplots(1, 2, figsize=(A.COL_W, 1.75), sharey=True,
                             gridspec_kw={"wspace": 0.12, "left": 0.14, "right": 0.985, "top": 0.9, "bottom": 0.40})
    for ax, (key, xlab) in zip(axes, (("rel_mse", "weight error (%)"), ("out_err", "output error (%)"))):
        for v in V.values():
            lab, col, mk, _ = change_kind(v["spec"])
            ax.scatter(100 * v["overall"][key], v["ppl"], s=34 if mk == "*" else 18, marker=mk, color=col, edgecolor=A.C["ink"],
                       linewidth=0.35, zorder=4)
        ax.axhline(bf, color=A.C["gray"], lw=0.6, ls=(0, (1, 1.5)), zorder=1)
        ax.set_xlim(left=0)
        ax.set_xlabel(xlab, labelpad=1)
    axes[0].set_ylabel("PPL", labelpad=1)
    panel(axes[0], "a")
    panel(axes[1], "b")
    h = [Line2D([], [], marker=mk, ls="", color=col, markeredgecolor=A.C["ink"], markeredgewidth=0.3, markersize=5 if mk == "*" else 3.6,
                label=lab) for lab, col, mk, _ in reversed(CHANGE_KIND)]
    h.append(Line2D([], [], color=A.C["gray"], lw=0.6, ls=(0, (1, 1.5)), label="BF16"))
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.53, -0.02), ncol=3, frameon=False, fontsize=5.5,
               handletextpad=0.2, columnspacing=0.7)
    save(fig, "quantization_error")


FIGS = {"design_changes": fig_design_changes, "quantization_error": fig_quantization_error,
        "generation_failures": fig_generation_failures, "feedback_search": fig_feedback_search}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("figures", nargs="*", metavar="FIGURE", help="default: every figure")
    parser.add_argument("--data", type=Path, default=A.DATA, help="the analysis directory")
    parser.add_argument("--out", type=Path, help="where the figures go (default DATA/figures)")
    parser.add_argument("--chain", help="generation_failures (a): the run whose versions are drawn")
    parser.add_argument("--feedback-runs", help="feedback_search (a): comma-separated run ids")
    args = parser.parse_args(argv)
    unknown = [f for f in args.figures if f not in FIGS]
    if unknown:
        parser.error(f"unknown figure(s) {unknown}; known: {', '.join(FIGS)}")
    A.configure(args.data, args.out)
    OPTIONS.update(chain=args.chain, feedback_runs=args.feedback_runs.split(",") if args.feedback_runs else None)
    for name in args.figures or list(FIGS):
        FIGS[name]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
