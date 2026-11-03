#!/usr/bin/env python3
"""Figures of the trajectory analysis: outcome, design, allocation, correctness, reading and budget use per run.

    python3 compression_bench/scripts/analysis_figures.py [--data DIR] [--out DIR] [--families FILE] [--released FILE]
                                                          [FIGURE ...]      # default: every figure

Inputs, all in DIR (default evaluation_progress/trajectory_analysis/), written by the analysis scripts:
features.json (trajectory_features.py; the figures draw its `selected` runs), byte_allocation.json (byte_allocation.py),
search_paths.json (search_paths.py), paper_usage.json (paper_usage.py), gate_accuracy.json (gate_accuracy.py),
insights_hidden.json (insight_data.py); plus each selected run's trajectory (TIMING.json, resources.jsonl.gz, the
lineage's versions and METRICS.json). Figures are written as PDF and PNG to --out (default DIR/figures). Needs
matplotlib and numpy.

Optional annotations:
  --families FILE  the bulk quantizer family of each run's final design, as read from its notes and code:
                   {"<run id>": {"family": "vq" | "rot" | "scalar" | "lut", "removal": "" | "mild" | "heavy"}}
                   (vq: trellis / vector codebook; rot: scalar grid with a Hadamard rotation; scalar: group-wise
                   RTN / GPTQ grid; lut: lookup table with rotation; removal: experts or MLP width removed, heavy
                   for a fifth or more). Without it every run is drawn with one marker.
  --released FILE  released_baseline.py's RESULTS.json (default evaluation_progress/baselines/released/RESULTS.json):
                   each task's released quantization, drawn as a reference point where it was measured.

Figures (names as given on the command line, and of the files written):
  commitment       package size at the first and last valid check; the public reward after the first valid model,
                   by trajectory pattern, coloured by the run's outcome on its task
  representation   hidden PPL above BF16 of every run under each size limit, marked by quantizer family
  prior_knowledge  papers opened per run, when in the run they were read, and what kind of paper
  allocation       resident bytes by component, expert vs non-expert bits, and a dense model's bits per component
  correctness      multi-benchmark gate accuracy against hidden PPL, overall and per benchmark
  budget_timeline  every run's GPU use, first stop and weight changes over its budget
  budget_outcome   GPU use, first stop and tool calls against the gap to the task's best run; budget left at first stop
  code_blocks      excess cross-entropy on code-seeded hidden blocks; HumanEval accuracy against PPL at the tighter limits
  bits_vs_ppl      byte split and bits per bulk weight against hidden PPL

Hidden-set quality is drawn as perplexity above BF16's; the public reward appears only where it is what the agent saw.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_common import BASELINES, DATA, MODEL_KEY, run_dir, trajectory_dir, utc_seconds  # noqa: E402

OUT = DATA / "figures"
CATALOG = Path(__file__).resolve().parent / "paper_catalog.json"
RELEASED = BASELINES / "released" / "RESULTS.json"
FAMILIES: dict = {}

# Flat palette: one colour per agent, red for failure, neutral grays.
C = {"sonnet": "#0F4D92", "glm": "#9A4D8E", "minimax": "#42949E", "deepseek": "#8BCF8B", "glm-flash": "#B7B7B7",
     "opus": "#D9822B", "kimi": "#8C5A3C", "ink": "#272727", "gray": "#767676", "light": "#CFCECE", "red": "#B64342", "green": "#3F9142",
     "indigo": "#0F4D92", "blue2": "#3775BA", "violet": "#9A4D8E", "teal": "#42949E", "tan": "#E9A6A1", "pink": "#F6CFCB",
     "lgreen": "#8BCF8B", "panel": "#F2F4F7"}
SPARE = ["#5C6BC0", "#C2185B", "#7CB342", "#00897B", "#6D4C41", "#F9A825"]   # colours for agents not listed here
AGENT = {"sonnet": "Sonnet 5.5", "glm": "GLM 5.3", "minimax": "MiniMax M3.1", "deepseek": "DeepSeek V4.1", "glm-flash": "GLM 5.3 Flash",
         "opus": "Opus 5.5", "kimi": "Kimi K3.1"}
AGENTS = ("sonnet", "opus", "glm", "minimax", "deepseek", "kimi")   # plotting order of the known agents
SHORT_AGENT = {"sonnet": "Son-\nnet", "opus": "Opus", "glm": "GLM", "minimax": "Mini-\nMax", "deepseek": "Deep-\nSeek", "kimi": "Kimi",
               "glm-flash": "GLM\nFlash"}
MODEL = {"nemotron": "Nemotron-3.5-Super 120B-A12B", "qwen122": "Qwen3.5-122B-A10B", "qwen8": "Qwen3.8-27B", "gemma": "Gemma-4 26B"}
PARAMS = {"nemotron": 124.45e9, "qwen122": 125.09e9, "qwen8": 27.78e9, "gemma": 25.81e9}   # BF16 checkpoint parameters
FAMILY_NAME = {"vq": "trellis / vector codebook", "rot": "scalar + Hadamard rotation", "scalar": "scalar group-wise (RTN / GPTQ)",
               "lut": "lookup table + rotation", "unknown": "family not annotated"}
FAMILY_MARK = {"vq": "*", "rot": "D", "scalar": "o", "lut": "s", "unknown": "o"}
PAPER_W = 6.75   # inches: a two-column figure, so its font sizes are the printed sizes
COL_W = 3.25     # inches: a one-column figure

SHORT_TASK = {"nemotron": "Nemotron", "qwen122": "Qwen3.5-122B", "qwen8": "Qwen3.8-27B", "gemma": "Gemma-4"}
SHORT = {"nemotron": "Nemo.", "qwen122": "Q3.5", "qwen8": "Q3.8", "gemma": "Gem."}
MODEL_ORDER = ["nemotron", "qwen122", "qwen8", "gemma"]
# Where a figure distinguishes the source models (their BF16 accuracy references), and the three hidden benchmarks.
MODEL_C = {"nemotron": "#4F8F5B", "qwen122": "#7E5AA6", "qwen8": "#A0612E", "gemma": "#3D7EA6"}
MODEL_MARK = {"nemotron": "^", "qwen122": "s", "qwen8": "o", "gemma": "v"}
MODEL_SHORT = {"nemotron": "Nemotron", "qwen122": "Qwen3.5", "qwen8": "Qwen3.8", "gemma": "Gemma-4"}
BENCH = [("math", "GSM8K", "#3D5A98"), ("code", "HumanEval", "#C96B63"), ("knowledge", "TriviaQA", "#C0943A")]
REF = dict(marker="P", s=30, facecolor="white", edgecolor=C["ink"], linewidth=0.8, zorder=6)

# Filled by load(): the agents, models and (model, capacity) tasks of the selected runs, in plotting order.
ACTIVE: list[str] = list(AGENTS)
ORDER: list[str] = []
TASKS: list[tuple[str, float]] = []
FAMILY_ROWS: list[dict] = []


def style() -> None:
    plt.rcParams.update({
        "font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 8, "axes.linewidth": 1.4, "xtick.major.width": 1.2, "ytick.major.width": 1.2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": C["ink"], "axes.labelcolor": C["ink"],
        "xtick.color": C["ink"], "ytick.color": C["ink"], "legend.frameon": False, "svg.fonttype": "none",
        "pdf.fonttype": 42, "axes.titleweight": "bold", "axes.grid": True, "grid.color": "#E6E9EF", "grid.linewidth": 0.6, "axes.axisbelow": True, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7, "axes.labelsize": 8, "axes.titlesize": 8.5, "figure.dpi": 150,
    })


def load():
    """features.json with derived fields; sets the agents, models and tasks the figures draw."""
    global ACTIVE, ORDER, TASKS
    rows = json.loads((DATA / "features.json").read_text())
    selected = [r for r in rows if r["group"] == "selected"]
    if not selected:
        raise SystemExit(f"no selected run in {DATA / 'features.json'}")
    spare = iter(SPARE)
    for r in selected:
        a = r["agent"]
        if a not in C:
            C[a] = next(spare, C["gray"])
        AGENT.setdefault(a, a.rsplit("/", 1)[-1])
        SHORT_AGENT.setdefault(a, AGENT[a])
    present = {r["agent"] for r in selected}
    ACTIVE = [a for a in AGENTS if a in present] + sorted(present - set(AGENTS))
    models = {r["model"] for r in selected}
    ORDER = [m for m in MODEL_ORDER if m in models] + sorted(models - set(MODEL_ORDER))
    for m in ORDER:
        MODEL.setdefault(m, m); SHORT.setdefault(m, m); SHORT_TASK.setdefault(m, m); MODEL_SHORT.setdefault(m, m)
        MODEL_C.setdefault(m, C["gray"]); MODEL_MARK.setdefault(m, "o")
    tight_cap = {m: min(r["cap_gib"] for r in selected if r["model"] == m) for m in ORDER}
    TASKS = sorted({(r["model"], r["cap_gib"]) for r in selected},
                   key=lambda t: (t[1] != tight_cap[t[0]], ORDER.index(t[0]), t[1]))
    for r in rows:
        r["bpw"] = (r["package_gib"] * 8 * 2 ** 30 / PARAMS[r["model"]]) if r.get("package_gib") and r["model"] in PARAMS else None
        r["zero"] = r["reward"] == 0
        r["tight"] = r["model"] in tight_cap and r["cap_gib"] == tight_cap[r["model"]]
    FAMILY_ROWS[:] = selected
    return rows


def family(r) -> tuple[str, str]:
    """(quantizer family, removal) of a run's final design, from --families; ("unknown", "") when not annotated."""
    got = FAMILIES.get(r["run"]) or {}
    fam = got.get("family", "unknown")
    return (fam if fam in FAMILY_MARK else "unknown"), got.get("removal") or ""


def family_handles(markersize=4.5):
    seen = sorted({family(r)[0] for r in FAMILY_ROWS}, key=list(FAMILY_MARK).index)
    return [Line2D([], [], marker=FAMILY_MARK[k], ls="", color=C["gray"], markersize=markersize * (1.35 if k == "vq" else 1),
                   label=FAMILY_NAME[k]) for k in seen]


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def panel(ax, tag: str, size: float = 8.5) -> None:
    """Caption-style panel heading, left-aligned: "(a) short description"."""
    t = ax.get_title()
    ax.set_title("", loc="center")
    ax.set_title(f"({tag}) {t}" if t else f"({tag})", loc="left", fontsize=size, fontweight="bold", pad=5)


def flush_titles(fig, *axes) -> None:
    """Move each panel's left title out to the left edge of its y-axis label, so a long title has the panel's full width."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax in axes:
        x0 = ax.yaxis.get_tightbbox(r).x0
        ax0 = ax.get_window_extent(r)
        ax._left_title.set_x((x0 - ax0.x0) / ax0.width)   # the title set with loc="left"; its size and height stay


def top_legend(fig, handles, ncol=None, y=None, room=0.6, **kw) -> None:
    """The legend sits under the panels, in as few rows as fit; `room` (inches) is left for the x tick labels and title."""
    ncol = ncol or len(handles)
    rows = -(-len(handles) // ncol)
    h = fig.get_size_inches()[1]
    fig.subplots_adjust(bottom=(room + 0.15 * rows) / h)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=ncol, frameon=False,
               columnspacing=1.2, handletextpad=0.3, **kw)


def agent_handles(zero=True, label="not passed"):
    h = [Line2D([], [], marker="o", ls="", color=C[k], markeredgecolor=C["ink"], markeredgewidth=0.5, markersize=4.5, label=AGENT[k])
         for k in ACTIVE]
    if zero:
        h.append(zero_handle(label))
    return h


def red_edge_handle(label="red edge: not passed"):
    return Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=C["red"], markeredgewidth=0.5, markersize=4.5, label=label)


def zero_handle(label="not passed"):
    return Line2D([], [], marker="X", ls="", color=C["light"], markeredgecolor=C["red"], markeredgewidth=0.5, markersize=5, label=label)


def dot(ax, x, y, r, s=18, **kw):
    """One run: a dot in its agent's colour; an X with a thin red edge if it did not pass."""
    if r["zero"]:
        ax.scatter(x, y, s=s * 1.4, marker="X", color=C[r["agent"]], edgecolor=C["red"], linewidth=0.5, zorder=4, **kw)
    else:
        ax.scatter(x, y, s=s, marker="o", color=C[r["agent"]], edgecolor=C["ink"], linewidth=0.5, zorder=3, **kw)


def sweep_rows(rows):
    return [r for r in rows if r["group"] == "selected"]


def ppl_up(f: float) -> float:
    """Hidden-set perplexity above BF16's, in %, from fidelity F = PPL_BF16 / PPL."""
    return 100 * (1 / f - 1)


def task_tag(m, cap) -> str:
    return f"{SHORT.get(m, m)} {cap:g}G"


def swarm(ys, d, xin, yin):
    """Horizontal offsets (data units) that keep dots of diameter `d` inches apart; xin, yin: inches per data unit."""
    placed, off = [], {}
    for i in sorted(range(len(ys)), key=lambda i: ys[i]):
        y = ys[i] * yin
        for k in range(200):
            x = (k + 1) // 2 * d / 4 * (1 if k % 2 else -1)
            if all((x - px) ** 2 + (y - py) ** 2 >= d * d for px, py in placed):
                break
        placed.append((x, y)); off[i] = x / xin
    return [off[i] for i in range(len(ys))]


def log_limits(values, lo_pad=0.8, hi_pad=1.25, floor=0.1):
    """Limits of a log axis that shows every positive value."""
    v = [x for x in values if x and x > 0]
    return (max(min(v) * lo_pad, floor), max(v) * hi_pad) if v else (floor, 100)


def plain_log_ticks(ax, axis="x"):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)
    ax.minorticks_off()


# --------------------------------------------------------------------------------------------------------------------
#: Outcome classes of a run on its task.
OUTCOME = [("best", "best on the task", "#3D5A98"), ("close", "PPL within 5% of the best", "#5FA37F"),
           ("behind", "further behind", "#C0943A"), ("zero", "not passed", "#C96B63")]


def outcome_of(r, gap) -> str:
    if r["zero"]:
        return "zero"
    return "best" if gap[r["run"]] <= 1e-9 else ("close" if gap[r["run"]] < 5 else "behind")


def fig_commitment(rows) -> None:
    """Package size at the first and last valid check; the public reward after the first valid model, by pattern."""
    style()
    paths = json.loads((DATA / "search_paths.json").read_text())
    by = {r["run"]: r for r in rows}
    stop = first_stops(rows); gap = gap_to_best(rows)
    col = {k: c for k, _, c in OUTCOME}
    H, TOP, BOT = 1.82, 0.74, 0.27                   # inches; the axes tops and bottoms as figure fractions
    fig = plt.figure(figsize=(PAPER_W, H))
    fig.subplots_adjust(left=0.075, right=0.985, top=TOP, bottom=BOT)
    gs = fig.add_gridspec(1, 5, width_ratios=[0.6, 1, 1, 1, 1], wspace=0.3)
    a = fig.add_subplot(gs[0, 0])
    bs = [fig.add_subplot(gs[0, j]) for j in range(1, 5)]
    up = lambda inch: 1 + inch / ((TOP - BOT) * H)  # axes fraction, `inch` above the axes top
    P = [(by[k], [x for x in v if x[1] > 0.5]) for k, v in paths.items() if k in by and by[k]["group"] == "selected" and v]
    P = [(r, p) for r, p in P if p]

    # (a) package size at the first and the last valid check
    rng = np.random.default_rng(1)
    LO = min([94.0] + [100 * p[i][1] for _, p in P for i in (0, -1)])
    for i, idx in enumerate((0, -1)):
        for r, p in P:
            a.scatter(i + rng.uniform(-0.18, 0.18), 100 * p[idx][1], s=14, color=col[outcome_of(r, gap)], edgecolor=C["ink"],
                      linewidth=0.4, zorder=3, clip_on=False)
        if P:
            a.hlines(100 * np.median([p[idx][1] for r, p in P]), i - 0.3, i + 0.3, color=C["ink"], lw=1.4, zorder=5)
    a.set_xticks([0, 1]); a.set_xticklabels(["first\ncheck", "last\ncheck"]); a.set_xlim(-0.5, 1.5); a.set_ylim(LO, 100.5)
    a.set_ylabel("package size\n(% of cap)", labelpad=1)

    # (b) the public reward the agent saw after its first valid model, one panel per pattern; colour = the run's outcome
    STRONG, MOVE = 9.0, 0.25   # first valid reward that counts as a strong start; change that counts as movement
    groups = {"hold": [], "refine": [], "climb": [], "stall": []}
    for r, p in P:
        ok = [x for x in p if x[2] and x[2] > 0.5]
        if not ok:
            continue
        first, change = ok[0][2], ok[-1][2] - ok[0][2]
        key = ("refine" if change >= MOVE else "hold") if first >= STRONG else ("climb" if change >= MOVE else "stall")
        groups[key].append((r, ok))
    top = max([10.0] + [x[2] for g in groups.values() for _, ok in g for x in ok])
    spec = [("hold", "Strong start, hold", (STRONG - 0.4, top)), ("refine", "Strong start, refine", (STRONG - 0.4, top)),
            ("climb", "Weak start, climb", (0.0, top)), ("stall", "Weak start, stall", (0.0, top))]
    for k, (key, title, (y0, y1)) in enumerate(spec):
        b = bs[k]
        G = sorted(groups[key], key=lambda g: ["best", "close", "behind", "zero"].index(outcome_of(g[0], gap)))
        band = 0.16 * (y1 - y0)                      # room under the data for the "first declared done" dots
        for r, ok in G:
            o = outcome_of(r, gap)
            t = [100 * x[0] for x in ok]; y = [max(x[2], y0) for x in ok]   # a reward below the panel's floor is drawn on it
            b.step(t + [100], y + [y[-1]], where="post", color=col[o], lw=1.1, alpha=0.9, zorder=3,
                   ls=(0, (3, 1.6)) if o == "zero" else "-", clip_on=False)
            b.scatter(t[0], y[0], s=16, facecolor="white", edgecolor=col[o], linewidth=0.9, zorder=4, clip_on=False)
            if r["run"] in stop:
                b.scatter(100 * stop[r["run"]], y0 - band * 0.55 + band * 0.25 * rng.uniform(-1, 1), s=9, color=col[o],
                          linewidth=0, zorder=4, clip_on=False)
        b.axhline(y0, color=C["light"], lw=0.6, zorder=1)
        b.set_xlim(0, 100); b.set_ylim(y0 - band, y1); b.set_xticks([0, 50, 100])
        b.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        # header: the pattern, and under it the outcome mix of its runs as a stacked bar
        counts = {o: sum(outcome_of(r, gap) == o for r, _ in G) for o, _, _ in OUTCOME}
        n = len(G)
        b.text(0, up(0.175), title, transform=b.transAxes, fontsize=7.5, fontweight="bold", va="bottom")
        left, y_bar, h_bar = 0.0, up(0.035), up(0.15) - up(0.035)
        for o, _, c in OUTCOME:
            if counts[o]:
                w = counts[o] / n
                b.add_patch(plt.Rectangle((left, y_bar), w, h_bar, transform=b.transAxes, color=c, clip_on=False, lw=0))
                b.text(left + w / 2, y_bar + h_bar / 2, str(counts[o]), transform=b.transAxes, fontsize=7, color="white",
                       ha="center", va="center_baseline", fontweight="bold")
                left += w
        if k == 0:
            b.set_ylabel("public reward", labelpad=1)
    x0, x1 = bs[0].get_position().x0, bs[-1].get_position().x1
    fig.text((x0 + x1) / 2, BOT - 0.215 / H, "budget used (%)", ha="center", va="top", fontsize=8)
    head = TOP + 0.335 / H
    fig.text(x0, head, "(b) Public reward after the first valid model, by trajectory pattern", fontsize=8.5,
             fontweight="bold", ha="left", va="bottom")
    fig.canvas.draw()
    fig.text(fig.transFigure.inverted().transform((a.yaxis.get_tightbbox(fig.canvas.get_renderer()).x0, 0))[0], head,
             "(a) Package size", fontsize=8.5, fontweight="bold", ha="left", va="bottom")
    lab = {"best": "best on the task", "close": "PPL within 5% of best", "behind": "further behind", "zero": "not passed"}
    o = [Patch(color=c, label=lab[k]) for k, _, c in OUTCOME]
    m = [Line2D([], [], color=C["gray"], lw=1.1, ls=(0, (3, 1.6)), label="run not passed"),
         Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=C["gray"], markersize=4.5, label="first valid model"),
         Line2D([], [], marker="o", ls="", color=C["gray"], markersize=3.5, label="agent first declared done")]
    fig.legend(handles=o + m, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=7, fontsize=7, columnspacing=0.55,
               handletextpad=0.25, handlelength=0.9, borderaxespad=0, borderpad=0)
    save(fig, "commitment")


# --------------------------------------------------------------------------------------------------------------------
NON_EXPERT = ("shared", "attention", "embed", "lm_head")


def fig_allocation(rows) -> None:
    """Where the bytes go: share of resident bytes, expert vs non-expert bits, bits by component of a dense model."""
    style()
    A = json.loads((DATA / "byte_allocation.json").read_text())
    H = 2.55
    fig = plt.figure(figsize=(PAPER_W, H))
    box = lambda x, y, w, h: fig.add_axes((x / PAPER_W, y / H, w / PAPER_W, h / H))   # inches from the bottom left
    # (a) on the left at full height; (b) and (c) side by side on the right, the legend under them
    a = box(0.56, 0.33, 2.42, H - 0.33 - 0.2)
    b = box(3.86, 0.86, 1.12, H - 0.86 - 0.4)
    c = box(5.58, 0.86, 1.15, H - 0.86 - 0.4)
    comp = [("routed", "routed experts", C["indigo"]), ("dense_mlp", "dense MLP", C["blue2"]), ("shared", "shared experts", C["violet"]),
            ("attention", "attention / mixer", C["teal"]), ("embed", "embeddings", C["lgreen"]), ("lm_head", "output head", C["light"])]
    order = [r for r in sweep_rows(rows) if r["tight"] and r["run"] in A]
    order.sort(key=lambda r: (ORDER.index(r["model"]), -(r["ppl_fidelity"] or 0)))
    ypos, ylab, yy, last = [], [], 0.0, None
    for r in order:
        if last is not None and r["model"] != last:
            yy += 0.25
        if r["model"] != last:                       # the group's name on its own row, above its bars
            a.text(0, yy + 0.3, f"{MODEL[r['model']]}, {r['cap_gib']:g} GiB", fontsize=7, fontweight="bold", va="baseline")
            yy += 1.0
        last = r["model"]; left = 0.0
        for k, _, col in comp:
            w = A[r["run"]]["share"][k] * 100
            a.barh(yy, w, left=left, color=col, edgecolor=C["ink"], linewidth=0.5, height=0.8); left += w
        ypos.append(yy); ylab.append(AGENT[r["agent"]].split(" ")[0])
        if r["ppl_fidelity"]:
            a.text(102, yy, f"+{ppl_up(r['ppl_fidelity']):.0f}%", va="center", fontsize=7, color=C["red"] if r["zero"] else C["ink"],
                   fontweight="bold" if r["zero"] else "normal")
        yy += 1
    if ypos:
        a.set_yticks(ypos); a.set_yticklabels(ylab, fontsize=7); a.set_ylim(max(ypos) + 0.55, -0.55)
    a.set_xlim(0, 100)
    a.tick_params(axis="y", length=0, pad=2)
    a.text(102, 0.3, "PPL", fontsize=7, color=C["ink"], va="baseline"); a.grid(False)
    a.set_xlabel("share of resident bytes (%)", labelpad=2); a.set_title("Resident bytes by component, tighter limits"); panel(a, "a")
    # (b) MoE models: bits per non-expert weight against bits per routed-expert weight, with each limit's budget line
    budget, xs_all = {}, []
    for r in sweep_rows(rows):
        al = A.get(r["run"])
        if not al or not al["params"].get("routed"):
            continue
        ne_b = sum(al["gib"][k] for k in NON_EXPERT); ne_p = sum(al["params"][k] for k in NON_EXPERT)
        x = ne_b * 2 ** 30 * 8 / ne_p
        xs_all.append(x)
        fam = family(r)[0]
        b.scatter(x, al["bits_per_weight"]["routed"], s=36 if fam == "vq" else 18, marker=FAMILY_MARK[fam],
                  facecolor=C[r["agent"]], edgecolor=C["red"] if r["zero"] else C["ink"], linewidth=0.5, zorder=3)
        share = sum(al["params_share"][k] for k in NON_EXPERT)
        budget.setdefault((r["model"], r["cap_gib"]), (r["cap_gib"] * 2 ** 30 * 8 / sum(al["params"].values()), share))
    if xs_all:
        xs = np.linspace(min(xs_all) * 0.9, max(xs_all) * 1.05, 40)
        for (m, cap), (cap_bits, share) in sorted(budget.items()):
            # every byte under the limit: share * x + (1 - share) * y = the limit in bits per kept weight
            ys = (cap_bits - share * xs) / (1 - share)
            b.plot(xs, ys, color=C["gray"], lw=1, ls="--", zorder=1)
            b.text(xs[-1], ys[-1], f" {task_tag(m, cap)}", color=C["ink"], fontsize=6, ha="left", va="center")
    b.set_xlabel("bits on non-expert weights", labelpad=2); b.set_ylabel("bits on routed\nexperts", labelpad=2)
    b.set_title("Expert vs non-expert\nbits per weight"); panel(b, "b")
    # (c) a dense model (no routed experts) at its tighter limit: bits per original weight by component
    dense = [r for r in sweep_rows(rows) if r["tight"] and r["run"] in A and not A[r["run"]]["params"].get("routed")]
    dense_model = dense[0]["model"] if dense else None
    dense = sorted((r for r in dense if r["model"] == dense_model), key=lambda r: -(r["ppl_fidelity"] or 0))
    names = [("dense_mlp", "MLP"), ("attention", "attn"), ("embed", "embed"), ("lm_head", "head")]
    w = 0.8 / max(len(dense), 1)
    for j, r in enumerate(dense):
        vals = [A[r["run"]]["bits_per_weight"][k] or 0 for k, _ in names]
        c.bar(np.arange(4) + (j - (len(dense) - 1) / 2) * w, vals, width=w, color=C[r["agent"]], edgecolor=C["red"] if r["zero"] else C["ink"], linewidth=0.5)
    c.set_xticks(range(4)); c.set_xticklabels([n for _, n in names]); c.set_ylabel("bits per\noriginal weight", labelpad=2)
    c.tick_params(axis="x", length=0, labelsize=7); c.grid(axis="x", visible=False); c.set_xlim(-0.5, 3.5)
    where = f"{SHORT_TASK[dense_model]} {dense[0]['cap_gib']:g} GiB" if dense else "Dense model"
    c.set_title(f"{where}:\nbits by component"); panel(c, "c")
    flush_titles(fig, a, b, c)
    # legend under (b) and (c): the components of (a), then the quantizer families of (b) and the red edge of (b) and (c)
    x0 = 3.42 / PAPER_W
    fig.legend(handles=[Patch(color=col, label=lab) for _, lab, col in comp], loc="upper left", bbox_to_anchor=(x0, 0.49 / H),
               ncol=3, fontsize=7, columnspacing=0.9, handletextpad=0.3, handlelength=1.0, labelspacing=0.25, borderaxespad=0)
    fig.legend(handles=family_handles() + [red_edge_handle()],
               loc="upper left", bbox_to_anchor=(x0, 0.225 / H), ncol=2, fontsize=7, columnspacing=0.9, handletextpad=0.3,
               handlelength=1.0, labelspacing=0.25, borderaxespad=0)
    save(fig, "allocation")


# --------------------------------------------------------------------------------------------------------------------
def fig_representation(rows) -> None:
    """Hidden PPL above BF16 of every run under each size limit, marked by its quantizer family."""
    style()
    sweep = [r for r in sweep_rows(rows) if r["ppl_fidelity"]]
    dh = lambda r: ppl_up(r["ppl_fidelity"])
    refs = released_refs()
    H = 2.6
    fig = plt.figure(figsize=(max(COL_W, 0.55 * len(TASKS) + 0.8), H))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.35, 1.2], hspace=0.08)
    top, a = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    lab = [f"{SHORT_TASK[m].split('-')[0]}\n{c:g} GiB" for m, c in TASKS]
    YMAX = 50   # above it a run is drawn in the band over the break, at its value
    TOPMAX = max([YMAX * 2] + [dh(r) * 1.1 for r in sweep])
    n = len(ACTIVE)
    for i, t in enumerate(TASKS):
        for r in (r for r in sweep if (r["model"], r["cap_gib"]) == t):
            fam = family(r)[0]
            x = i + (ACTIVE.index(r["agent"]) - (n - 1) / 2) * min(0.12, 0.7 / n)
            y = dh(r)
            ax = top if y > YMAX else a
            ax.scatter(x, y, s=34 if fam == "vq" else 15, marker=FAMILY_MARK[fam], facecolor=C[r["agent"]],
                       edgecolor=C["red"] if r["zero"] else C["ink"], linewidth=0.5, zorder=3)
            if y > YMAX:
                ax.annotate(f"+{y:.0f}%", (x, y), xytext=(4, 0), textcoords="offset points", fontsize=5.8, va="center")
        if t in refs:
            a.scatter(i + 0.43, refs[t]["ppl_up"], **dict(REF, s=22))
    ntight = sum(1 for m, c in TASKS if any(r["tight"] and (r["model"], r["cap_gib"]) == (m, c) for r in sweep_rows(rows)))
    for ax in (top, a):
        if ntight:
            ax.axvspan(-0.5, ntight - 0.5, color=C["panel"], zorder=0)    # the tighter limits
        ax.set_xlim(-0.5, len(TASKS) - 0.5)
    top.set_ylim(YMAX, TOPMAX)
    top.tick_params(axis="x", labelbottom=False, length=0); top.spines["bottom"].set_visible(False)
    for ax, y in ((top, 0), (a, 1)):    # the break: slashes on the left spine where the scale changes
        ax.plot([-0.025, 0.025], [y - 0.03, y + 0.03], transform=ax.transAxes, color=C["ink"], lw=0.8, clip_on=False)
    a.set_ylim(0, YMAX)
    a.set_xticks(range(len(TASKS))); a.set_xticklabels(lab, fontsize=6.5)
    a.tick_params(axis="x", length=0, pad=2)
    a.set_ylabel("PPL above BF16 (%)", labelpad=2, fontsize=7)
    a.yaxis.set_label_coords(-0.1, 0.62)
    top.set_title("PPL under the same size limit"); panel(top, "a", size=7.5)
    h = [Line2D([], [], marker="o", ls="", color=C[k], markersize=4, label=AGENT[k]) for k in ACTIVE] + [red_edge_handle()]
    if refs:
        h.append(ref_handle("released quantization"))
    h += family_handles(4)
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=3, fontsize=6, columnspacing=1.0,
               handletextpad=0.2, handlelength=1.0, labelspacing=0.25)
    fig.subplots_adjust(left=0.16, right=0.99, top=0.9, bottom=(0.45 + 0.13 * -(-len(h) // 3)) / H)
    flush_titles(fig, top)
    save(fig, "representation")


# --------------------------------------------------------------------------------------------------------------------
def fig_hidden_criteria(rows) -> None:
    """Correctness: multi-benchmark accuracy against the PPL above BF16 in final verification, and every run's accuracy on
    each benchmark. A log PPL axis shows every run at its value; plus signs: the released quantizations."""
    style()
    bf16, gate = gate_accuracy()
    G = [(r, gate[r["run"]]) for r in sweep_rows(rows) if r["run"] in gate and gate[r["run"]]["final"] and r["ppl_fidelity"]]
    X = np.array([ppl_up(r["ppl_fidelity"]) for r, _ in G])
    refs = released_refs()
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL_W, 2.1), sharey=True, gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.42})
    XMIN, XMAX = log_limits(list(X) + [r["ppl_up"] for r in refs.values()])
    models = [m for m in ORDER if m in bf16]
    for m in models:   # each model's BF16 accuracy and its gate, 0.8 x BF16
        a.axhline(bf16[m]["accuracy_pct"], color=MODEL_C[m], lw=0.7, zorder=1)
        a.axhline(bf16[m]["gate_pct"], color=MODEL_C[m], lw=0.7, ls=(0, (3, 1.6)), zorder=1)
    for x, (r, g) in zip(X, G):
        a.scatter(x, g["final"]["accuracy_pct"], s=11, marker=MODEL_MARK[r["model"]], color=MODEL_C[r["model"]],
                  edgecolor=C["red"] if not g["passed"] else C["ink"], linewidth=0.8 if not g["passed"] else 0.3, zorder=3)
    for ref in refs.values():
        a.scatter(ref["ppl_up"], ref["accuracy_pct"], **dict(REF, s=18, linewidth=0.6))
    a.set_ylim(0, 100)
    a.set_ylabel("accuracy (%)", fontsize=6.5, labelpad=1)
    a.set_title("Multi-benchmark accuracy"); panel(a, "a", size=7)
    # (b) every run's accuracy on each benchmark, with its median over log-spaced PPL bins (each with at least 2 runs)
    EDGES = list(np.geomspace(XMIN, XMAX, 8))
    for k, _, col in BENCH:
        pts = sorted((x, g["final"]["by_benchmark_pct"][k], r["model"]) for x, (r, g) in zip(X, G))
        for x, y, m in pts:
            b.scatter(x, y, s=9, marker=MODEL_MARK[m], color=col, edgecolor=C["ink"], linewidth=0.3, alpha=0.9, zorder=3)
        line = []
        for lo, hi in zip(EDGES, EDGES[1:]):
            inside = [(x, y) for x, y, _ in pts if lo <= x < hi]
            if len(inside) >= 2:
                line.append((math.exp(np.mean([math.log(x) for x, _ in inside])), float(np.median([y for _, y in inside]))))
        if line:
            b.plot(*zip(*line), color=col, lw=1.4, zorder=4, solid_capstyle="round")
    b.tick_params(axis="y", labelleft=True)
    b.set_ylabel("accuracy (%)", fontsize=6.5, labelpad=1)
    b.set_title("Each benchmark"); panel(b, "b", size=7)
    for ax in (a, b):
        ax.set_xscale("log"); ax.set_xlim(XMIN, XMAX); plain_log_ticks(ax)
        ax.set_xlabel("PPL above BF16 (%)", fontsize=6.5, labelpad=1)
        ax.tick_params(labelsize=6, pad=1.5)
    mh = [Line2D([], [], marker=MODEL_MARK[m], ls="-", lw=0.7, color=MODEL_C[m], markersize=3.5, markeredgecolor=C["ink"],
                 markeredgewidth=0.3, label=f"{MODEL_SHORT[m]} (BF16 {bf16[m]['accuracy_pct']:.2f}%)") for m in models]
    lines = [Line2D([], [], color=C["gray"], lw=0.7, label="BF16 (a)"),
             Line2D([], [], color=C["gray"], lw=0.7, ls=(0, (3, 1.6)), label="gate: 0.8 × BF16"),
             Line2D([], [], marker="o", ls="", color="white", markeredgecolor=C["red"], markeredgewidth=0.8, markersize=3.5,
                    label="not passed")]
    if refs:
        lines.append(ref_handle("released quantization (a)"))
    bench = [Line2D([], [], color=col, lw=1.3, marker="o", markersize=3, markeredgecolor=C["ink"], markeredgewidth=0.3,
                    label=f"{name} (b)") for _, name, col in BENCH]
    h = mh + lines + bench
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=5.5, columnspacing=0.8,
               handletextpad=0.3, handlelength=1.4, handleheight=0.6, labelspacing=0.3)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.91, bottom=0.18 + 0.065 * -(-len(h) // 3))
    save(fig, "correctness")


# --------------------------------------------------------------------------------------------------------------------
PRUNE = re.compile(r"prun|merg|expert dropping|dropping|puzzle|slim|minitron|expert reduction|sparse|reap|structured|width|depth|"
                   r"layer removal|decompos|low-rank|factoriz", re.I)
QUANT = re.compile(r"quant|bit|codebook|gptq|awq|fp4|fp8|vq\b|xfp|lattice|rotation|outlier|precision", re.I)


def paper_kind(title: str) -> str:
    q, p = bool(QUANT.search(title)), bool(PRUNE.search(title))
    return "quantization" if q and not p else ("pruning / merging" if p and not q else ("both" if p and q else "other"))


def fig_prior_knowledge(rows) -> None:
    """How agents used the paper collection: papers opened, when, and of what kind."""
    style()
    cat = {p["id"]: p for p in json.load(open(CATALOG))["papers"]}
    U = json.loads((DATA / "paper_usage.json").read_text())
    by = {r["run"]: r for r in rows}
    sweep = [(by[k], u) for k, u in U.items() if k in by and by[k]["group"] == "selected"]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.2, 2.25), gridspec_kw={"width_ratios": [1.1, 0.9, 1.3], "wspace": 0.42})
    order = [ag for ag in ACTIVE if any(r["agent"] == ag for r, _ in sweep)]
    rng = np.random.default_rng(2)
    for i, ag in enumerate(order):
        for r, u in (t for t in sweep if t[0]["agent"] == ag):
            dot(a, i + rng.uniform(-0.17, 0.17), u["distinct_read"], r, s=14, clip_on=False)
    a.set_ylim(0, max([10] + [u["distinct_read"] for _, u in sweep]) * 1.1)
    a.set_xticks(range(len(order))); a.set_xticklabels([SHORT_AGENT[k] for k in order], fontsize=6.8)
    a.set_xlim(-0.6, len(order) - 0.4)
    a.set_ylabel("distinct papers opened"); a.set_title("Papers opened per run"); panel(a, "a")
    # (b) when the reads happen: cumulative share of each agent's reads over the position in the run
    last = 0.0
    for ag in order:
        t = np.sort([e["at"] for r, u in sweep if r["agent"] == ag for e in u["events"] if e["kind"] == "read"])
        if len(t):
            b.step(np.concatenate([[0], t]), np.concatenate([[0], np.arange(1, len(t) + 1) / len(t)]), where="post", color=C[ag], lw=1.6)
            last = max(last, float(np.quantile(t, 0.98)))
    b.set_xlim(0, min(1.0, max(last * 1.1, 0.02)))
    b.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    b.set_xlabel("position in the run"); b.set_ylabel("cumulative share of reads"); b.set_title("When papers are read"); panel(b, "b")
    # (c) what kind of paper each agent opened, over all its runs
    cols = {"quantization": C["indigo"], "pruning / merging": C["tan"], "both": C["violet"], "other": C["light"]}
    for i, ag in enumerate(order):
        k = collections.Counter(paper_kind(cat.get(p, {}).get("title", "")) for r, u in sweep if r["agent"] == ag
                                for p in u["papers_read"])
        n, bot = sum(k.values()), 0
        for key in cols:
            if n:
                c.bar(i, 100 * k[key] / n, bottom=bot, color=cols[key], edgecolor=C["ink"], linewidth=0.5, width=0.74)
                bot += 100 * k[key] / n
        c.text(i, 103, str(n), ha="center", fontsize=6.5, color=C["ink"])
    c.set_xticks(range(len(order))); c.set_xticklabels([SHORT_AGENT[k] for k in order], fontsize=6.3)
    c.set_ylim(0, 112); c.set_yticks([0, 50, 100]); c.set_ylabel("share of papers opened (%)"); c.grid(axis="x", visible=False)
    c.set_title("Kind of paper opened (n above)"); panel(c, "c")
    h = [Patch(color=v, label=k) for k, v in cols.items()]
    top_legend(fig, agent_handles() + h, ncol=6, fontsize=6.3, room=0.47)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9)
    save(fig, "prior_knowledge")


# --------------------------------------------------------------------------------------------------------------------
def process_records(rows):
    """Per selected run, from its trajectory: GPU use per minute, session stops, public checks, weight changes."""
    T = utc_seconds
    WEIGHT = re.compile(r"\.(safetensors|bin|pt|pth|npz|gguf)$")
    out = {}
    for r in sweep_rows(rows):
        found, root = trajectory_dir(r["run"]), run_dir(r["run"])
        if not found or not root or not (root / "METRICS.json").is_file():
            print(f"{r['run']}: no trajectory or METRICS.json; left out of the process figures", file=sys.stderr)
            continue
        traj, lineage, run = str(found), str(found.parents[1]), str(root)
        m = json.load(open(run + "/METRICS.json"))
        t0, t1 = T(m["timeline"]["agent_started_utc"]), T(m["timeline"]["agent_ended_utc"]); span = t1 - t0
        f = lambda t: (t - t0) / span
        launches = json.load(open(traj + "/TIMING.json"))["launches"]
        # a session the API's credit limit ended (the harness paused it and gave the time back) is no stop of the agent's
        held = {e.get("since_utc") for e in map(json.loads, open(run + "/journal.jsonl"))
                if e.get("event") == "agent_credit_hold_ended"}
        stops = [f(T(l["finished_utc"])) for l in launches if l["finished_utc"] not in held] or [1.0]
        res = [json.loads(line) for line in gzip.open(traj + "/resources.jsonl.gz", "rt")]
        gpu = [(f(T(u["utc"])), (u["gpu"].get("utilization_percent") or 0) > 20) for u in res if u.get("gpu") and t0 <= T(u["utc"]) <= t1]
        checks = [(f(T(c["utc"])), c.get("public_reward")) for c in m["public_checks"]["checks"]
                  if c.get("action") == "full-check" and c.get("valid") and c.get("public_reward")]
        prev, changes = None, []
        for line in open(lineage + "/versions.jsonl"):
            v = json.loads(line)
            if v.get("run_id") != r["run"] or v.get("source") != "agent":
                continue
            sig = tuple(sorted((e["path"], e.get("sha256")) for e in json.load(open(f"{lineage}/{v['name']}/MANIFEST.json"))
                               if WEIGHT.search(e["path"])))
            if sig != prev:
                changes.append(f(T(v["captured_utc"])))
            prev = sig
        busy = np.mean([b for _, b in gpu]) if gpu else 0.0
        out[r["run"]] = {"stop1": stops[0], "stops": stops, "resumes": len(launches) - 1, "gpu": gpu, "busy": busy,
                         "minute": 60 / span, "busy_after": np.mean([b for t, b in gpu if t > stops[0]] or [0.0]),
                         "checks": checks, "changes": changes, "changes_after": sum(c > stops[0] + 1e-3 for c in changes)}
    return out


_RECORDS: dict = {}


def records(rows):
    if not _RECORDS:
        _RECORDS.update(process_records(rows))
    return _RECORDS


def first_stops(rows):
    """Each selected run's first stop (share of the budget), from its trajectory."""
    return {k: v["stop1"] for k, v in records(rows).items()}


def gap_to_best(rows):
    """Each run's hidden-set perplexity above the best run's on its task, in % (0 for the best run)."""
    best = {}
    runs = [r for r in sweep_rows(rows) if r["ppl_fidelity"]]
    for r in runs:
        k = (r["model"], r["cap_gib"]); best[k] = max(best.get(k, 0), r["ppl_fidelity"])
    gap = {r["run"]: 100 * (best[(r["model"], r["cap_gib"])] / r["ppl_fidelity"] - 1) for r in runs}
    return collections.defaultdict(lambda: float("inf"), gap)


def timeline_order(R, rows):
    """The agents ordered by the mean share of their runs' time with the GPU busy, most first."""
    have = [ag for ag in ACTIVE if any(r["agent"] == ag and r["run"] in R for r in sweep_rows(rows))]
    busy = {ag: np.mean([R[r["run"]]["busy"] for r in sweep_rows(rows) if r["agent"] == ag and r["run"] in R]) for ag in have}
    return sorted(have, key=lambda ag: -busy[ag])


def fig_budget_timeline(rows) -> None:
    """Process timeline of every run: compute in use, the agent's first stop and its resumes, checks and weight changes."""
    style()
    R = records(rows)
    order = []
    for ag in timeline_order(R, rows):
        G = [r for r in sweep_rows(rows) if r["agent"] == ag and r["run"] in R]
        order += [None] + sorted(G, key=lambda r: R[r["run"]]["stop1"], reverse=True)
    order = order[1:]
    n = sum(r is not None for r in order)
    fig, a = plt.subplots(figsize=(7.2, 0.9 + 0.11 * n + 0.07 * order.count(None)))
    y = 0; yt, yl = [], []
    for r in order:
        if r is None:
            y += 0.6; continue
        rec = R[r["run"]]
        for t, b in rec["gpu"]:                                   # one cell per minute of the run
            a.add_patch(plt.Rectangle((t, y - 0.36), rec["minute"], 0.72, color=C[r["agent"]] if b else C["panel"], lw=0, zorder=1))
        a.add_patch(plt.Rectangle((rec["stop1"], y - 0.36), 1 - rec["stop1"], 0.72, fill=False, hatch="/////", edgecolor=C["gray"],
                                  lw=0, alpha=0.55, zorder=2))
        a.plot([rec["stop1"]] * 2, [y - 0.45, y + 0.45], color=C["ink"], lw=1.2, zorder=4)
        for t in rec["changes"]:
            a.scatter(t, y, marker="v", s=12, color="white", edgecolor=C["ink"], linewidth=0.5, zorder=5)
        yt.append(y); yl.append(task_tag(r["model"], r["cap_gib"]))
        if r["ppl_fidelity"]:
            a.text(1.012, y, f"+{ppl_up(r['ppl_fidelity']):.0f}%", va="center", fontsize=5.6, color=C["red"] if r["zero"] else C["ink"],
                   fontweight="bold" if r["zero"] else "normal", transform=a.get_yaxis_transform())
        y += 1
    ys = 0
    for i, r in enumerate(order):
        if r is None:
            ys += 0.6; continue
        if i == 0 or order[i - 1] is None:          # the first run of an agent's group: the agent's name beside the group
            k = sum(1 for q in order[i:] if q is not None and q["agent"] == r["agent"])
            a.text(-0.135, ys + (k - 1) / 2, AGENT[r["agent"]].replace(" ", "\n"), transform=a.get_yaxis_transform(), ha="center",
                   va="center", fontsize=7, fontweight="bold", color=C[r["agent"]])
        ys += 1
    a.set_yticks(yt); a.set_yticklabels(yl, fontsize=5.6); a.set_ylim(y - 0.4, -1.0); a.set_xlim(0, 1)
    a.spines["left"].set_visible(False); a.tick_params(axis="y", length=0); a.grid(False)
    a.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); a.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    a.set_xlabel("development budget used")
    a.text(1.012, -0.75, "PPL", fontsize=5.8, color=C["ink"], transform=a.get_yaxis_transform())
    a.set_title("GPU use, first stop and weight changes over the budget", loc="left", fontsize=8.5, fontweight="bold", pad=5)
    h = [Patch(color=C["gray"], label="GPU computing (in the agent's colour)"), Patch(color=C["panel"], label="GPU idle"),
         Patch(facecolor="white", hatch="/////", edgecolor=C["gray"], lw=0, label="after the agent first declared it was done"),
         Line2D([], [], color=C["ink"], lw=1.2, label="first stop"),
         Line2D([], [], marker="v", ls="", mfc="white", mec=C["ink"], markersize=4.5, label="weights changed")]
    top_legend(fig, h, ncol=5, fontsize=6.3, room=0.42)
    fig.subplots_adjust(left=0.13, right=0.94, top=1 - 0.3 / fig.get_size_inches()[1])
    save(fig, "budget_timeline")


def fig_budget_outcome(rows) -> None:
    """Effort against outcome, within each task: compute used, when the agent stopped, what the resumes bought."""
    style()
    R = records(rows); gap = gap_to_best(rows)
    runs = [r for r in sweep_rows(rows) if r["run"] in R]
    fig, (a, b, d, c) = plt.subplots(1, 4, figsize=(7.2, 2.1), gridspec_kw={"wspace": 0.42, "width_ratios": [1, 1, 1, 1.25]})
    med_calls = {t: np.median([r["tool_calls"] or 0 for r in runs if (r["model"], r["cap_gib"]) == t] or [0]) for t in TASKS}
    for r in runs:
        R[r["run"]]["calls"] = (r["tool_calls"] or 0) / (med_calls[(r["model"], r["cap_gib"])] or 1)
    HI = 30   # runs above it sit on the top edge
    for ax, key, xlabel, title in ((a, "busy", "share of run with GPU busy", "GPU use vs PPL gap"),
                                   (b, "stop1", "first stop (share of budget)", "First stop vs PPL gap"),
                                   (d, "calls", "tool calls / task median", "Tool calls vs PPL gap")):
        for r in runs:
            dot(ax, R[r["run"]][key], min(gap[r["run"]], HI), r, clip_on=False)
        ax.set_ylim(-1, HI); ax.axhline(0, color=C["light"], lw=0.8)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
    a.set_ylabel("PPL above task's best (%)", labelpad=1)
    panel(a, "a", 7.8); panel(b, "b", 7.8); panel(d, "c", 7.8)
    # (d) every run's budget left at its first stop, by agent: filled if the resumed time changed its weights
    order = timeline_order(R, rows)
    for i, ag in enumerate(order):
        G = [r for r in runs if r["agent"] == ag]
        ys = [100 * (1 - R[r["run"]]["stop1"]) for r in G]
        dx = swarm(ys, math.sqrt(16) / 72 * 1.1, 1.62 / max(len(order), 1), 1.45 / 100)
        for r, y, o in zip(G, ys, dx):
            changed = R[r["run"]]["changes_after"] > 0
            c.scatter(i + o, y, s=16, color=C[ag] if changed else "white", edgecolor=C[ag] if not changed else C["ink"],
                      linewidth=0.9 if not changed else 0.5, zorder=3)
        c.plot([i - 0.3, i + 0.3], [np.median(ys)] * 2, color=C["ink"], lw=1.0, zorder=2)
        c.text(i, 104, f"{sum(R[r['run']]['changes_after'] > 0 for r in G)}/{len(G)}", ha="center", va="center", fontsize=6.5)
    c.set_xticks(range(len(order))); c.set_xticklabels([SHORT_AGENT[k] for k in order], fontsize=6.0)
    c.tick_params(axis="x", length=0)
    c.grid(axis="x", visible=False); c.set_ylim(0, 110); c.set_yticks([0, 25, 50, 75, 100]); c.set_xlim(-0.6, len(order) - 0.4)
    c.set_ylabel("budget left at first stop (%)", labelpad=1); c.set_title("Budget left at first stop"); panel(c, "d", 7.8)
    top_legend(fig, agent_handles() + [
        Line2D([], [], marker="o", ls="", color=C["gray"], markeredgecolor=C["ink"], markeredgewidth=0.5, markersize=4.5,
               label="weights changed after it (d)"),
        Line2D([], [], marker="o", ls="", color="white", markeredgecolor=C["gray"], markeredgewidth=0.9, markersize=4.5,
               label="unchanged (d)"),
        Line2D([], [], color=C["ink"], lw=1.0, label="median (d)")], ncol=5, fontsize=6.3, room=0.5)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88)
    save(fig, "budget_outcome")


# --------------------------------------------------------------------------------------------------------------------
# Hidden-set measurements per run (insights_hidden.json, insight_data.py) and the gate accuracies (gate_accuracy.py).
def gate_accuracy():
    """Hidden gate accuracy recomputed from correct counts (compression_bench/scripts/gate_accuracy.py): BF16 references per
    model, and per run (by run id) its final and checkpoint verdicts."""
    d = json.loads((DATA / "gate_accuracy.json").read_text())
    return d["bf16"], {r["run"]: r for r in d["runs"]}


def released_refs():
    """The released quantization of each task measured through the final verification (released_baseline.py
    RESULTS.json), keyed (model, capacity GiB): reference points with their measured PPL,
    accuracy and decode latency. Their protocol outcome (0: a released quantization is never the submission's own
    compression) stays in RESULTS.json."""
    if not RELEASED.is_file():
        return {}
    out = {}
    for r in json.loads(RELEASED.read_text())["rows"]:
        if r["kind"] != "released" or r.get("status") != "measured":
            continue
        slug, cap = r["task"].rsplit("_", 1)
        m = MODEL_KEY.get(slug, slug)
        by = r["accuracy_by_benchmark"]
        out[(m, float(cap.rstrip("g")))] = {
            "release": r["release"], "ppl": r["ppl"], "ppl_up": r["ppl_increase_percent"], "accuracy_pct": 100 * r["accuracy"],
            "by_benchmark_pct": {k: 100 * v["accuracy"] for k, v in by.items()}, "package_gib": r["package_gib"],
            "bpw": r["package_gib"] * 8 * 2 ** 30 / PARAMS[m] if m in PARAMS else None, "decode_s": r["decode_latency_seconds"],
            "protocol_reward": r["protocol_reward"]}
    return out


def ref_handle(label="released quantization (reference)"):
    return Line2D([], [], marker=REF["marker"], ls="", markerfacecolor="white", markeredgecolor=C["ink"],
                  markeredgewidth=0.8, markersize=5, label=label)


def insights():
    """insights_hidden.json's runs, by run id."""
    return {r["run"]: r for r in json.loads((DATA / "insights_hidden.json").read_text())["runs"]}


def fig_code_blocks(rows) -> None:
    """Code-block concentration of the hidden excess cross-entropy, and HumanEval accuracy against PPL at the tighter
    limits (squares: runs that removed capacity, from --families)."""
    style()
    H = insights()
    tight = {r["run"] for r in sweep_rows(rows) if r["tight"]}
    removed = {k for k in H if family({"run": k})[1]}
    fig, (b, c) = plt.subplots(1, 2, figsize=(5.0, 2.25), gridspec_kw={"width_ratios": [1.0, 1.1], "wspace": 0.42})
    # (a) per source model: the excess on the code-seeded blocks over the mean excess, every run
    models = [m for m in ORDER if any(r["model"] == m and "block" in r for r in H.values())]
    allys = []
    for i, m in enumerate(models):
        runs = sorted(((k, r) for k, r in H.items() if r["model"] == m and "block" in r), key=lambda kr: kr[1]["block"]["code_excess_ratio"])
        ys = [r["block"]["code_excess_ratio"] for _, r in runs]
        allys += ys
        dx = swarm(ys, math.sqrt(16) / 72 * 1.1, 1.7 / max(len(models), 1), 1.35 / 1.5)
        for (k, r), y, o in zip(runs, ys, dx):
            b.scatter(i + o, y, s=16, color=C.get(r["agent"], C["gray"]), marker="s" if k in removed else "o",
                      edgecolor=C["red"] if not r["passed"] else C["ink"], linewidth=0.7 if not r["passed"] else 0.4, zorder=3)
            if r["block"]["code_ratio_p"] < 0.05:
                b.text(i + o, y + 0.035, "*", ha="center", va="bottom", fontsize=8)
    b.axhline(1, color=C["ink"], lw=0.8)
    b.set_xticks(range(len(models))); b.set_xticklabels([SHORT[m] for m in models], fontsize=6.4)
    b.set_xlim(-0.6, len(models) - 0.4)
    if allys:
        b.set_ylim(min(allys + [1]) - 0.1, max(allys + [1]) + 0.15)
    b.set_ylabel("code / mean excess CE", labelpad=1)
    b.tick_params(axis="x", length=0); b.grid(axis="x", visible=False)
    b.set_title("Code-block excess CE"); panel(b, "a", 7.8)
    # (b) HumanEval accuracy against PPL in final verification at the tighter limits, with BF16's per model
    bf16, gate = gate_accuracy()
    refs = released_refs()
    pts = [(k, r) for k, r in H.items() if k in tight and k in gate and gate[k]["final"] and r.get("ppl_above_bf16_pct")]
    tight_caps = {(r["model"], r["cap_gib"]) for r in sweep_rows(rows) if r["tight"]}
    XMIN, XMAX = log_limits([r["ppl_above_bf16_pct"] for _, r in pts] + [v["ppl_up"] for t, v in refs.items() if t in tight_caps])
    lines = {}
    for m in [m for m in ORDER if m in bf16]:
        lines.setdefault(round(bf16[m]["by_benchmark_pct"]["code"], 2), []).append(SHORT[m])
    for y, names in lines.items():
        c.axhline(y, color=C["gray"], lw=0.7, zorder=1)
        c.text(XMAX * 0.92, y + 1.5, f"BF16 {y:.0f}%\n{', '.join(names)}", ha="right", va="bottom", fontsize=5.6,
               color=C["gray"], linespacing=1.0)
    for k, r in pts:
        g = gate[k]
        c.scatter(r["ppl_above_bf16_pct"], g["final"]["by_benchmark_pct"]["code"], s=24, color=C.get(r["agent"], C["gray"]),
                  marker="s" if k in removed else "o", edgecolor=C["red"] if not g["passed"] else C["ink"], linewidth=0.6,
                  zorder=3, clip_on=False)
    for t, ref in refs.items():
        if t in tight_caps:
            c.scatter(ref["ppl_up"], ref["by_benchmark_pct"]["code"], **dict(REF, s=24))
    c.set_xscale("log"); c.set_xlim(XMIN, XMAX); c.set_ylim(0, 100); plain_log_ticks(c)
    c.set_xlabel("PPL above BF16 (%)"); c.set_ylabel("HumanEval accuracy (%)", labelpad=1)
    c.set_title("HumanEval accuracy vs PPL, tighter limits"); panel(c, "b", 7.8)
    h = agent_handles(zero=False) + [Line2D([], [], marker="s", ls="", color=C["gray"], markersize=4.5, label="capacity removed"),
                          Line2D([], [], marker="$*$", ls="", color=C["ink"], markersize=5, label="p < 0.05 (a)"),
                          red_edge_handle()]
    if refs:
        h.append(ref_handle("released quantization (b)"))
    top_legend(fig, h, ncol=5, fontsize=6.2, room=0.5)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.88)
    save(fig, "code_blocks")


def fig_bits_vs_ppl(rows) -> None:
    """The byte split at the looser limits, and every run's bits per bulk weight (routed experts; the MLP of a dense
    model) against its hidden PPL (markers: the bulk quantizer's family, from --families)."""
    style()
    H = insights()
    loose = {r["run"] for r in sweep_rows(rows) if not r["tight"]}
    fig, (b, c) = plt.subplots(1, 2, figsize=(5.0, 2.2), gridspec_kw={"width_ratios": [1.0, 1.15], "wspace": 0.42})

    def mark(ax, x, y, k, r, s=26, **kw):
        fam = family({"run": k})[0]
        ax.scatter(x, y, s=s * (1.9 if fam == "vq" else 1), marker=FAMILY_MARK[fam], color=C.get(r["agent"], C["gray"]),
                   edgecolor=C["ink"] if r["passed"] else C["red"], linewidth=0.5 if r["passed"] else 0.8, zorder=3, **kw)
    for k, r in H.items():
        if k not in loose or "allocation" not in r or r.get("ppl_above_bf16_pct") is None:
            continue
        share = r["allocation"]["share"]
        mark(b, 100 * (1 - share["routed"] - share.get("dense_mlp", 0)), r["ppl_above_bf16_pct"], k, r)
    b.set_xlabel("non-bulk bytes (%)")
    b.set_ylabel("PPL above BF16 (%)", labelpad=1)
    b.set_ylim(bottom=0); b.set_title("Looser limits: byte split vs PPL"); panel(b, "a", 7.8)
    ys = []
    for k, r in H.items():
        if "allocation" not in r or r.get("ppl_above_bf16_pct") is None:
            continue
        bpw = r["allocation"]["bits_per_weight"]
        bits = bpw.get("routed") or bpw.get("dense_mlp")
        if bits:
            mark(c, bits, r["ppl_above_bf16_pct"], k, r, s=20, clip_on=False)
            ys.append(r["ppl_above_bf16_pct"])
    c.set_yscale("log"); c.set_ylim(*log_limits(ys)); plain_log_ticks(c, "y")
    c.set_xlabel("bits per expert / MLP weight")
    c.set_ylabel("PPL above BF16 (%)", labelpad=1)
    c.set_title("Every run: bits per weight vs PPL"); panel(c, "b", 7.8)
    h = agent_handles(zero=False) + [red_edge_handle()] + family_handles(4.2)
    top_legend(fig, h, ncol=4, fontsize=6.2, room=0.5)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.88)
    save(fig, "bits_vs_ppl")


FIGS = {"commitment": fig_commitment, "representation": fig_representation, "prior_knowledge": fig_prior_knowledge,
        "allocation": fig_allocation, "correctness": fig_hidden_criteria, "budget_timeline": fig_budget_timeline,
        "budget_outcome": fig_budget_outcome, "code_blocks": fig_code_blocks, "bits_vs_ppl": fig_bits_vs_ppl}


def configure(data: Path | None = None, out: Path | None = None, families: Path | None = None,
              released: Path | None = None) -> None:
    """Point the module at an analysis directory (findings_figures.py shares this configuration)."""
    global DATA, OUT, RELEASED
    if data:
        DATA = Path(data)
        OUT = DATA / "figures"
    if out:
        OUT = Path(out)
    if released:
        RELEASED = Path(released)
    if families:
        FAMILIES.clear()
        FAMILIES.update(json.loads(Path(families).read_text()))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("figures", nargs="*", metavar="FIGURE", help="default: every figure")
    parser.add_argument("--data", type=Path, default=DATA, help="the analysis directory")
    parser.add_argument("--out", type=Path, help="where the figures go (default DATA/figures)")
    parser.add_argument("--families", type=Path, help="quantizer family per run (format above)")
    parser.add_argument("--released", type=Path, help="released_baseline.py RESULTS.json")
    args = parser.parse_args(argv)
    unknown = [f for f in args.figures if f not in FIGS]
    if unknown:
        parser.error(f"unknown figure(s) {unknown}; known: {', '.join(FIGS)}")
    configure(args.data, args.out, args.families, args.released)
    rows = load()
    for name in args.figures or list(FIGS):
        FIGS[name](rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
