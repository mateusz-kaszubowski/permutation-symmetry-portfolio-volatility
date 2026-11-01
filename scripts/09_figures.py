"""Figures for the paper (vector PDF, drawn at the final printed width so that fonts are not scaled).

Style: text set by LaTeX (usetex: STIX, Type 1 fonts with correct text encoding); Okabe-Ito colors; panel titles
(a), (b), ... aligned left; comparisons across networks as dot plots. The forest plot and the learning curves are
drawn by scripts/53_forest_plot.py and scripts/54_learning_curves.py with the same style.
Run once (it reads both universes).
"""
import os
import sys
from pathlib import Path

if os.environ.get("TEXBIN"):  # bin directory of the LaTeX installation used by usetex
    os.environ["PATH"] = os.environ["TEXBIN"] + os.pathsep + os.environ["PATH"]

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pvol.config import ROOT  # noqa: E402

FIG = ROOT / "paper" / "figures"
PREVIEW = ROOT / "results" / "figure_previews"
FIG.mkdir(parents=True, exist_ok=True)
PREVIEW.mkdir(parents=True, exist_ok=True)
UNIVERSES = [("ETF", ROOT / "results"), ("Stock", ROOT / "results_stocks")]  # the figures read both universes

TEXTWIDTH = 6.84  # inches, the width of a two-column figure
COLOR = {"MLP": "#0072B2", "PENet": "#D55E00", "SetTransformer": "#009E73", "EQCov": "#CC79A7", "DeepSets": "#E69F00"}
INK, INK2, GRID, AXIS = "#1a1a1a", "#4d4d4d", "#e6e6e6", "#9a9a9a"
LABEL = {"SetTransformer": "Set Transformer", "DeepSets": "Deep Sets", "EQCov": "EQ-Cov", "PENet": "PE-Net",
         "SetGRU": "Set-GRU", "HARX": "HAR-X", "GJR": "GJR-GARCH", "DCCNL": "DCC-NL"}

plt.rcParams.update({
    "text.usetex": True, "text.latex.preamble": r"\usepackage{stix}", "font.family": "serif", "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.7, "lines.linewidth": 1.6, "lines.markersize": 5,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def lab(m):
    return LABEL.get(m, m)


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(PREVIEW / f"{name}.png", dpi=150)
    plt.close(fig)


# figures
# One message per figure, few large panels, text at 9-10 pt at the printed size.
UCOL = {"ETF": "#0072B2", "Stock": "#D55E00"}


def _levels(root):
    """QLIKE relative to EWMA (in-range, h=5) of every network variant, from the evaluation tables."""
    out = {}
    for f in ("conditions_levels.csv", "routes_levels.csv"):
        p = root / "main" / "tables" / f
        if p.exists():
            d = pd.read_csv(p)
            if "h" in d.columns:
                d = d[d.h == 5]
            col = "qlike_rel_ewma_in_range" if "qlike_rel_ewma_in_range" in d.columns else "qlike_rel_ewma"
            out.update(dict(zip(d.model, d[col])))
    return out


def fig_conditions():
    """How much accuracy each network loses when an aid of the main design is removed: a dot
    plot with one row per network, grouped into unconstrained and symmetric networks."""
    U, S = "unconstrained networks", "symmetric networks"
    panels = [
        ("(a) Fixed pool of 50 training portfolios",
         [(U, "MLP", "MLP", "MLP-P50"), (U, "MLP, shuffled order", "MLP", "MLP-P50s"),
          (U, "MLP-sorted", "MLP-sorted", "MLP-sorted-P50"), (S, "Deep Sets", "DeepSets", "DeepSets-P50"),
          (S, "PE-Net", "PENet", "PENet-P50"), (S, "Set Transformer", "SetTransformer", "SetTransformer-P50"),
          (S, "EQ-Cov", "EQCov", "EQCov-P50")]),
        ("(b) Raw inputs, without the anchor and the portfolio summaries",
         [(U, "MLP", "MLP", "MLP-RAW"), (U, "MLP-sorted", "MLP-sorted", "MLP-sorted-RAW"),
          (S, "Deep Sets", "DeepSets", "DeepSets-RAW"), (S, "Deep Sets + pairs", "DeepSetsP", "DeepSetsP-RAW"),
          (S, "PE-Net", "PENet", "PENet-RAW"), (S, "Set Transformer", "SetTransformer", "SetTransformer-RAW")]),
    ]
    lev = {u: _levels(root) for u, root in UNIVERSES}
    # the reference of every variant is the same network in the main design (MLP-sorted: its own main-design level)
    for u, root in UNIVERSES:
        d = pd.read_csv(root / "main" / "tables" / "losses_h5.csv")
        d = d[d.block == "in-range"]
        lev[u].update({m: v for m, v in zip(d.model, d.qlike_rel_ewma) if m not in lev[u]})
    marker = {"ETF": "o", "Stock": "s"}
    xlim, step = (-3.2, 9.2), 1.0  # step: vertical distance between rows
    layouts = []
    for title, rows in panels:  # y positions: a header line above each group, rows below it
        y, ys, heads, last = 0.0, [], [], None
        for g, *_ in rows:
            if g != last:
                y -= 0.35 if last is None else 0.75
                heads.append((g, y))
                y -= 0.85
                last = g
            ys.append(y)
            y -= step
        layouts.append((title, rows, ys, heads, y + step - 0.6))
    heights = [-lo + 0.5 for *_, lo in layouts]
    fig, axes = plt.subplots(2, 1, figsize=(TEXTWIDTH, 5.0), gridspec_kw=dict(height_ratios=heights, hspace=0.42))
    for ax, (title, rows, ys, heads, lo) in zip(axes, layouts):
        for g, yh in heads:
            if g == U:  # light band behind the unconstrained networks
                yb = [y_ for (g_, *_), y_ in zip(rows, ys) if g_ == U]
                ax.axhspan(min(yb) - 0.5, max(yb) + 0.5, xmin=0, xmax=1, color="#f4f4f4", lw=0, zorder=0)
            ax.text(xlim[0] - 0.02 * (xlim[1] - xlim[0]), yh, g, ha="right", va="center", fontsize=8.5, color=INK2,
                    style="italic")
        for (g, name, ref, var), y_ in zip(rows, ys):
            v = {u: 100 * (lev[u].get(var, np.nan) - lev[u].get(ref, np.nan)) for u, _ in UNIVERSES}
            close = abs(v["ETF"] - v["Stock"]) < 0.35  # nearly equal values: offset vertically so both stay visible
            dy = {"ETF": 0.24 * close, "Stock": -0.24 * close}
            ax.plot([v["ETF"], v["Stock"]], [y_ + dy["ETF"], y_ + dy["Stock"]], color="#c8c8c8", lw=2.2,
                    solid_capstyle="round", zorder=2)
            for u, _ in UNIVERSES:
                ax.plot(v[u], y_ + dy[u], marker=marker[u], ms=7 if u == "ETF" else 6.4, color=UCOL[u], mec="white", mew=1.0,
                        ls="none", zorder=3)
        ax.set_yticks(ys)
        ax.set_yticklabels([r[1] for r in rows])
        ax.tick_params(axis="y", length=0, pad=6, labelsize=9.5, labelcolor=INK)
        ax.set_ylim(lo, 0.2)
        ax.set_xlim(*xlim)
        ax.set_xticks(range(-2, 10, 2))
        ax.axvline(0, color=INK, lw=0.9, zorder=1)
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
        ax.set_title(title, loc="left", fontsize=10, color=INK, pad=4)
    axes[1].set_xlabel("change in QLIKE when the aid is removed (percentage points of the EWMA loss)")
    axes[1].annotate("", xy=(xlim[0] + 0.15, -0.16), xytext=(-0.25, -0.16), xycoords=("data", "axes fraction"),
                     arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.7, shrinkA=0, shrinkB=0))
    axes[1].text(-0.35, -0.21, "more accurate", transform=axes[1].get_xaxis_transform(), ha="right", va="top",
                 fontsize=8.5, color=INK2)
    axes[1].annotate("", xy=(xlim[1] - 0.15, -0.16), xytext=(0.25, -0.16), xycoords=("data", "axes fraction"),
                     arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.7, shrinkA=0, shrinkB=0))
    axes[1].text(0.35, -0.21, "less accurate", transform=axes[1].get_xaxis_transform(), ha="left", va="top",
                 fontsize=8.5, color=INK2)
    axes[1].xaxis.labelpad = 22
    handles = [plt.Line2D([], [], marker=marker[u], ms=7, color=UCOL[u], mec="white", mew=1.0, ls="none")
               for u, _ in UNIVERSES]
    fig.legend(handles, ["ETF universe", "Stock universe"], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0),
               handletextpad=0.3, columnspacing=2.0)
    save(fig, "conditions")


if __name__ == "__main__":
    fig_conditions()
    print("figures written to", FIG)
