"""Scoreboard of all hypotheses for both universes, built from results*/main/tables/hypotheses*.csv.

Each cell counts the pairs (or comparisons) of the family whose verdict is 'more accurate', out of the pairs tested;
a trailing '-k' marks k comparisons with the opposite significant verdict ('less accurate').
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


OUT = Path(__file__).resolve().parents[1] / "paper" / "tables"
ROOT_DIR = Path(__file__).resolve().parents[1]
BASE = ROOT_DIR / "results"
UNIVERSES = [("ETF", BASE), ("Stocks", ROOT_DIR / "results_stocks")]
E, R = "economic_tests.csv", "hypotheses_routes.csv"

LINEAR_TESTS = "hypotheses_linear.csv"
# (label, file, filter[, suffix of the results directory that replaces the stock universe]); file None = section title
# main text: the families of the three research questions
ROWS = [
    ("RQ1: architecture against learned symmetry", None, None),
    ("A1: raw inputs", "hypotheses_conditions.csv", dict(hypothesis="A1")),
    ("A2: pool of 50 portfolios", "hypotheses_conditions.csv", dict(hypothesis="A2", block="in-range, pool 50")),
    ("A2: pool of 500 portfolios", "hypotheses_conditions.csv", dict(hypothesis="A2", block="in-range, pool 500")),
    ("RQ2: simpler routes to invariance", None, None),
    ("B1: pool of 50 vs shuffled counterpart", R, dict(hypothesis="B1a", block="in-range, pool 50, shuffled counterpart")),
    ("B1: pool of 500 vs shuffled counterpart", R,
     dict(hypothesis="B1a", block="in-range, pool 500, shuffled counterpart")),
    ("B1: shuffling improves the counterpart$^{a}$", R, dict(hypothesis="B1b")),
    ("B2: vs MLP-sorted, raw inputs", R, dict(hypothesis="B2", block="in-range, raw inputs")),
    ("B2: vs MLP-sorted, pool of 50", R, dict(hypothesis="B2", block="in-range, pool 50")),
    ("B2: vs MLP-sorted, pool of 500", R, dict(hypothesis="B2", block="in-range, pool 500")),
    ("B3: vs counterpart averaged over orderings, $h=1$", R, dict(hypothesis="B3", h=1)),
    ("B3: vs counterpart averaged over orderings, $h=5$", R, dict(hypothesis="B3", h=5)),
    ("B4: H1 on 54 stocks, $h=1$", R, dict(hypothesis="B4", h=1), "_stocks54"),
    ("B4: H1 on 54 stocks, $h=5$", R, dict(hypothesis="B4", h=5), "_stocks54"),
    ("H1 with ten seeds, $h=1$", LINEAR_TESTS, dict(hypothesis="K1-H1", h=1)),
    ("B3 with ten seeds, $h=1$", LINEAR_TESTS, dict(hypothesis="K1-B3", h=1)),
    ("B4 with ten seeds, $h=1$", LINEAR_TESTS, dict(hypothesis="K1-B4", h=1), "_stocks54"),
    ("RQ3: flexibility and classical benchmarks", None, None),
    ("C2: vs invariant linear, raw inputs", "hypotheses_raw.csv", dict(hypothesis="C2")),
]
# the main comparisons, shown in the main text as a forest plot (scripts/53_forest_plot.py)
ROWS_MAIN = [
    *[(f"H1: symmetric vs counterpart, $h={h}$", "hypotheses.csv", dict(hypothesis="H1", h=h, block="in-range"))
      for h in (1, 5, 22)],
    *[(f"C1: symmetric vs invariant linear, $h={h}$", LINEAR_TESTS, dict(hypothesis="C1", h=h, benchmark="Linear"))
      for h in (1, 5, 22)],
    *[(f"H2: symmetric vs best classical, $h={h}$", "hypotheses.csv", dict(hypothesis="H2", h=h, block="in-range"))
      for h in (1, 5, 22)],
]
# secondary families (supplementary material, written by scripts/46_supplement_tables.py)
ROWS_SUPP = [
    *ROWS_MAIN,
    *[(f"H2 (classical models without range-based variances), $h={h}$", "hypotheses_h2norange.csv",
       dict(hypothesis="H2", h=h, block="in-range")) for h in (1, 5, 22)],
    ("H1: minimum-variance portfolios", E, dict(hypothesis="H1-minvar")),
    *[(f"H1: volatility targeting, $h={h}$", E, dict(hypothesis="H1-voltarget", h=h)) for h in (1, 5, 22)],
    ("H2: minimum-variance portfolios", E, dict(hypothesis="H2-minvar")),
    *[(f"H2: volatility targeting, $h={h}$", E, dict(hypothesis="H2-voltarget", h=h)) for h in (1, 5, 22)],
    ("G1: advantage grows with $N$", "hypotheses.csv", dict(hypothesis="G1a", h=5)),
    ("G1: long--short weights, $h=5$", "hypotheses.csv", dict(hypothesis="G1b", h=5)),
    ("G1: unseen assets, $h=5$", "hypotheses.csv", dict(hypothesis="G1c", h=5)),
    ("G2: short training history", "hypotheses.csv", dict(hypothesis="G2")),
    ("A1: larger advantage than with summaries", "hypotheses_conditions.csv", dict(hypothesis="A1d")),
    ("A2: larger advantage with 50 than 500", "hypotheses_conditions.csv", dict(hypothesis="A2d")),
    ("A3: portfolios with $N>20$", "hypotheses_conditions.csv", dict(hypothesis="A3")),
    ("Deep Sets + pairs vs MLP, raw inputs", R, dict(hypothesis="A1-pairs")),
    *[(f"C3: symmetric vs GBM, $h={h}$", LINEAR_TESTS, dict(hypothesis="C3", h=h, benchmark="GBM")) for h in (1, 5, 22)],
]


IN_RANGE = ["N5", "N10", "N20", "N10-conc", "N10-EW"]


def ewma_loss(root, h, _cache={}):
    """Mean in-range QLIKE of EWMA (the unit of the effect sizes: one percentage point of the EWMA loss = 0.01 of it)."""
    key = (str(root), h)
    if key not in _cache:
        d = root / "main" / f"h{h}"
        if not (d / "targets.npy").exists():
            _cache[key] = None
            return None
        y = np.maximum(np.load(d / "targets.npy"), 1e-12)
        f = np.exp(np.load(d / "preds" / "EWMA.npy").astype(np.float64))
        ports = json.loads((root / "main" / "portfolios.json").read_text())
        cols = [j for j, p in enumerate(ports) if p["group"] in IN_RANGE]
        r = y[:, cols] / f[:, cols]
        _cache[key] = float(np.mean(r - np.log(r) - 1))
    return _cache[key]


def cell(root, fname, flt):
    """(verdict count, effect range, range of Holm-adjusted p-values): effect = mean daily loss differential (first model
    minus second) in percentage points of the in-range EWMA loss; negative favors the symmetric network. Economic
    comparisons: no effect size."""
    f = root / "main" / "tables" / fname
    if not f.exists():
        return "--", "", ""
    d = pd.read_csv(f)
    for k, v in flt.items():
        if k not in d.columns:
            return "--", "", ""
        d = d[d[k] == v]
    if not len(d):
        return "--", "", ""
    good = (d.verdict == "more accurate").sum()
    bad = (d.verdict == "less accurate").sum()
    count = f"{good}/{len(d)}" + (f" ($-${bad})" if bad else "")
    eff = ""
    if fname != E and "mean_diff" in d.columns:
        h = int(d.h.iloc[0]) if "h" in d.columns and d.h.notna().all() else 5
        e = ewma_loss(root, h)
        if e:
            v = 100 * d.mean_diff.to_numpy() / e
            lo, hi = v.min(), v.max()
            fmt = lambda x: f"${x:.1f}$".replace("$-", "$-")
            eff = fmt(lo) if abs(hi - lo) < 0.05 else f"{fmt(lo)} to {fmt(hi)}"
    pv = ""
    if "p_holm" in d.columns:
        fp = lambda x: "$<$0.001" if x < 0.001 else f"{x:.3f}"
        lo, hi = d.p_holm.min(), d.p_holm.max()
        pv = fp(lo) if fp(lo) == fp(hi) else f"{fp(lo)}--{fp(hi)}"
    return count, eff, pv


def write(rows, name, label, caption):
    L = ["\\begin{table}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}",
         "\\caption{" + caption + " Verdict: comparisons decided in favor of the symmetric network out of those tested; "
         "($-k$): $k$ against it; `--': not run. Effect: range of the mean QLIKE differential in pp, "
         "negative favors the symmetric network" + ("" if name == "scoreboard" else " (growth with $N$, G2: the stated condition); no effect size for the economic "
                      "comparisons") + ". $p_{\\text{Holm}}$: range of the Holm-adjusted $p$-values."
         + ("" if name != "scoreboard" else " $^{a}$Verdict and effect favor the shuffled counterpart.")
         + "}\\label{" + label + "}",
         "\\begin{tabular}{lcrrcrr}\\toprule",
         " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{u}}}" for u, _ in UNIVERSES) + "\\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
         "Hypothesis & " + " & ".join("verdict & effect (pp) & $p_{\\text{Holm}}$" for _ in UNIVERSES)
         + "\\\\ \\midrule"]
    for i, (lbl, fname, flt, *alt) in enumerate(rows):
        if fname is None:
            L += ([] if i == 0 else ["\\midrule"]) + [f"\\multicolumn{{7}}{{l}}{{\\emph{{{lbl}}}}}\\\\"]
            continue
        roots = [root for _, root in UNIVERSES]
        if alt:  # run on a sub-universe only (B4): no ETF cell, stock cell from that directory
            roots = [None, Path(str(BASE) + alt[0])]
        cells = [cell(r, fname, flt) if r else ("--", "", "") for r in roots]
        L.append(f"{lbl} & " + " & ".join(f"{c} & {e} & {p}" for c, e, p in cells) + "\\\\")
    L += ["\\bottomrule\\end{tabular}", "\\end{table}"]
    (OUT / f"{name}.tex").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[4:-2]))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    write(ROWS, "scoreboard", "tab:scoreboard",
          "Verdicts of the main exploratory hypotheses. B4 and its repetition with ten seeds use 54 stocks.")
