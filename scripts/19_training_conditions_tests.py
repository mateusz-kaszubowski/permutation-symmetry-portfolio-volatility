"""Tests of A1-A3 with the decision rule of H1.

A1   raw inputs: symmetric-RAW vs counterpart-RAW (in-range), Holm over 4 pairs;
A1d  (symmetric - counterpart) differential lower in RAW than in the main experiment (difference-in-differences);
A2   fixed pool of P portfolios: symmetric-PP vs counterpart-PP (in-range), Holm over 5 pairs, for P = 50 and 500;
A2d  advantage larger for P = 50 than for P = 500 (difference-in-differences);
A3   training on N <= 54: symmetric-N54 vs MLP-N54 in the larger-N block (Holm over 4); in-range reported as well.
Usage: python scripts/19_training_conditions_tests.py            (PVOL_UNIVERSE=stocks for the stock universe)
"""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pvol.config import RESULTS  # noqa: E402
from pvol.evaluate import daily_loss, dm_test, group_index, load_run, qlike  # noqa: E402

_spec = importlib.util.spec_from_file_location("hyp", Path(__file__).with_name("15_hypotheses.py"))
hyp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hyp)

H = 5
PAIRS_RAW = [("DeepSets", "MLP"), ("PENet", "MLP"), ("SetTransformer", "MLP"), ("SetGRU", "GRU")]
PAIRS_POOL = PAIRS_RAW + [("EQCov", "MLP")]
PAIRS_N54 = [("DeepSets", "MLP"), ("PENet", "MLP"), ("SetTransformer", "MLP"), ("EQCov", "MLP")]


def diff(L, a, b, cols):
    return daily_loss(L[a], cols) - daily_loss(L[b], cols)


def test(d):
    stat, p = dm_test(d, H)
    return dict(mean_diff=np.mean(d), dm=stat, p=p)


if __name__ == "__main__":
    R = load_run("main", H)
    y, F = R["y"], R["F"]
    gidx = group_index(R["ports"])
    L = {m: qlike(y, f) for m, f in F.items()}
    ci = hyp.cols_of(gidx, "in-range")
    cl = hyp.cols_of(gidx, "larger-N")
    e = np.nanmean(L["EWMA"][:, ci])
    # larger-N levels on the portfolios the 54-slot MLP can score (all of them for ETFs, N = 30, 40 for stocks)
    cs = cl[np.isfinite(L["MLP-N54"][:, cl]).all(0)] if "MLP-N54" in L else cl
    rows, levels = [], []
    for m in sorted(L):
        if any(m.endswith(s) for s in ("-RAW", "-P50", "-P500", "-N54")) or m in {a for p in PAIRS_POOL for a in p}:
            levels.append(dict(model=m, qlike_rel_ewma_in_range=np.nanmean(L[m][:, ci]) / e,
                               qlike_rel_ewma_larger=np.nanmean(L[m][:, cs]) / np.nanmean(L["EWMA"][:, cs])
                               if np.isfinite(L[m][:, cs]).all() else np.nan))

    def have(*ms):
        return all(m in L for m in ms)

    # A1 and A1d
    fam, famd = [], []
    for a, b in PAIRS_RAW:
        if have(a + "-RAW", b + "-RAW"):
            d_raw = diff(L, a + "-RAW", b + "-RAW", ci)
            fam.append(dict(hypothesis="A1", block="in-range", model=a + "-RAW", benchmark=b + "-RAW", **test(d_raw)))
            if have(a, b):
                famd.append(dict(hypothesis="A1d", block="in-range, RAW minus main", model=a, benchmark=b,
                                 **test(d_raw - diff(L, a, b, ci))))
    rows += hyp.family(fam) + hyp.family(famd)
    # A2 and A2d
    for P in (50, 500):
        fam = []
        for a, b in PAIRS_POOL:
            if have(f"{a}-P{P}", f"{b}-P{P}"):
                fam.append(dict(hypothesis="A2", block=f"in-range, pool {P}", model=f"{a}-P{P}", benchmark=f"{b}-P{P}",
                                **test(diff(L, f"{a}-P{P}", f"{b}-P{P}", ci))))
        rows += hyp.family(fam)
    fam = []
    for a, b in PAIRS_POOL:
        if have(f"{a}-P50", f"{b}-P50", f"{a}-P500", f"{b}-P500"):
            D = diff(L, f"{a}-P50", f"{b}-P50", ci) - diff(L, f"{a}-P500", f"{b}-P500", ci)
            fam.append(dict(hypothesis="A2d", block="in-range, pool 50 minus pool 500", model=a, benchmark=b, **test(D)))
    rows += hyp.family(fam)
    # A3
    for block, cols in (("larger-N", cl), ("in-range", ci)):
        fam = []
        for a, b in PAIRS_N54:
            if have(a + "-N54", b + "-N54"):
                # compare only portfolios both models can score (the 54-slot MLP cannot score N > 54, e.g. 122 stocks)
                ok = np.isfinite(L[a + "-N54"][:, cols]).all(0) & np.isfinite(L[b + "-N54"][:, cols]).all(0)
                fam.append(dict(hypothesis="A3" if block == "larger-N" else "A3 (aux)", block=block, model=a + "-N54",
                                benchmark=b + "-N54", n_portfolios=int(ok.sum()),
                                **test(diff(L, a + "-N54", b + "-N54", cols[ok]))))
        rows += hyp.family(fam)
    out = RESULTS / "main" / "tables"
    pd.DataFrame(rows).to_csv(out / "hypotheses_conditions.csv", index=False)
    pd.DataFrame(levels).to_csv(out / "conditions_levels.csv", index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(levels).round(3).to_string(index=False))
    if rows:
        print(pd.DataFrame(rows)[["hypothesis", "block", "model", "benchmark", "mean_diff", "dm", "p", "p_holm", "verdict"]]
              .round(4).to_string(index=False))
