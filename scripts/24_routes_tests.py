"""Tests of the routes to invariance (B1-B4) and of A1-pairs.

QLIKE, HLN-corrected Diebold-Mariano test on daily cross-sectional average loss differentials, 5%, Holm within family;
in-range portfolios. Writes results*/main/tables/hypotheses_routes.csv and, except for the 54-stock subsample,
routes_levels.csv for the current universe (PVOL_UNIVERSE).
"""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pvol.config import RESULTS, UNIVERSE_NAME  # noqa: E402
from pvol.evaluate import daily_loss, dm_test, group_index, load_run, qlike  # noqa: E402

_spec = importlib.util.spec_from_file_location("hyp", Path(__file__).with_name("15_hypotheses.py"))
hyp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hyp)

SYM = ["DeepSets", "PENet", "SetTransformer", "SetGRU", "EQCov"]
CP = {"DeepSets": "MLP", "PENet": "MLP", "SetTransformer": "MLP", "EQCov": "MLP", "SetGRU": "GRU"}


def comparisons(L, ci, h, fam, block, pairs):
    rows = []
    for a, b in pairs:
        if a in L and b in L:
            d = daily_loss(L[a], ci) - daily_loss(L[b], ci)
            stat, p = dm_test(d, h)
            rows.append(dict(hypothesis=fam, h=h, block=block, model=a, benchmark=b, mean_diff=np.mean(d), dm=stat, p=p))
    return hyp.family(rows)


if __name__ == "__main__":
    rows, levels = [], []
    for h in (1, 5):
        if not (RESULTS / "main" / f"h{h}" / "targets.npy").exists():
            continue
        R = load_run("main", h)
        ci = hyp.cols_of(group_index(R["ports"]), "in-range")
        L = {m: qlike(R["y"], f) for m, f in R["F"].items()}
        if UNIVERSE_NAME == "stocks54":  # B4: H1 on the 54-stock subsample
            rows += comparisons(L, ci, h, "B4", "in-range", [(a, CP[a]) for a in SYM])
            continue
        e = np.nanmean(L["EWMA"][:, ci])
        for m in sorted(L):
            levels.append(dict(h=h, model=m, qlike_rel_ewma=np.nanmean(L[m][:, ci]) / e))
        # B3: symmetric network vs counterpart averaged over 8 orderings at prediction time
        rows += comparisons(L, ci, h, "B3", "in-range", [(a, CP[a] + "-TTA") for a in SYM])
        rows += comparisons(L, ci, h, "B3 (aux: TTA vs single ordering)", "in-range",
                            [(m + "-TTA", m) for m in ("MLP", "GRU")])
        if h != 5:
            continue
        for n in (50, 500):  # B1
            rows += comparisons(L, ci, h, "B1a", f"in-range, pool {n}, shuffled counterpart",
                                [(f"{a}-P{n}", f"{CP[a]}-P{n}s") for a in SYM])
        rows += comparisons(L, ci, h, "B1b", "in-range, shuffled vs fixed order",
                            [(f"{m}-P{n}s", f"{m}-P{n}") for n in (50, 500) for m in ("MLP", "GRU")])
        rows += comparisons(L, ci, h, "B2", "in-range, raw inputs",
                            [(f"{a}-RAW", "MLP-sorted-RAW") for a in ("DeepSets", "PENet", "SetTransformer")])
        for n in (50, 500):
            rows += comparisons(L, ci, h, "B2", f"in-range, pool {n}",
                                [(f"{a}-P{n}", f"MLP-sorted-P{n}") for a in ("DeepSets", "PENet", "SetTransformer", "EQCov")])
        rows += comparisons(L, ci, h, "B2 (aux: sorted vs MLP)", "in-range",
                            [("MLP-sorted-RAW", "MLP-RAW"), ("MLP-sorted-P50", "MLP-P50"), ("MLP-sorted-P500", "MLP-P500")])
        rows += comparisons(L, ci, h, "A1-pairs", "in-range, raw inputs", [("DeepSetsP-RAW", "MLP-RAW")])
        rows += comparisons(L, ci, h, "A1-pairs (aux)", "in-range",
                            [("DeepSetsP", "MLP"), ("DeepSetsP-RAW", "DeepSets-RAW")])
    out = RESULTS / "main" / "tables"
    pd.DataFrame(rows).to_csv(out / "hypotheses_routes.csv", index=False)
    if levels:
        pd.DataFrame(levels).to_csv(out / "routes_levels.csv", index=False)
    pd.set_option("display.width", 220)
    if rows:
        print(pd.DataFrame(rows)[["hypothesis", "h", "block", "model", "benchmark", "mean_diff", "dm", "p_holm", "verdict"]]
              .round(4).to_string(index=False))
