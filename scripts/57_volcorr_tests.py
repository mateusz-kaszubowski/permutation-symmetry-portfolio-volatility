"""Tests of K15 (per-asset volatility network with the EWMA correlation, scripts/56_volcorr.py).

K15a: each of the five symmetric networks vs VolCorr-asset, Holm over the five.
K15b: Linear vs VolCorr-asset.
K15c: VolCorr (trained on portfolios) vs VolCorr-asset (trained on single assets).
In-range portfolios, every h, DM-HLN; best classical model vs VolCorr-asset descriptively.
Writes results*/main/tables/hypotheses_volcorr.csv.

Usage: [PVOL_UNIVERSE=stocks] python scripts/57_volcorr_tests.py
"""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pvol.config import HORIZONS, RESULTS  # noqa: E402
from pvol.evaluate import CLASSICAL, group_index, load_run, qlike  # noqa: E402

_spec = importlib.util.spec_from_file_location("hyp", Path(__file__).with_name("15_hypotheses.py"))
hyp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hyp)

REF = "VolCorr-asset"

if __name__ == "__main__":
    rows = []
    for h in HORIZONS:
        R = load_run("main", h)
        cols = hyp.cols_of(group_index(R["ports"]), "in-range")
        L = {m: qlike(R["y"], f) for m, f in R["F"].items()}
        ewma = np.nanmean(L["EWMA"][:, cols])

        def row(label, a, b):
            res, _ = hyp.compare(L[a], L[b], cols, h)
            return dict(hypothesis=label, h=h, block="in-range", model=a, benchmark=b, rel_a=res["loss_a"] / ewma,
                        rel_b=res["loss_b"] / ewma, **res)

        rows += hyp.family([row("K15a", a, REF) for a in hyp.SYMMETRIC])
        rows += hyp.family([row("K15b", "Linear", REF)])
        rows += hyp.family([row("K15c", "VolCorr", REF)])
        bench = min((m for m in CLASSICAL if m in L), key=lambda m: np.nanmean(L[m][:, cols]))
        rows.append(dict(row("K15 (descriptive)", REF, bench), p_holm=np.nan, verdict="--"))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "main" / "tables" / "hypotheses_volcorr.csv", index=False)
    pd.set_option("display.width", 220)
    print(df[["hypothesis", "h", "model", "benchmark", "rel_a", "rel_b", "p", "p_holm", "verdict"]].round(4)
          .to_string(index=False))
