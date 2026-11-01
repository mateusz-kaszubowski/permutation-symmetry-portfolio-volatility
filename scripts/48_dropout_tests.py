"""Tests of K10, the dropout control of MLP-sorted.

K10a: B2 repeated against MLP-sorted with dropout 0.3 (raw inputs: Deep Sets, PE-Net, Set Transformer; pools of 50 and
     500: also EQ-Cov), Holm within each design as in B2.
K10b (aux): MLP-sorted with dropout 0.3 against the MLP under the same condition (raw inputs, pools, main design).
QLIKE, HLN-corrected Diebold-Mariano test on daily cross-sectional average loss differentials, in-range portfolios, h=5.
Writes results*/main/tables/hypotheses_dropout.csv for the current universe.

Usage: [PVOL_UNIVERSE=stocks] python scripts/48_dropout_tests.py
"""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402

from pvol.config import RESULTS  # noqa: E402
from pvol.evaluate import group_index, load_run, qlike  # noqa: E402

_spec = importlib.util.spec_from_file_location("routes", Path(__file__).with_name("24_routes_tests.py"))
routes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(routes)

if __name__ == "__main__":
    h = 5
    R = load_run("main", h)
    ci = routes.hyp.cols_of(group_index(R["ports"]), "in-range")
    L = {m: qlike(R["y"], f) for m, f in R["F"].items()}
    rows = routes.comparisons(L, ci, h, "K10a", "in-range, raw inputs",
                             [(f"{a}-RAW", "MLP-sorted-RAW-d3") for a in ("DeepSets", "PENet", "SetTransformer")])
    for n in (50, 500):
        rows += routes.comparisons(L, ci, h, "K10a", f"in-range, pool {n}",
                                  [(f"{a}-P{n}", f"MLP-sorted-P{n}-d3")
                                   for a in ("DeepSets", "PENet", "SetTransformer", "EQCov")])
    rows += routes.comparisons(L, ci, h, "K10b (aux: sorted d0.3 vs MLP)", "in-range",
                              [("MLP-sorted-RAW-d3", "MLP-RAW"), ("MLP-sorted-P50-d3", "MLP-P50"),
                               ("MLP-sorted-P500-d3", "MLP-P500"), ("MLP-sorted-d3", "MLP")])
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "main" / "tables" / "hypotheses_dropout.csv", index=False)
    pd.set_option("display.width", 220)
    print(df[["hypothesis", "block", "model", "benchmark", "mean_diff", "dm", "p_holm", "verdict"]].round(4)
          .to_string(index=False))
