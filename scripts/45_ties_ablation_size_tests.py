"""Tests of the checks K5-K7.

QLIKE, HLN-corrected Diebold-Mariano test on daily cross-sectional average loss differentials, 5%, Holm within family;
in-range portfolios; seed ensembles (seeds 0-2).
K5a: MLP-tsort vs MLP-sorted, Holm over the 3 horizons.      K5b: MLP-tsort vs MLP, Holm over the 3 horizons.
K6:  each Linear variant vs Linear, Holm over 5 per h.
K7:  Deep Sets vs MLP-small and DeepSets-L vs MLP, Holm over 2 per h; H1 Deep Sets vs MLP repeated for reference.
Writes results*/main/tables/hypotheses_checks.csv and checks_levels.csv for the current universe.

Usage: [PVOL_UNIVERSE=stocks] python scripts/45_ties_ablation_size_tests.py
"""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pvol.config import RESULTS  # noqa: E402
from pvol.evaluate import group_index, load_run, qlike  # noqa: E402

_spec = importlib.util.spec_from_file_location("hyp", Path(__file__).with_name("15_hypotheses.py"))
hyp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hyp)
_spec = importlib.util.spec_from_file_location("search", Path(__file__).with_name("38_search_sort_key_tests.py"))
search = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(search)

ABLATION_MODELS = ["Linear-noasset", "Linear-noeuler", "Linear-nopairs", "Linear-noport", "Linear-portonly"]
LEVELS = ["MLP", "MLP-sorted", "MLP-tsort", "MLP-small", "DeepSets", "DeepSets-L", "Linear"] + ABLATION_MODELS + ["HEAVY"]

if __name__ == "__main__":
    rows, levels, k5a, k5b = [], [], [], []
    for h in (1, 5, 22):
        R = load_run("main", h)
        ci = hyp.cols_of(group_index(R["ports"]), "in-range")
        L = {m: qlike(R["y"], f) for m, f in R["F"].items()}
        e = np.nanmean(L["EWMA"][:, ci])
        for m in LEVELS:
            if m in L:
                levels.append(dict(h=h, model=m, qlike_rel_ewma=np.nanmean(L[m][:, ci]) / e))
        k5a += search.comparisons(L, ci, h, "K5a", "in-range", [("MLP-tsort", "MLP-sorted")])
        k5b += search.comparisons(L, ci, h, "K5b", "in-range", [("MLP-tsort", "MLP")])
        rows += search.comparisons(L, ci, h, "K6", "in-range", [(m, "Linear") for m in ABLATION_MODELS])
        rows += search.comparisons(L, ci, h, "K7", "in-range", [("DeepSets", "MLP-small"), ("DeepSets-L", "MLP")])
        rows += search.comparisons(L, ci, h, "H1 (reference)", "in-range", [("DeepSets", "MLP")])
    for fam in (k5a, k5b):  # Holm over the three horizons
        for r in fam:
            r.pop("p_holm", None)
        rows += hyp.family(fam)
    out = RESULTS / "main" / "tables"
    df = pd.DataFrame(rows)
    df["diff_pp"] = 100 * (df.loss_a - df.loss_b) / df.loss_b
    df.to_csv(out / "hypotheses_checks.csv", index=False)
    lv = pd.DataFrame(levels)
    lv.to_csv(out / "checks_levels.csv", index=False)
    pd.set_option("display.width", 250)
    print(df[["hypothesis", "h", "model", "benchmark", "loss_a", "loss_b", "diff_pp", "dm", "p_holm", "verdict"]]
          .round(4).to_string(index=False))
    print(lv.pivot(index="model", columns="h", values="qlike_rel_ewma").round(3))
