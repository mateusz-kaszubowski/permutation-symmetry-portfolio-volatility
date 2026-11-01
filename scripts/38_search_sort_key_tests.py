"""Tests of the broader search (K3, K14) and the sort key (K4).

QLIKE, HLN-corrected Diebold-Mariano test on daily cross-sectional average loss differentials, 5%, Holm within family;
in-range portfolios; seed ensembles (seeds 0-2; P50: 0-1).
K3a (H1):  symmetric network vs counterpart, both with the settings of the broader search, Holm over 5 per h.
K3b (C1): each of the 7 networks (-wide) vs Linear-wide, Holm over 7 per h.
K3c (H2):  symmetric network (-wide) vs the best classical model of all fourteen (ex post), Holm over 5 per h.
K3d (aux):   model -wide vs the same model in the main protocol, Holm over 8 per h; best validation epoch of every run.
K14a-c:      K3a-c with the settings selected at the horizon of evaluation (-wideh), h = 1 and 22.
K4a:         MLP-idsort vs MLP in the main design, with raw inputs and with a pool of 50 portfolios (h = 5), Holm over 3.
K4b:         MLP-sorted vs MLP-idsort in the same three conditions, Holm over 3.
Writes results*/main/tables/hypotheses_search.csv, search_levels.csv and search_epochs.csv for the current universe.

Usage: [PVOL_UNIVERSE=stocks] python scripts/38_search_sort_key_tests.py
"""
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pvol.config import RESULTS  # noqa: E402
from pvol.evaluate import CLASSICAL, daily_loss, dm_test, group_index, load_run, qlike  # noqa: E402

_spec = importlib.util.spec_from_file_location("hyp", Path(__file__).with_name("15_hypotheses.py"))
hyp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hyp)
SYM, CP = hyp.SYMMETRIC, hyp.COUNTERPART
NETS = ["MLP", "GRU"] + SYM
WIDE_MODELS = NETS + ["Linear"]


def comparisons(L, ci, h, fam, block, pairs):
    rows = []
    for a, b in pairs:
        if a in L and b in L:
            d = daily_loss(L[a], ci) - daily_loss(L[b], ci)
            stat, p = dm_test(d, h)
            rows.append(dict(hypothesis=fam, h=h, block=block, model=a, benchmark=b, loss_a=np.nanmean(L[a][:, ci]),
                             loss_b=np.nanmean(L[b][:, ci]), mean_diff=np.mean(d), dm=stat, p=p))
    return hyp.family(rows)


def best_epochs(h):
    out = []
    for f in (RESULTS / "main" / f"h{h}" / "info").glob("*__s*__y*.json"):
        d = json.loads(f.read_text())
        tag = d.get("tag", d["model"])
        base = tag.rsplit("-wide", 1)[0]
        if base not in WIDE_MODELS or tag not in (base, base + "-wide", base + "-wideh") or not d.get("history") or d["seed"] > 2:
            continue
        out.append(dict(h=h, model=base, protocol={base: "main", base + "-wide": "wide", base + "-wideh": "wideh"}[tag],
                        best_epoch=1 + int(np.argmin([r[2] for r in d["history"]])), epochs=d["epochs"]))
    return out


if __name__ == "__main__":
    rows, levels, ep = [], [], []
    for h in (1, 5, 22):
        if not (RESULTS / "main" / f"h{h}" / "targets.npy").exists():
            continue
        R = load_run("main", h)
        ci = hyp.cols_of(group_index(R["ports"]), "in-range")
        L = {m: qlike(R["y"], f) for m, f in R["F"].items()}
        e = np.nanmean(L["EWMA"][:, ci])
        for m in sorted(L):
            if m.endswith(("-wide", "-wideh")) or "idsort" in m or m in WIDE_MODELS:
                levels.append(dict(h=h, model=m, qlike_rel_ewma=np.nanmean(L[m][:, ci]) / e))
        if all(m + "-wide" in L for m in WIDE_MODELS):
            rows += comparisons(L, ci, h, "K3a", "in-range", [(a + "-wide", CP[a] + "-wide") for a in SYM])
            rows += comparisons(L, ci, h, "K3b", "in-range", [(a + "-wide", "Linear-wide") for a in NETS])
            bench = min((m for m in CLASSICAL if m in L), key=lambda m: np.nanmean(L[m][:, ci]))
            rows += comparisons(L, ci, h, "K3c", "in-range", [(a + "-wide", bench) for a in SYM])
            rows += comparisons(L, ci, h, "K3d (aux: broader search vs main protocol)", "in-range",
                                [(m + "-wide", m) for m in WIDE_MODELS])
        if h != 5 and all(m + "-wideh" in L for m in WIDE_MODELS):
            bench = min((m for m in CLASSICAL if m in L), key=lambda m: np.nanmean(L[m][:, ci]))
            rows += comparisons(L, ci, h, "K14a", "in-range", [(a + "-wideh", CP[a] + "-wideh") for a in SYM])
            rows += comparisons(L, ci, h, "K14b", "in-range", [(a + "-wideh", "Linear-wideh") for a in NETS])
            rows += comparisons(L, ci, h, "K14c", "in-range", [(a + "-wideh", bench) for a in SYM])
        ep += best_epochs(h)
        if h == 5:
            conds = [("", "main design", None), ("-RAW", "raw inputs", None), ("-P50", "pool 50", [0, 1])]
            pairs1, pairs2 = [], []
            for sfx, block, seeds in conds:
                Ls = L if seeds is None else {m: qlike(R["y"], f) for m, f in load_run("main", h, seeds=seeds)["F"].items()}
                pairs1 += comparisons(Ls, ci, h, "K4a", block, [(f"MLP-idsort{sfx}", f"MLP{sfx}")])
                pairs2 += comparisons(Ls, ci, h, "K4b", block, [(f"MLP-sorted{sfx}", f"MLP-idsort{sfx}")])
            for fam in (pairs1, pairs2):  # Holm over the three conditions
                for r in fam:
                    r.pop("p_holm", None)
                rows += hyp.family(fam)
    out = RESULTS / "main" / "tables"
    df = pd.DataFrame(rows)
    df.to_csv(out / "hypotheses_search.csv", index=False)
    pd.DataFrame(levels).to_csv(out / "search_levels.csv", index=False)
    E = pd.DataFrame(ep)
    if len(E):
        E = (E.assign(le2=E.best_epoch <= 2).groupby(["h", "model", "protocol"])
             .agg(runs=("best_epoch", "size"), median_best_epoch=("best_epoch", "median"), share_le2=("le2", "mean"),
                  median_epochs=("epochs", "median")).reset_index())
        E.to_csv(out / "search_epochs.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    if len(df):
        print(df[["hypothesis", "h", "block", "model", "benchmark", "loss_a", "loss_b", "mean_diff", "dm", "p_holm",
                  "verdict"]].round(4).to_string(index=False))
    if len(E):
        print(E.round(3).to_string(index=False))
