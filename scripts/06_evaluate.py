"""Evaluation tables for an experiment: losses by portfolio group and block; for the main experiment also the Model
Confidence Set of every block, the permutation consistency of the networks and the value-at-risk backtest.

Horizons without network forecasts are skipped (the held-out experiment of the stock universe runs at h = 5 only).
Usage: python scripts/06_evaluate.py main [seeds...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pvol.config import HORIZONS, RESULTS  # noqa: E402
from pvol.data import load_panel  # noqa: E402
from pvol.evaluate import CLASSICAL, NEURAL, daily_loss, group_index, load_run, mcs, mse, qlike, var_backtest  # noqa: E402
from pvol.portfolios import load  # noqa: E402

GROUP_SETS = {
    "main": {
        "in-range": ["N5", "N10", "N20", "N10-conc", "N10-EW"],
        "larger-N": ["N30", "N40", "EW-all"],
        "long-short": ["N10-LS"],
        "structured": ["60/40", "AllWeather", "GlobalMulti"],
    },
    "heldout": {"unseen-assets": ["H5", "H10", "H10-EW"]},
}
MAIN_SEEDS = (0, 1, 2)  # seeds 3-9 (K1, stocks at h = 1) enter only the repetition with ten seeds


def write_rows(rows, path):
    """Write the rows as CSV; with no rows (e.g. no networks with permuted forecasts) remove a stale file instead."""
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)
    else:
        path.unlink(missing_ok=True)


def order(models):
    return [m for m in CLASSICAL + NEURAL if m in models]


def evaluate_h(exp, h, seeds, out):
    run = load_run(exp, h, seeds)
    y, F, ports = run["y"], run["F"], run["ports"]
    models = order(F)
    gidx = group_index(ports)
    Lq = {m: qlike(y, F[m]) for m in models}
    Lm = {m: mse(y, F[m]) for m in models}
    rows = []
    # block sets present in this universe (e.g. no structured allocations for the stock universe)
    group_sets = {b: gs for b, gs in GROUP_SETS[exp].items() if any(g in gidx for g in gs)}
    # losses per individual group and per group set
    blocks = {**{g: [g] for g in gidx}, **group_sets}
    for b, groups in blocks.items():
        cols = np.concatenate([gidx[g] for g in groups if g in gidx])
        for m in models:
            q = Lq[m][:, cols]
            if np.isnan(q).all():
                continue
            rows.append(dict(h=h, block=b, model=m, qlike=np.nanmean(q), mse=np.nanmean(Lm[m][:, cols]),
                             coverage=np.isfinite(q).mean()))
    tab = pd.DataFrame(rows)
    ref = tab[tab.model == "EWMA"].set_index("block")
    tab["qlike_rel_ewma"] = tab.qlike / tab.block.map(ref.qlike)
    tab["mse_rel_ewma"] = tab.mse / tab.block.map(ref.mse)
    tab.to_csv(out / f"losses_h{h}.csv", index=False)
    if exp != "main":  # the held-out experiment reports the losses; G1c is tested in scripts/15_hypotheses.py
        return tab

    # Model Confidence Set on the daily cross-sectional average QLIKE, per block
    mcs_rows = []
    for b, groups in group_sets.items():
        cols = np.concatenate([gidx[g] for g in groups if g in gidx])
        daily = {m: daily_loss(Lq[m], cols) for m in models if np.isfinite(Lq[m][:, cols]).all()}
        pv = mcs(pd.DataFrame(daily), size=0.10, h=h)
        mcs_rows += [dict(h=h, block=b, model=m, p_value=v) for m, v in pv.items()]
    pd.DataFrame(mcs_rows).to_csv(out / f"mcs_h{h}.csv", index=False)

    # permutation consistency
    perm_path = RESULTS / exp / "perm_portfolios.json"
    pp = load(perm_path) if perm_path.exists() else []
    src = np.array([p["src"] for p in pp], dtype=int)
    perm_k = np.array([p["perm"] for p in pp], dtype=int)
    d = RESULTS / exp / f"h{h}" / "preds"
    pc = []
    for m in NEURAL + ["MLP-TTA", "GRU-TTA"]:  # with the networks averaged over eight orderings (B3)
        files = sorted(f for f in d.glob(f"{m}__s*__perm.npy") if int(f.name.split("__s")[1].split("__")[0]) in MAIN_SEEDS)
        if not files or not len(pp):
            continue
        stats_m = []
        for f in files:
            s = np.load(f).astype(np.float64)  # (T, n_ports * n_perm)
            n_perm = perm_k.max() + 1
            S = np.stack([s[:, perm_k == k] for k in range(n_perm)], 0)  # (n_perm, T, n_ports)
            spread = S.max(0) - S.min(0)  # range of log forecasts across orderings
            srcs = src[perm_k == 0]
            yq = y[:, srcs]
            ql = np.stack([qlike(yq, np.exp(S[k])).mean() for k in range(n_perm)])
            stats_m.append((np.mean(np.std(S, 0)), np.median(spread), np.percentile(spread, 99), ql.max() - ql.min()))
        a = np.array(stats_m).mean(0)
        pc.append(dict(h=h, model=m, mean_sd_logvar=a[0], median_range_logvar=a[1], p99_range_logvar=a[2],
                       qlike_range_over_orderings=a[3]))
    write_rows(pc, out / f"perm_h{h}.csv")

    # VaR backtest (h = 1, in-range portfolios)
    if h == 1:
        ci = np.concatenate([gidx[g] for g in group_sets["in-range"] if g in gidx])
        R = load_panel()["R"].astype(np.float64)
        r_next = np.column_stack([R[run["origins"] + 1][:, q["ids"]] @ np.asarray(q["weights"]) for q in ports])
        vr = []
        for alpha in (0.01, 0.05):
            for m in models:
                res = var_backtest(r_next[:, ci], F[m][:, ci], alpha)
                vr.append(dict(alpha=alpha, model=m, **res))
        pd.DataFrame(vr).to_csv(out / "var_h1.csv", index=False)
    return tab


if __name__ == "__main__":
    exp = sys.argv[1] if len(sys.argv) > 1 else "main"
    seeds = [int(s) for s in sys.argv[2:]] or [0, 1, 2]
    out = RESULTS / exp / "tables"
    out.mkdir(parents=True, exist_ok=True)
    for h in HORIZONS:
        d = RESULTS / exp / f"h{h}"
        if not (d / "targets.npy").exists() or not any((d / "preds").glob("*__s*.npy")):
            continue
        tab = evaluate_h(exp, h, seeds, out)
        piv = tab[tab.block.isin(GROUP_SETS[exp])].pivot(index="model", columns="block", values="qlike_rel_ewma")
        print(f"\n=== {exp} h={h}: QLIKE relative to EWMA ===")
        print(piv.reindex(order(piv.index)).round(3).to_string())
