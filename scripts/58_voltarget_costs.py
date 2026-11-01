"""Volatility targeting with a limit on leverage and transaction costs (K16; uses saved forecasts only).

Every in-range portfolio is held with exposure e_t = min(sigma* / sigma_hat_t, L_MAX), sigma* = 10% per year, reset every
h days as in scripts/12_vol_targeting.py. Trading the exposure costs COST per unit of |e_t - e_{t-1}| (the constituents
are traded in proportion to the weights, so the turnover of the portfolio equals that of the exposure).
Metrics (averaged over the portfolios): rmse of the monthly log deviation of realized volatility from the target, share
of capped resets, annual turnover, annual cost, Sharpe ratio before and after costs.
Tests: H1 and H2 families on the monthly squared log deviation with the cap (as scripts/17_economic_tests.py); turnover,
costs and Sharpe ratios are descriptive (the mean return depends mainly on the level of the exposure).
Writes results*/main/tables/voltarget_costs.csv and voltarget_costs_tests.csv.

Usage: [PVOL_UNIVERSE=stocks] python scripts/58_voltarget_costs.py
"""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pvol.config import HORIZONS, RESULTS  # noqa: E402
from pvol.data import load_panel  # noqa: E402
from pvol.evaluate import CLASSICAL, NEURAL, dm_test, group_index, load_run  # noqa: E402

_spec = importlib.util.spec_from_file_location("hyp", Path(__file__).with_name("15_hypotheses.py"))
hyp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hyp)

TARGET = 0.10
L_MAX = 2.0
COST = 0.001  # 10 basis points per unit of exposure traded
MONTH = 21


def managed(r, f, origins, h):
    """r: (T_all, P) daily returns; f: (T, P) annualized variance forecasts at `origins`. Returns the daily managed
    returns before costs, the daily costs (charged on the first day after a reset) and the share of capped resets."""
    reb = np.arange(0, len(origins), h)
    e = np.minimum(TARGET / np.sqrt(f[reb]), L_MAX)  # (resets, P)
    trade = np.abs(np.diff(e, axis=0, prepend=e[:1]))  # no cost for the first position
    ret, cost = [], []
    for k, t in enumerate(origins[reb]):
        block = r[t + 1 : t + 1 + h] * e[k]
        c = np.zeros_like(block)
        c[0] = COST * trade[k]
        ret.append(block)
        cost.append(c)
    return np.concatenate(ret), np.concatenate(cost), (e >= L_MAX).mean(0)


def monthly(x, fn):
    n = (len(x) // MONTH) * MONTH
    return fn(x[:n].reshape(-1, MONTH, x.shape[1]))


if __name__ == "__main__":
    R = load_panel()["R"].astype(np.float64)
    rows, tests = [], []
    for h in HORIZONS:
        run = load_run("main", h)
        cols = hyp.cols_of(group_index(run["ports"]), "in-range")
        r = np.column_stack([R[:, q["ids"]] @ np.asarray(q["weights"]) for q in (run["ports"][j] for j in cols)])
        sq = {}
        for m in [x for x in CLASSICAL + NEURAL if x in run["F"]]:
            f = run["F"][m][:, cols] * 252
            if not np.isfinite(f).all():
                continue
            g, c, capped = managed(r, f, run["origins"], h)
            years = len(g) / 252
            mv = monthly(g, lambda x: np.sqrt((x**2).mean(1) * 252))
            sq[m] = (np.log(mv / TARGET) ** 2).mean(1)  # monthly, averaged over the portfolios
            sharpe = lambda x: x.mean(0) / x.std(0) * np.sqrt(252)
            rows.append(dict(h=h, model=m, rmse_log_monthly=np.sqrt(sq[m].mean()), capped_share=capped.mean(),
                             turnover_per_year=(c.sum(0) / COST / years).mean(), cost_per_year=(c.sum(0) / years).mean(),
                             sharpe_gross=sharpe(g).mean(), sharpe_net=sharpe(g - c).mean()))
        bench = min((m for m in CLASSICAL if m in sq), key=lambda m: sq[m].mean())
        for fam, pairs in (("H2", [(a, bench) for a in hyp.SYMMETRIC]),
                           ("H1", [(a, hyp.COUNTERPART[a]) for a in hyp.SYMMETRIC])):
            fr = []
            for a, b in pairs:
                d = sq[a] - sq[b]
                stat, p = dm_test(d, 1)
                fr.append(dict(hypothesis=f"K16-{fam}", h=h, model=a, benchmark=b, loss_a=np.sqrt(sq[a].mean()),
                               loss_b=np.sqrt(sq[b].mean()), mean_diff=d.mean(), dm=stat, p=p))
            tests += hyp.family(fr)
    out = RESULTS / "main" / "tables"
    pd.DataFrame(rows).to_csv(out / "voltarget_costs.csv", index=False)
    T = pd.DataFrame(tests)
    T.to_csv(out / "voltarget_costs_tests.csv", index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(T[["hypothesis", "h", "model", "benchmark", "loss_a", "loss_b", "p_holm", "verdict"]]
          .round(5).to_string(index=False))
