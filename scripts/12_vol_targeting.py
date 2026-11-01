"""Economic evaluation 1: volatility targeting with each model's forecasts (uses saved forecasts only).

Every in-range evaluation portfolio is levered by sigma*/sigma_hat at each rebalancing origin (every h days, using the
h-day forecast) and held for h days. A good risk model delivers a managed portfolio whose realised volatility is
close to the target in every month. Metric: root mean squared log deviation of the monthly (21-day) realised volatility
from the target (ideal 0), averaged over the portfolios. The target cancels out, so it is set to 1. The tests are in
scripts/17_economic_tests.py.

Usage: [PVOL_UNIVERSE=stocks] python scripts/12_vol_targeting.py [h ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pvol.config import RESULTS  # noqa: E402
from pvol.data import load_panel  # noqa: E402
from pvol.evaluate import CLASSICAL, NEURAL, group_index, load_run  # noqa: E402

IN_RANGE = ["N5", "N10", "N20", "N10-conc", "N10-EW"]


def managed_returns(r, f, origins, h):
    """r: (T_all, P) daily portfolio returns; f: (T, P) variance forecasts at `origins`; rebalance every h origins."""
    reb = np.arange(0, len(origins), h)
    out = []
    for k in reb:
        t = origins[k]
        expo = 1.0 / np.sqrt(f[k])  # (P,)
        out.append(r[t + 1 : t + 1 + h] * expo)
    return np.concatenate(out)  # (~T, P) daily managed returns, in chronological order


def rmse_log_monthly(m, month=21):
    """Root mean squared log deviation of the monthly realised volatility from the target 1, per portfolio."""
    n = (len(m) // month) * month
    mv = np.sqrt((m[:n].reshape(-1, month, m.shape[1]) ** 2).mean(1) * 252)  # (months, P)
    return np.sqrt(np.mean(np.log(mv) ** 2, axis=0))


if __name__ == "__main__":
    horizons = [int(x) for x in sys.argv[1:]] or [1, 5, 22]
    R = load_panel()["R"].astype(np.float64)
    rows = []
    for h in horizons:
        run = load_run("main", h)
        ports, origins = run["ports"], run["origins"]
        gidx = group_index(ports)
        cols = np.concatenate([gidx[g] for g in IN_RANGE if g in gidx])
        r = np.column_stack([R[:, ports[j]["ids"]] @ np.asarray(ports[j]["weights"]) for j in cols])
        # annualise the target: daily forecasts are average daily variances -> vol target 1 in annual units
        for m in [x for x in CLASSICAL + NEURAL if x in run["F"]]:
            f = run["F"][m][:, cols] * 252
            if not np.isfinite(f).all():
                continue
            rows.append(dict(h=h, block="in-range", model=m,
                             rmse_log_monthly=rmse_log_monthly(managed_returns(r, f, origins, h)).mean()))
        print(f"h={h} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "main" / "tables" / "vol_targeting.csv", index=False)
    print(df.pivot(index="model", columns="h", values="rmse_log_monthly").round(4))
