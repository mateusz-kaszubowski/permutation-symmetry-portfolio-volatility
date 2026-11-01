"""Model Confidence Set with the T_max statistic for the in-range portfolios (the main tables use the range statistic).

Same models, daily losses, bootstrap, block length, replications and seed as scripts/06_evaluate.py; only the
statistic differs. The range statistic is sensitive to models with very noisy losses: LW stays in the 90% set although
it is significantly worse than HEAVY in a direct Diebold-Mariano test, which is reported as well.
Writes results*/main/tables/mcs_tmax.csv and dm_lw_heavy.csv for the current universe.

Usage: [PVOL_UNIVERSE=stocks] python scripts/49_mcs_tmax.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pvol.config import HORIZONS, RESULTS  # noqa: E402
from pvol.evaluate import daily_loss, dm_test, group_index, load_run, mcs, qlike  # noqa: E402

T = RESULTS / "main" / "tables"
IN_RANGE = ["N5", "N10", "N20", "N10-conc", "N10-EW"]

rows, dm_rows = [], []
for h in HORIZONS:
    run = load_run("main", h)
    gidx = group_index(run["ports"])
    cols = np.concatenate([gidx[g] for g in IN_RANGE if g in gidx])
    tested = pd.read_csv(T / f"mcs_h{h}.csv").query("block == 'in-range'").model
    daily = {m: daily_loss(qlike(run["y"], run["F"][m]), cols) for m in tested}
    pv = mcs(pd.DataFrame(daily), size=0.10, method="max", h=h)
    rows += [dict(h=h, block="in-range", model=m, p_value=v) for m, v in pv.items()]
    d = daily["LW"] - daily["HEAVY"]
    stat, p = dm_test(d, h)
    dm_rows.append(dict(h=h, block="in-range", model="LW", benchmark="HEAVY", mean_diff=d.mean(), dm=stat, p=p))
    print(f"h={h}: in the 90% set: {' '.join(m for m, v in pv.items() if v > 0.10)}; LW - HEAVY: DM p = {p:.4f}")
pd.DataFrame(rows).to_csv(T / "mcs_tmax.csv", index=False)
pd.DataFrame(dm_rows).to_csv(T / "dm_lw_heavy.csv", index=False)
