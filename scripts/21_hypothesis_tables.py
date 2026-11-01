"""Tables of the supplementary material written to paper/tables/, built from results*/*/tables/*.csv only.

heldout.tex     QLIKE relative to EWMA for portfolios of assets excluded from training (G1c)
structured.tex  weights of the structured allocations
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


OUT = Path(__file__).resolve().parents[1] / "paper" / "tables"
ROOT_DIR = Path(__file__).resolve().parents[1]
BASE = ROOT_DIR / "results"
LAB = {"SetTransformer": "Set Transformer", "DeepSets": "Deep Sets", "EQCov": "EQ-Cov", "PENet": "PE-Net",
       "SetGRU": "Set-GRU", "HARX": "HAR-X", "GJR": "GJR-GARCH", "EWMANLS": "EWMA-NLS", "DCCNL": "DCC-NL"}
NETS = ["PortMLP", "MLP", "MLP-sorted", "GRU", "DeepSets", "PENet", "SetTransformer", "SetGRU", "EQCov"]
HS = (1, 5, 22)


def lab(m):
    return LAB.get(m, m)


def _losses(root, h, exp="main"):
    f = root / exp / "tables" / f"losses_h{h}.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


def heldout_table():
    """Unseen assets (G1c): QLIKE relative to EWMA for portfolios composed only of assets excluded from training."""
    from pvol.evaluate import CLASSICAL

    cols = [("ETF", BASE, h) for h in HS] + [("Stocks", ROOT_DIR / "results_stocks", 5)]
    vals = {}
    for u, root, h in cols:
        d = _losses(root, h, "heldout")
        if d.empty:
            continue
        d = d[d.block == "unseen-assets"].set_index("model").qlike_rel_ewma
        vals[(u, h)] = d
    models = [m for m in CLASSICAL + NETS if any(m in v.index for v in vals.values())]
    held = [len(json.loads((root / "heldout" / "heldout_assets.json").read_text())) for root in (BASE, ROOT_DIR / "results_stocks")]
    L = ["\\begin{table}[H]\\centering\\footnotesize\\setlength{\\tabcolsep}{5pt}",
         "\\caption{Unseen assets (G1c): QLIKE relative to EWMA for portfolios of 5 and 10 assets composed only of the "
         f"assets excluded from training ({held[0]} funds, {held[1]} stocks); the stock universe was run at $h=5$ only. "
         "Bold: best in the column.}\\label{tab:heldout}",
         "\\begin{tabular}{lrrrr}\\toprule",
         " & \\multicolumn{3}{c}{ETF} & Stocks\\\\ \\cmidrule(lr){2-4}\\cmidrule(lr){5-5}",
         "Model & $h=1$ & $h=5$ & $h=22$ & $h=5$\\\\ \\midrule"]
    for m in models:
        if m == "PortMLP":
            L.append("\\midrule")
        cells = []
        for u, _, h in cols:
            v = vals.get((u, h))
            if v is None or m not in v.index:
                cells.append("--")
                continue
            c = f"{v[m]:.3f}"
            cells.append(f"\\textbf{{{c}}}" if np.isclose(v[m], v.min()) else c)
        L.append(f"{lab(m)} & " + " & ".join(cells) + "\\\\")
    L += ["\\bottomrule\\end{tabular}", "\\end{table}"]
    (OUT / "heldout.tex").write_text("\n".join(L), encoding="utf-8")


def structured_table():
    from pvol.portfolios import STRUCTURED

    names = {"60/40": "60/40", "AllWeather": "All-weather type", "GlobalMulti": "Global multi-asset"}
    L = ["\\begin{table}[H]\\centering\\footnotesize\\setlength{\\tabcolsep}{5pt}",
         "\\caption{Structured allocations of the ETF universe (constant weights, rebalanced daily). The long--short "
         "portfolios have ten constituents with weights $z_k/\\sum_l|z_l|$, $z_k$ independent standard normal, i.e.\\ unit "
         "gross exposure and a random net exposure with mean zero.}\\label{tab:structured}",
         "\\begin{tabular}{lp{11cm}}\\toprule", "Allocation & Weights\\\\ \\midrule"]
    for k, spec in STRUCTURED.items():
        L.append(f"{names.get(k, k)} & " + ", ".join(f"{t} {100 * w:g}\\%" for t, w in spec.items()) + "\\\\")
    L += ["\\bottomrule\\end{tabular}", "\\end{table}"]
    (OUT / "structured.tex").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    heldout_table()
    structured_table()
    print("tables written to", OUT)
