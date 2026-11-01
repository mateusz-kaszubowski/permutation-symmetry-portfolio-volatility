"""LaTeX tables of the robustness analyses (scripts/28_robustness_analyses.py), both universes.

- paper/tables/alt_tests.tex: verdict counts of H1 and H2 under the Diebold-Mariano test on the full test sample, on
  the test origins outside 2020 and outside 2026, with the Holm correction over all comparisons
  (scripts/32_precision_multiplicity.py), under the conditional Giacomini-White test and under the alternative training
  protocols (protocol B, refit); share of seed pairs with a lower loss of the symmetric network.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402


OUT = Path(__file__).resolve().parents[1] / "paper" / "tables"
ROOT_DIR = Path(__file__).resolve().parents[1]
BASE = ROOT_DIR / "results"
UNIVERSES = [("ETF", BASE), ("Stocks", ROOT_DIR / "results_stocks")]
H = (1, 5, 22)


def tab(root, name):
    return pd.read_csv(root / "main" / "tables" / f"{name}.csv")


def count(d):
    good, bad = (d.verdict == "more accurate").sum(), (d.verdict == "less accurate").sum()
    return f"{good}/{len(d)}" + (f" ($-${bad})" if bad else "") if len(d) else "--"


def opt_tab(root, name):
    f = root / "main" / "tables" / f"{name}.csv"
    return pd.read_csv(f) if f.exists() else None


def global_holm(universe):
    """H1/H2 rows of the main families with the verdict re-decided by the Holm correction over all comparisons."""
    g = opt_tab(BASE, "global_multiplicity")
    if g is None:
        return None
    g = g[(g.universe == universe) & g.family.str.match(r"H[12]: symmetric vs (counterpart|best classical)")].copy()
    g["hypothesis"] = g.family.str[:2]
    g["h"] = g.family.str.extract(r"h=(\d+)").astype(int)
    g["verdict"] = [v if p < 0.05 else "indistinguishable" for v, p in zip(g.verdict, g.p_global_holm)]
    return g


def tests():
    g = opt_tab(BASE, "global_multiplicity")
    L = ["\\begin{table}[H]\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}",
         "\\caption{H1 and H2 under alternative tests, samples and protocols: comparisons decided in favor of the "
         "symmetric network out of five; ($-k$): $k$ against it; `--': not run. Excl.\\ 2020 (2026): origins outside "
         f"that year; global Holm: over {0 if g is None else len(g)} comparisons; GW: conditional Giacomini--White test "
         "with the differential lagged by $h$ (constant and lagged "
         "differential); protocol B, refit: see the text of this section. Seed pairs: share of seed pairs in which the "
         "symmetric network has the lower loss (100 pairs for stocks at $h=1$, otherwise 9).}"
         "\\label{tab:alt_tests}",
         "\\begin{tabular}{ll" + "c" * 2 * len(UNIVERSES) + "}\\toprule",
         " & & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{u}}}" for u, _ in UNIVERSES) + "\\\\",
         "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
         "Test & $h$ & " + " & ".join("H1 & H2" for _ in UNIVERSES) + "\\\\ \\midrule"]
    in_range = lambda d: None if d is None else d[d.block == "in-range"]
    excl = lambda yr: lambda d: None if d is None else d[d.block == f"in-range, excl. {yr}"]
    specs = [("DM", lambda r, u: in_range(opt_tab(r, "hypotheses"))),
             ("DM, excl.\\ 2020", lambda r, u: excl(2020)(opt_tab(r, "hypotheses_excl_year"))),
             ("DM, excl.\\ 2026", lambda r, u: excl(2026)(opt_tab(r, "hypotheses_excl_year"))),
             ("DM, global Holm", lambda r, u: global_holm(u)),
             ("GW", lambda r, u: opt_tab(r, "gw")),
             ("Protocol B", lambda r, u: in_range(opt_tab(r, "hypotheses_P2"))),
             ("Refit", lambda r, u: in_range(opt_tab(r, "hypotheses_RF")))]
    for label, get in specs:
        D = {u: get(r, u) for u, r in UNIVERSES}
        hs = [h for h in H if any(d is not None and (d.h == h).any() for d in D.values())]
        for i, h in enumerate(hs):
            cells = []
            for u, _ in UNIVERSES:
                d = D[u]
                d = None if d is None else d[d.h == h]
                cells += [count(d[d.hypothesis == f]) if d is not None and len(d) else "--" for f in ("H1", "H2")]
            L.append(f"{label if i == 0 else ''} & {h} & " + " & ".join(cells) + "\\\\")
        if hs:
            L.append("\\addlinespace")
    seeds = {u: tab(r, "seed_signs") for u, r in UNIVERSES}
    for i, h in enumerate(H):
        cells = []
        for u, _ in UNIVERSES:
            s = seeds[u]
            s = s[s.h == h].share_lower
            cells += [f"{s.min():.2f}--{s.max():.2f}" if s.max() - s.min() > 5e-3 else f"{s.min():.2f}", ""]
        L.append(f"{'Seed pairs' if i == 0 else ''} & {h} & " + " & ".join(cells) + "\\\\")
    L += ["\\bottomrule\\end{tabular}", "\\end{table}"]
    (OUT / "alt_tests.tex").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    tests()
    print((OUT / "alt_tests.tex").read_text(encoding="utf-8"))
