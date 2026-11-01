"""Tables of the supplementary material: full results of the robustness checks.

Reads the CSV files of both universes (no new forecasts, except that K9 repeats the minimum-variance tests of
scripts/17_economic_tests.py on the weights of scripts/44_minvar_check.py) and writes paper/supplement/tables/*.tex.
Differences are in percentage points of the in-range EWMA loss; 95% confidence intervals from the HAC standard error of
the Diebold-Mariano test (SE = mean / DM statistic, Student t with T-1 degrees of freedom), as in
scripts/32_precision_multiplicity.py. Parameter counts are taken from the models built with the settings in use, in a
subprocess per universe (the feature dimensions depend on the universe).

Usage: python scripts/46_supplement_tables.py
"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

SIZE_SETTINGS = {"MLP-small": ("MLP", {"hidden": [16, 16]}), "MLP": ("MLP", {}), "DeepSets": ("DeepSets", {}),
               "DeepSets-L": ("DeepSets", {"d": 400, "phi": [400], "rho": [400]})}

if len(sys.argv) > 1 and sys.argv[1] == "--params":  # child process: parameter counts for the current universe
    from pvol.experiment import load_tuned_hp
    from pvol.train import build_model
    hp = load_tuned_hp()
    print(json.dumps({tag: sum(p.numel() for p in build_model(m, {**hp[m], **extra}).parameters())
                      for tag, (m, extra) in SIZE_SETTINGS.items()}))
    sys.exit()

from pvol.evaluate import dm_test  # noqa: E402

_spec = importlib.util.spec_from_file_location("hyp", ROOT / "scripts" / "15_hypotheses.py")
hyp = importlib.util.module_from_spec(_spec)
sys.argv = sys.argv[:1]
_spec.loader.exec_module(hyp)

ROOT_DIR = Path(__file__).resolve().parents[1]
BASE = ROOT_DIR / "results"
UNIVERSES = [("ETF", BASE, "etf"), ("Stocks", ROOT_DIR / "results_stocks", "stocks")]
OUT = ROOT / "paper" / "supplement" / "tables"
HS = (1, 5, 22)
NAME = {"DeepSets": "Deep Sets", "DeepSets-L": "Deep Sets, $d=400$", "MLP-small": "MLP, $(16,16)$", "PENet": "PE-Net",
        "SetTransformer": "Set Transformer", "SetGRU": "Set-GRU", "EQCov": "EQ-Cov", "PortMLP": "PortMLP",
        "EWMANLS": "EWMA-NLS", "LW": "LW", "GJR": "GJR-GARCH", "Linear-noasset": "without element features",
        "Linear-noeuler": "without Euler contributions", "Linear-nopairs": "without correlation features",
        "Linear-noport": "without portfolio-level features", "Linear-portonly": "portfolio-level features only"}
nm = lambda m: NAME.get(m, m)


def f(x, d=2):
    return "--" if pd.isna(x) else f"{x:.{d}f}".replace("-", "$-$")


def pval(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def write(name, caption, label, colspec, header, body):
    OUT.mkdir(parents=True, exist_ok=True)
    tex = ["\\begin{table}[H]\\centering\\small", f"\\caption{{{caption}}}\\label{{{label}}}",
           f"\\begin{{tabular}}{{{colspec}}}\\toprule", *header, "\\midrule", *body, "\\bottomrule\\end{tabular}",
           "\\end{table}"]
    (OUT / f"{name}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def tests(root):
    """Tests of the robustness checks with effects and confidence intervals in EWMA points."""
    d = pd.read_csv(root / "main" / "tables" / "hypotheses_checks.csv")
    lv = pd.read_csv(root / "main" / "tables" / "checks_levels.csv").set_index(["h", "model"]).qlike_rel_ewma
    e = np.array([r.loss_a / lv[(r.h, r.model)] for r in d.itertuples()])
    T = d.h.map({h: len(np.load(root / "main" / f"h{h}" / "origins.npy")) for h in HS})
    se = np.abs(d.mean_diff / d.dm) / e * 100
    d["eff"] = 100 * d.mean_diff / e
    tq = stats.t.ppf(0.975, T - 1)
    d["lo"], d["hi"] = d.eff - tq * se, d.eff + tq * se
    return d, lv


def cells(r):
    return [f(r.eff), f"[{f(r.lo)}, {f(r.hi)}]", pval(r.p_holm)]


def two_universe_header(first, sub):
    n = len(sub)
    return [f"{first} & \\multicolumn{{{n}}}{{c}}{{ETF}} & \\multicolumn{{{n}}}{{c}}{{Stocks}}\\\\",
            f"\\cmidrule(lr){{2-{n + 1}}}\\cmidrule(lr){{{n + 2}-{2 * n + 1}}}",
            " & " + " & ".join(sub * 2) + "\\\\"]


def table_ties(T):
    sub = [f"$h={h}$" for h in HS]
    body = []
    for m in ("MLP", "MLP-sorted", "MLP-tsort"):
        body.append(nm(m) + " & " + " & ".join(f(T[u][1][(h, m)], 3) for u in T for h in HS) + "\\\\")
    body.append("\\addlinespace")
    for hname, b in (("K5a", "MLP-sorted"), ("K5b", "MLP")):
        rows = [T[u][0][T[u][0].hypothesis == hname].set_index("h") for u in T]
        body.append(f"MLP-tsort $-$ {b} & " + " & ".join(f(r.loc[h, "eff"]) for r in rows for h in HS) + "\\\\")
        body.append("\\quad Holm-adjusted $p$ & " + " & ".join(pval(r.loc[h, "p_holm"]) for r in rows for h in HS)
                    + "\\\\")
    body.append("\\addlinespace")
    rng = {u: {m: pd.read_csv(root / "main" / "tables" / fn, index_col=0).loc["mean"]
               for m, fn in (("MLP-sorted", "sorted_ties.csv"), ("MLP-tsort", "sorted_ties_MLP-tsort.csv"))}
           for u, root, _ in UNIVERSES}
    for m in ("MLP-sorted", "MLP-tsort"):
        body.append(f"{m}: range, median / p99 & " + " & ".join(
            f"\\multicolumn{{3}}{{c}}{{{rng[u][m]['median']:.3g} / {rng[u][m]['p99']:.3g}}}" for u in T) + "\\\\")
    write("supp_ties", "K5, ties in the sort by weight. Top: in-range QLIKE relative to EWMA (seed ensembles). Middle: "
          "differences in pp and Holm-adjusted $p$ (Holm over the three horizons). Bottom: range "
          "of the log-variance forecast over five orderings of 100 equally weighted ten-asset portfolios ($h=5$, "
          "median and 99th percentile over portfolios and origins, mean over seeds).", "tab:supp_ties",
          "l" + "r" * 6, two_universe_header("", sub), body)


def table_tests(T, hname, models, name, caption, label, extra=None):
    sub = ["$\\Delta$", "95\\% CI", "$p_{\\mathrm{Holm}}$"]
    body = []
    for h in HS:
        body.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{$h={h}$}}}}\\\\")
        for a, b in models:
            row = [nm(a) + (f" vs {nm(b)}" if extra is None else extra(a, b))]
            for u in T:
                d = T[u][0]
                r = d[(d.hypothesis == hname) & (d.h == h) & (d.model == a) & (d.benchmark == b)].iloc[0]
                row += cells(r)
            body.append(" & ".join(row) + "\\\\")
    write(name, caption, label, "l" + "rrr" * 2, two_universe_header("", sub), body)


def table_size_params():
    P = {}
    for u, _, env in UNIVERSES:
        out = subprocess.run([sys.executable, __file__, "--params"], capture_output=True, text=True, check=True,
                             env={**os.environ, "PVOL_UNIVERSE": env}).stdout
        P[u] = json.loads(out.strip().splitlines()[-1])
    body = [nm(m) + " & " + " & ".join(f"{P[u][m]:,}" for u in P) + "\\\\" for m in SIZE_SETTINGS]
    write("supp_size_params", "K7, number of parameters of the capacity-matched networks.", "tab:supp_size_params",
          "lrr", ["Network & ETF & Stocks\\\\"], body)


def minvar_tests(df, cov_models):
    by = df.groupby(["model", "origin"]).realised_var.mean().unstack(0)
    bench = min((m for m in cov_models if m in by), key=lambda m: by[m].mean())
    rows = []
    for fam, pairs in (("H2", [(a, bench) for a in hyp.SYMMETRIC]),
                       ("H1", [(a, hyp.COUNTERPART[a]) for a in hyp.SYMMETRIC])):
        fr = []
        for a, b in pairs:
            d = (by[a] - by[b]).to_numpy()
            stat, p = dm_test(d, 5)
            fr.append(dict(hypothesis=fam, model=a, benchmark=b, mean_diff=d.mean(), p=p, dm=stat))
        rows += hyp.family(fr)
    return pd.DataFrame(rows).set_index(["hypothesis", "model"])


def table_minvar_check():
    sub = ["Adam", "best", "change", "rank"]
    S, Tst = {}, {}
    for u, root, _ in UNIVERSES:
        t = root / "main" / "tables"
        S[u] = pd.read_csv(t / "minvar_check_summary.csv", index_col=0)
        orig = pd.read_csv(t / "minvar_h5.csv")
        chk = pd.read_csv(t / "minvar_check.csv")
        m = orig.merge(chk, on=["year", "origin", "subset", "model"], how="left")
        cov = ["EWMA", "DCC", "DCCNL", "LW", "EWMANLS", "PCA-F"]
        Tst[u] = (minvar_tests(orig, cov), minvar_tests(m.assign(realised_var=m.rv_best.fillna(m.realised_var)), cov))
    order = S["Stocks"].sort_values("vol_adam").index
    body = []
    for mdl in order:
        row = [nm(mdl)]
        for u in S:
            r = S[u].loc[mdl]
            row += [f(r.vol_adam, 3), f(r.vol_best, 3), f(r.change_pp, 3),
                    f"{int(r.rank_adam)}$\\to${int(r.rank_best)}"]
        body.append(" & ".join(row) + "\\\\")
    body.append("\\addlinespace")
    for fam, label in (("H1", "H1: decided for the symmetric network"), ("H2", "H2: decided for the symmetric network")):
        row = [label]
        for u in S:
            a, b = (sum(t.loc[fam].verdict == "more accurate") for t in Tst[u])
            row += [f"\\multicolumn{{4}}{{c}}{{{a}/5 (Adam), {b}/5 (best)}}"]
        body.append(" & ".join(row) + "\\\\")
    write("supp_minvar_check", "K9, minimum-variance optimizer ($h=5$, seed 0). Annual realized volatility (\\%) of the "
          "portfolios chosen with the stored weights (200 Adam steps from equal weights) and with the better solution "
          "(exact projected-gradient solution for the covariance models, best of seven runs by the model's own "
          "forecast for the networks), its change in percentage points, and the rank among the 13 models. Bottom: "
          "verdicts of the minimum-variance tests of H1 and H2 with either set of weights (DCC and DCC-NL keep "
          "their stored weights).", "tab:supp_minvar_check", "l" + "rrrr" * 2, two_universe_header("Model", sub), body)


def table_yearly():
    years = None
    body = []
    for u, root, _ in UNIVERSES:
        d = pd.read_csv(root / "main" / "tables" / "yearly_pairs.csv")
        years = sorted(d.year.unique())
        body.append(f"\\multicolumn{{{len(years) + 1}}}{{l}}{{\\emph{{{u}}}}}\\\\")
        for (a, b) in d[["model", "benchmark"]].drop_duplicates().itertuples(index=False):
            for h in HS:
                s = d[(d.model == a) & (d.benchmark == b) & (d.h == h)].set_index("year")
                if s.empty:
                    continue
                vals = [f(s.loc[y, "mean_diff_pp"], 1) + ("$^*$" if s.loc[y, "p"] < 0.05 else "") for y in years]
                body.append(f"{nm(a)} $-$ {nm(b)}, $h={h}$ & " + " & ".join(vals) + f" & {f(s.mean_diff_pp.median(), 1)}\\\\")
        body.append("\\addlinespace")
    body.pop()
    write("supp_yearly", "K8, results by test year: mean in-range QLIKE difference in points of that year's EWMA "
          "loss (negative: the first model is more accurate) and its median over the years; $^*$: unadjusted "
          "Diebold--Mariano $p<0.05$ within the year (descriptive, no correction). One month ahead Linear is also "
          "compared with the best classical model of that horizon (HAR for the funds, GJR-GARCH for the stocks). 2026: "
          "January to the end of the sample.", "tab:supp_yearly",
          "l" + "r" * (len(years) + 1), ["Comparison & " + " & ".join(str(y) for y in years) + " & median\\\\"], body)


# ---------------------------------------------------------------- other tables: results reported in the text only

def load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), ROOT / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, sys.argv[:1]
    spec.loader.exec_module(mod)
    sys.argv = argv
    return mod


def adapt(name, label=None):
    """Adapt a table written by a paper script to the supplement: [H] placement, sections of the main text by number."""
    fl = OUT / f"{name}.tex"
    t = fl.read_text(encoding="utf-8")
    for a, b in (("\\begin{table}[t]", "\\begin{table}[H]"),
                 ("Section~\\ref{sec:hypotheses}", "Section~III-C of the main text"),
                 ("Section~\\ref{sec:res:robust}", "Section~VI-E of the main text")):
        t = t.replace(a, b)
    if label:
        t = t.replace("\\label{tab:main_all}", f"\\label{{{label}}}").replace("\\label{tab:var}", f"\\label{{{label}}}")
    fl.write_text(t, encoding="utf-8")


def ewma_loss(root, h):
    d = pd.read_csv(root / "main" / "tables" / f"losses_h{h}.csv")
    return d[(d.block == "in-range") & (d.model == "EWMA")].qlike.iloc[0]


def with_effects(d, root):
    """Effect and 95% CI in EWMA points for rows of a hypothesis table (columns h, mean_diff, dm)."""
    d = d.copy()
    e = d.h.map({h: ewma_loss(root, h) for h in HS})
    T = d.h.map({h: len(np.load(root / "main" / f"h{h}" / "origins.npy")) for h in HS})
    se = np.abs(d.mean_diff / d.dm) / e * 100
    d["eff"] = 100 * d.mean_diff / e
    tq = stats.t.ppf(0.975, T - 1)
    d["lo"], d["hi"] = d.eff - tq * se, d.eff + tq * se
    return d


def mname(m):
    for suf in ("-wideh", "-wide", "-RAW", "-P50"):
        if m.endswith(suf):
            return nm(m[: -len(suf)])
    return nm(m)


def search_rows(H, fams, h, title_prefix=""):
    """Rows of the K3/K14 tables: one block per family, ETF and stocks side by side."""
    body = []
    for fam, title in fams:
        body.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{title_prefix}{title}}}}}\\\\")
        e = H["ETF"]
        for r in e[(e.hypothesis == fam) & (e.h == h)].itertuples():
            if fam.startswith("K3d"):
                lbl = mname(r.model)
            elif fam in ("K3c", "K14c"):
                lbl = f"{mname(r.model)} vs {r.benchmark}"
            else:
                lbl = f"{mname(r.model)} vs {mname(r.benchmark)}"
            row = []
            for u in H:
                x = H[u]
                q = x[(x.hypothesis == fam) & (x.h == h) & (x.model == r.model)].iloc[0]
                c = cells(q)
                if fam in ("K3c", "K14c") and q.benchmark != r.benchmark:
                    c[0] += f" ({q.benchmark})"
                row += c
            body.append(lbl + " & " + " & ".join(row) + "\\\\")
    return body


def table_search():
    H = {u: with_effects(pd.read_csv(root / "main" / "tables" / "hypotheses_search.csv"), root)
         for u, root, _ in UNIVERSES}
    sub = ["$\\Delta$", "95\\% CI", "$p_{\\mathrm{Holm}}$"]
    fams = [("K3a", "K3, H1: symmetric vs counterpart"), ("K3b", "K3, C1: network vs Linear"),
            ("K3c", "K3, H2: symmetric vs best classical"),
            ("K3d (aux: broader search vs main protocol)", "K3: selected settings vs main design")]
    for h in HS:
        body = search_rows(H, fams, h)
        write(f"supp_wide_h{h}", f"K3, broader hyper-parameter search, $h={h}$: settings of all learned models selected by "
              "random search (21 configurations per model, validation over 2017--2019). Difference of the first "
              "model from the second ($\\Delta<0$: the first is more accurate) in pp, 95\\% "
              "confidence interval and Holm-adjusted $p$ (Holm within each family). H2: against the most accurate "
              "classical model of the ETF universe (in brackets when it differs for the stocks). Last rows: each "
              "model with the selected settings against the same model with the settings of the main design.", f"tab:supp_wide_h{h}",
              "l" + "rrr" * 2, two_universe_header("", sub), body)
    fams = [("K14a", "H1: symmetric vs counterpart"), ("K14b", "C1: network vs Linear"),
            ("K14c", "H2: symmetric vs best classical")]
    body = []
    for h in (1, 22):
        if body:
            body.append("\\addlinespace")
        body += search_rows(H, fams, h, f"$h={h}$, ")
    write("supp_wideh", "K14, broader search at the horizon of evaluation: the 21 configurations of K3 compared by the "
          "validation loss at $h=1$ or $h=22$ (2017--2019), main design repeated with the selected settings. Columns "
          "as in Table~\\ref{tab:supp_wide_h1}; H2 against the most accurate classical model of the ETF universe (in "
          "brackets when it differs for the stocks).", "tab:supp_wideh", "l" + "rrr" * 2, two_universe_header("", sub),
          body)
    chk = pd.read_csv(ROOT / "results" / "tuning" / "tuning_check.csv")
    cols = [("etf", 1), ("etf", 22), ("stocks", 5), ("stocks", 1)]
    body = []
    for m in chk.model.unique():
        vals = [chk[(chk.universe == u) & (chk.h == h) & (chk.model == m)].iloc[0] for u, h in cols]
        body.append(f"{nm(m)} & " + " & ".join("0" if r.same else f(r.gap_pct) for r in vals) + "\\\\")
    write("supp_tuning_check", "Grid of the main design (four configurations per model) repeated at another horizon or "
          "on the stocks: validation loss of the settings in use (selected on the ETF universe at $h=5$) above that of "
          "the best configuration of the repeated search, in percent (0: the same configuration is selected).",
          "tab:supp_tuning_check", "lrrrr",
          ["& \\multicolumn{2}{c}{ETF} & \\multicolumn{2}{c}{Stocks}\\\\",
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}", " & $h=1$ & $h=22$ & $h=5$ & $h=1$\\\\"], body)
    lv = {u: pd.read_csv(root / "main" / "tables" / "search_levels.csv").set_index(["h", "model"]).qlike_rel_ewma
          for u, root, _ in UNIVERSES}
    models = ["Linear", "MLP", "GRU", "DeepSets", "PENet", "SetTransformer", "SetGRU", "EQCov"]
    body = []
    for m in models:
        for tag, suffix in (("main", ""), ("broader search", "-wide")):
            vals = [f(lv[u].get((h, m + suffix), np.nan), 3) for u in lv for h in HS]
            body.append(f"{nm(m)}, {tag} & " + " & ".join(vals) + "\\\\")
        if m != models[-1]:
            body.append("\\addlinespace")
    write("supp_wide_levels", "K3, broader search: in-range QLIKE relative to EWMA with the settings of the main design and "
          "with those of the broader search.", "tab:supp_wide_levels", "l" + "r" * 6,
          two_universe_header("", [f"$h={h}$" for h in HS]), body)
    body = []
    for fam, title in (("K4a", "K4a: MLP-idsort vs MLP"), ("K4b", "K4b: MLP-sorted vs MLP-idsort")):
        body.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{title}}}}}\\\\")
        for blk in ("main design", "raw inputs", "pool 50"):
            row = []
            for u in H:
                x = H[u]
                row += cells(x[(x.hypothesis == fam) & (x.block == blk)].iloc[0])
            body.append(blk[0].upper() + blk[1:] + " & " + " & ".join(row) + "\\\\")
    write("supp_sortkey", "K4, sort key ($h=5$): MLP-idsort orders the slots by the asset index (invariant, "
          "unrelated to the weights). Difference of the first model from the second in pp, "
          "95\\% confidence interval and Holm-adjusted $p$ (Holm over the three designs).", "tab:supp_sortkey",
          "l" + "rrr" * 2, two_universe_header("Design", sub), body)


ECON_MODELS = ["HIST", "EWMA", "HAR", "HAR-WLS", "HARX", "LogHAR-X", "HEAVY", "GARCH", "GJR", "DCC", "LW", "EWMANLS",
               "DCCNL", "PCA-F", "PortMLP", "Linear", "GBM", "MLP", "MLP-sorted", "GRU", "DeepSets", "PENet",
               "SetTransformer", "SetGRU", "EQCov"]
LAB2 = {"HARX": "HAR-X", "GJR": "GJR-GARCH", "DCCNL": "DCC-NL", "EW": "Equal weights"}


def nm2(m):
    return LAB2.get(m, nm(m))


def table_economic():
    V = {u: pd.read_csv(root / "main" / "tables" / "vol_targeting.csv") for u, root, _ in UNIVERSES}
    body = []
    for m in ECON_MODELS:
        if m in ("PortMLP", "MLP"):
            body.append("\\addlinespace")
        vals = []
        for u in V:
            for h in HS:
                x = V[u][(V[u].model == m) & (V[u].h == h) & (V[u].block == "in-range")]
                vals.append(f(x.rmse_log_monthly.iloc[0], 3) if len(x) else "--")
        body.append(nm2(m) + " & " + " & ".join(vals) + "\\\\")
    write("supp_voltarget", "Volatility targeting, in-range portfolios: root mean squared log deviation of the monthly "
          "realized volatility of the levered portfolio from the target (lower is better), without transaction "
          "costs.", "tab:supp_voltarget", "l" + "r" * 6, two_universe_header("Model", [f"$h={h}$" for h in HS]), body)
    mv = []
    for u, root, _ in UNIVERSES:
        g = pd.read_csv(root / "main" / "tables" / "minvar_h5.csv").groupby("model")
        mv.append((100 * np.sqrt(g.realised_var.mean() * 252), g.hhi.mean()))
    body = [nm2(m) + " & " + " & ".join(f"{f(v[m], 2)} & {f(hh[m], 3)}" for v, hh in mv) + "\\\\"
            for m in mv[1][0].sort_values().index]
    write("supp_minvar", "Minimum-variance allocation ($h=5$, 100 random ten-asset subsets at every fifth origin, "
          "long-only weights from 200 Adam steps): annual realized volatility (\\%) of the chosen portfolios and mean "
          "Herfindahl index of the weights; ordered by the volatility for the stocks.", "tab:supp_minvar", "lrrrr",
          ["Model & \\multicolumn{2}{c}{ETF} & \\multicolumn{2}{c}{Stocks}\\\\",
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}", " & vol. & HHI & vol. & HHI\\\\"], body)
    X = {u: pd.read_csv(root / "main" / "tables" / "economic_tests.csv") for u, root, _ in UNIVERSES}
    body = []
    for fam, title, hs in (("H1-voltarget", "Volatility targeting, H1: symmetric vs counterpart", HS),
                           ("H2-voltarget", "Volatility targeting, H2: symmetric vs best classical model", HS),
                           ("H1-minvar", "Minimum variance, H1: symmetric vs counterpart", (None,)),
                           ("H2-minvar", "Minimum variance, H2: symmetric vs best covariance model", (None,))):
        body.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{title}}}}}\\\\")
        vt = fam.endswith("voltarget")
        for h in hs:
            e = X["ETF"]
            sel = (e.hypothesis == fam) & ((e.h == h) if h else e.h.isna())
            for r in e[sel].itertuples():
                row = []
                for u in X:
                    x = X[u]
                    q = x[(x.hypothesis == fam) & ((x.h == h) if h else x.h.isna()) & (x.model == r.model)].iloc[0]
                    sc, d = (1, 3) if vt else (100, 2)
                    row += [nm2(q.benchmark), f"{f(sc * q.loss_a, d)} / {f(sc * q.loss_b, d)}", pval(q.p_holm)]
                body.append(nm(r.model) + (f", $h={h}$" if h else "") + " & " + " & ".join(row) + "\\\\")
    write("supp_econ_tests", "Tests of the economic evaluation: benchmark, loss of the symmetric network / of the "
          "benchmark, and Holm-adjusted $p$ of the Diebold--Mariano test on monthly (volatility targeting) or "
          "per-origin (minimum variance) differences. Volatility targeting: square root of the mean over months of "
          "the squared log deviation averaged over portfolios, the quantity tested (Table~\\ref{tab:supp_voltarget} "
          "averages the root mean squared deviation of each portfolio instead, hence small differences); minimum "
          "variance: annual realized volatility in \\%.", "tab:supp_econ_tests", "l" + "lrr" * 2,
          two_universe_header("", ["benchmark", "loss", "$p_{\\mathrm{Holm}}$"]), body)
    tt = load_script("10_tables_tex.py")
    tt.OUT = OUT
    tt.var_table_both()
    (OUT / "var_main.tex").replace(OUT / "supp_var.tex")
    adapt("supp_var", "tab:supp_var")
    F = {u: pd.read_csv(root / "main" / "tables" / "var_t_fhs.csv") for u, root, _ in UNIVERSES}
    body = []
    for m in [m for m in ECON_MODELS if m in set(F["ETF"].model)]:
        vals = []
        for u in F:
            for kind in ("t", "FHS"):
                x = F[u][(F[u].model == m) & (F[u].kind == kind) & (F[u].alpha == 0.01)]
                vals += ([f(100 * x.hit_rate.iloc[0], 2), f(100 * x.kupiec_reject_share.iloc[0], 0)] if len(x)
                         else ["--"] * 2)
        body.append(nm2(m) + " & " + " & ".join(vals) + "\\\\")
    write("supp_var_alt", "One-day value-at-risk at the 1\\% level with Student-$t$ quantiles (unit-variance $t_\\nu$, "
          "$\\nu=4+6/\\kappa$ from the excess kurtosis $\\kappa$ of the last 250 standardized returns, bounded to "
          "$[4.1, 30]$) and with filtered historical simulation (FHS) from the same 250 returns: hit rate and share of portfolios for which the unconditional coverage test "
          "of \\citet{kupiec1995} rejects at 5\\% (both in per cent).", "tab:supp_var_alt", "l" + "rrrr" * 2,
          ["Model & \\multicolumn{4}{c}{ETF} & \\multicolumn{4}{c}{Stocks}\\\\",
           "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
           " & " + " & ".join(["hit $t$ & UC $t$ & hit FHS & UC FHS"] * 2) + "\\\\"], body)


def table_costs():
    V = {u: pd.read_csv(root / "main" / "tables" / "voltarget_costs.csv").set_index(["h", "model"])
         for u, root, _ in UNIVERSES}
    body = []
    for m in ECON_MODELS:
        if m in ("PortMLP", "MLP"):
            body.append("\\addlinespace")
        vals = []
        for u in V:
            x = V[u]
            vals += [f(x.loc[(1, m), "rmse_log_monthly"], 3), f(x.loc[(1, m), "turnover_per_year"], 1),
                     f(x.loc[(1, m), "sharpe_net"], 2), f(x.loc[(5, m), "turnover_per_year"], 1),
                     f(x.loc[(22, m), "turnover_per_year"], 1)]
        body.append(nm2(m) + " & " + " & ".join(vals) + "\\\\")
    sub = ["RMSE", "turn.", "SR", "turn.", "turn."]
    write("supp_costs", "K16, volatility targeting with a target of 10\\% per year, leverage limited to two and a cost of "
          "10 basis points per unit of exposure traded, in-range portfolios: root mean squared log deviation of the "
          "monthly realized volatility from the target and Sharpe ratio after costs one day ahead, and annual turnover "
          "of the exposure (the annual cost in \\% is a tenth of it) at $h=1$, 5 and 22.", "tab:supp_costs",
          "l" + "r" * 10,
          ["Model & \\multicolumn{5}{c}{ETF} & \\multicolumn{5}{c}{Stocks}\\\\",
           "\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}",
           " & " + " & ".join(["\\multicolumn{3}{c}{$h=1$} & $h=5$ & $h=22$"] * 2) + "\\\\",
           " & " + " & ".join(sub * 2) + "\\\\"], body)


def table_volcorr():
    V = {u: pd.read_csv(root / "main" / "tables" / "hypotheses_volcorr.csv") for u, root, _ in UNIVERSES}
    body = []
    for h in HS:
        body.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{$h={h}$}}}}\\\\")
        for lab, col in (("VolCorr-asset, relative to EWMA", "rel_b"),):
            r = [V[u][(V[u].h == h) & (V[u].hypothesis == "K15b")].iloc[0] for u in V]
            body.append(lab + " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{f(x[col], 3)}}}" for x in r) + "\\\\")
        for hname, models in (("K15a", hyp.SYMMETRIC), ("K15b", ["Linear"]), ("K15c", ["VolCorr"])):
            for m in models:
                row = [nm(m) if m != "VolCorr" else "VolCorr (portfolios)"]
                for u in V:
                    x = V[u][(V[u].h == h) & (V[u].hypothesis == hname) & (V[u].model == m)].iloc[0]
                    row += [f(100 * (x.rel_a - x.rel_b)), pval(x.p_holm)]
                body.append(" & ".join(row) + "\\\\")
        d = [V[u][(V[u].h == h) & (V[u].hypothesis == "K15 (descriptive)")].iloc[0] for u in V]
        body.append("Best classical model & " + " & ".join(
            f"\\multicolumn{{2}}{{c}}{{{nm2(x.benchmark)} {f(100 * (x.rel_b - x.rel_a))}}}" for x in d) + "\\\\")
    write("supp_volcorr", "K15, per-asset volatility network with the EWMA correlation, trained on single assets "
          "(VolCorr-asset): in-range QLIKE relative to EWMA, and difference of each model from it in points of the EWMA "
          "loss ($\\Delta<0$: the model is more accurate) with Holm-adjusted $p$ (Holm over the five symmetric "
          "networks; Linear and VolCorr trained on portfolios separately). Last row: best classical model and its "
          "difference from VolCorr-asset.", "tab:supp_volcorr", "lrrrr",
          two_universe_header("", ["$\\Delta$", "$p_{\\mathrm{Holm}}$"]), body)


def table_equivalence():
    E = {u: pd.read_csv(root / "main" / "tables" / "equivalence.csv") for u, root, _ in UNIVERSES}
    yn = lambda b: "yes" if b else "no"
    body = []
    for fam, title in (("H1", "H1: symmetric network vs counterpart"), ("C1", "C1: symmetric network vs Linear")):
        for h in HS:
            body.append(f"\\multicolumn{{11}}{{l}}{{\\emph{{{title}, $h={h}$}}}}\\\\")
            for m in hyp.SYMMETRIC:
                row = []
                for u in E:
                    x = E[u]
                    q = x[(x.family == fam) & (x.h == h) & (x.model == m)].iloc[0]
                    row += [f(q.mean_diff_pp), f"[{f(q.lo90)}, {f(q.hi90)}]", yn(q.equivalent_1pp),
                            yn(q.equivalent_2pp), yn(q.equivalent_2pp_bonf)]
                body.append(nm(m) + " & " + " & ".join(row) + "\\\\")
    write("supp_equivalence", "Precision and equivalence: mean QLIKE difference in pp, 90\\% "
          "confidence interval (HAC standard error) and the verdict of the two one-sided tests (TOST, 5\\%) for "
          "equivalence margins of $\\pm1$ and $\\pm2$ points (equivalent when the 90\\% interval lies inside the "
          "margin). The claim that all five networks of a block are equivalent is an intersection-union test and "
          "needs no multiplicity correction; B: the individual claims at the 2-point margin with Bonferroni over the "
          "five networks (98\\% interval inside the margin).",
          "tab:supp_equivalence", "l" + "rrccc" * 2,
          two_universe_header("", ["$\\Delta$", "90\\% CI", "$\\pm1$", "$\\pm2$", "B"]), body)


def hp_text(d):
    parts = []
    for k, v in d.items():
        if isinstance(v, list):
            v = "(" + ", ".join(str(x) for x in v) + ")"
        parts.append(f"{k.replace('_', ' ')} {v}")
    return "; ".join(parts)


def table_cost_hp():
    C = {u: pd.read_csv(root / "main" / "tables" / "compute_cost.csv").set_index("model") for u, root, _ in UNIVERSES}
    body = []
    for m in C["ETF"].index:
        vals = []
        for u in C:
            r = C[u].loc[m]
            vals += [f"{int(r.params):,}", f(r.ms_10, 2), f(r.ms_all, 1), f(r.train_sec, 0)]
        body.append(nm(m) + " & " + " & ".join(vals) + "\\\\")
    write("supp_cost", "Computational cost ($h=5$): number of parameters, forecast time for 1,000 portfolios of ten "
          "assets and of all assets of the universe (ms; `--': slot-based networks take at most 20 assets), and "
          "mean training time per seed and test year (s), on one consumer GPU (NVIDIA RTX 3060 Ti).", "tab:supp_cost", "l" + "rrrr" * 2,
          two_universe_header("Model", ["params", "$N=10$", "all", "train"]), body)
    body = []
    for u, root, _ in UNIVERSES:
        main = json.loads((root / "tuning" / "best_hp.json").read_text())
        wide = json.loads((root / "tuning" / "best_hp_wide.json").read_text())
        body.append(f"\\multicolumn{{3}}{{l}}{{\\emph{{{u}}}}}\\\\")
        for m in list(main) + [k for k in wide if k not in main]:
            a = hp_text(main[m]) if m in main else "--"
            b = hp_text({**wide[m]["hp"], **wide[m]["cfg"]}) if m in wide else "--"
            body.append(f"{nm(m)} & {a} & {b}\\\\")
    write("supp_hp", "Hyper-parameters in use: main design (training settings otherwise the defaults: learning rate "
          "$10^{-4}$, weight decay $10^{-4}$, patience 10) and the broader search K3 (architecture and "
          "training settings). d: width of the element embedding; phi, rho: hidden layers of the element and "
          "read-out maps. GBM has fixed settings (Section~\\ref{s:all}).", "tab:supp_hp", "lp{5.2cm}p{7.6cm}", ["Model & Main design & Broader search\\\\"], body)


def table_dropout():
    rows = []
    for u, root, _ in UNIVERSES:
        t = root / "main" / "tables"
        old = with_effects(pd.read_csv(t / "hypotheses_routes.csv").query("h == 5"), root)
        new = with_effects(pd.read_csv(t / "hypotheses_dropout.csv"), root)
        rows.append((old, new))
    designs = [("raw inputs", "RAW", ("DeepSets", "PENet", "SetTransformer")),
               ("pool 50", "P50", ("DeepSets", "PENet", "SetTransformer", "EQCov")),
               ("pool 500", "P500", ("DeepSets", "PENet", "SetTransformer", "EQCov"))]
    body = []
    for blk, suf, models in designs:
        body.append(f"\\multicolumn{{9}}{{l}}{{\\emph{{{blk[0].upper() + blk[1:]}}}}}\\\\")
        pairs = [(f"{a}-{suf}", f"MLP-sorted-{suf}", nm(a) + " vs MLP-sorted") for a in models]
        pairs.append((f"MLP-sorted-{suf}", f"MLP-{suf}", "MLP-sorted vs MLP"))
        for a, b, lbl in pairs:
            row = [lbl]
            for old, new in rows:
                o = old[(old.model == a) & (old.benchmark == b)].iloc[0]  # tags are unique within the family
                a3 = a.replace(f"-{suf}", f"-{suf}-d3") if a.startswith("MLP-sorted") else a
                b3 = b + "-d3" if b.startswith("MLP-sorted") else b
                n = new[(new.model == a3) & (new.benchmark == b3)].iloc[0]
                row += [f(o.eff), pval(o.p_holm), f(n.eff), pval(n.p_holm)]
            body.append(" & ".join(row) + "\\\\")
    write("supp_dropout", "K10, dropout of MLP-sorted ($h=5$): B2 and the comparison of MLP-sorted with the MLP with "
          "MLP-sorted trained with its selected dropout (0.1, as in the main text) and with the dropout of the MLP "
          "(0.3). Difference of the first model from the second in pp and Holm-adjusted $p$ "
          "(Holm within each design for B2).", "tab:supp_dropout", "l" + "rrrr" * 2,
          two_universe_header("", ["$\\Delta_{0.1}$", "$p$", "$\\Delta_{0.3}$", "$p$"]), body)


def table_raw():
    """Linear with raw inputs (C2), fixed-smoothing tests (K12), SPA (K13), and descriptive tables."""
    RAW_TESTS = {u: pd.read_csv(root / "main" / "tables" / "hypotheses_raw.csv") for u, root, _ in UNIVERSES}
    body = []
    for m in ("DeepSets", "DeepSetsP", "PENet", "SetTransformer", "SetGRU", "MLP", "GRU"):
        row = []
        for u, root, _ in UNIVERSES:
            d = RAW_TESTS[u]
            r = with_effects(d[d.hypothesis.str.startswith("C2") & (d.model == m + "-RAW")], root).iloc[0]
            row += [f(r.eff), f"[{f(r.lo)}, {f(r.hi)}]", pval(r.p_holm) if pd.notna(r.p_holm) else "--"]
        lbl = {"DeepSetsP": "Deep Sets + pairs"}.get(m, nm(m))
        body.append(lbl + " & " + " & ".join(row) + "\\\\")
    write("supp_raw", "C2, raw inputs ($h=5$): difference of each network with raw inputs from Linear with raw inputs, in "
          "pp ($\\Delta<0$: the network is more accurate), with 95\\% confidence interval and "
          "Holm-adjusted $p$ over the five symmetric networks (MLP and GRU descriptive).", "tab:supp_raw", "l" + "rrr" * 2,
          two_universe_header("", ["$\\Delta$", "95\\% CI", "$p_{\\text{Holm}}$"]), body)

    def count(d):
        good, bad = (d.verdict == "more accurate").sum(), (d.verdict == "less accurate").sum()
        return f"{good}/{len(d)}" + (f" ($-${bad})" if bad else "")
    base = {u: (pd.read_csv(root / "main" / "tables" / "hypotheses.csv"),
                pd.read_csv(root / "main" / "tables" / "hypotheses_linear.csv")) for u, root, _ in UNIVERSES}
    body = []
    for fam in ("H1", "C1", "H2"):
        for h in HS:
            row = []
            for u in RAW_TESTS:
                hy, linear_tests = base[u]
                dm = (linear_tests[(linear_tests.hypothesis == "C1") & (linear_tests.benchmark == "Linear") & (linear_tests.h == h) & (linear_tests.block == "in-range")]
                      if fam == "C1" else hy[(hy.hypothesis == fam) & (hy.h == h) & (hy.block == "in-range")])
                ew = RAW_TESTS[u][(RAW_TESTS[u].hypothesis == f"K12-{fam}") & (RAW_TESTS[u].h == h)]
                row += [count(dm), count(ew)]
            body.append(f"{fam}, $h={h}$ & " + " & ".join(row) + "\\\\")
    sz = pd.read_csv(BASE / "main" / "tables" / "ewc_size.csv")
    write("supp_ewc", "K12, H1, C1 and H2 with the Newey--West (DM) and the fixed-smoothing (EWC, "
          "$\\nu=\\lfloor0.4T^{2/3}\\rfloor$ cosine terms, $t_\\nu$ reference) long-run variance: comparisons decided "
          "in favor of the symmetric network out of five; ($-k$): $k$ against it. In a simulation with MA($h-1$) "
          "differentials of the length of the test period (300 replications) the size at 5\\% is "
          + ", ".join(f"{100 * r.size_dm:.1f}" for r in sz.itertuples()) + "\\% for DM and "
          + ", ".join(f"{100 * r.size_ewc:.1f}" for r in sz.itertuples()) + "\\% for EWC at $h=1,5,22$.",
          "tab:supp_ewc", "lcccc",
          two_universe_header("", ["DM", "EWC"]), body)

    body = []
    for h in HS:
        row = []
        for u, root, _ in UNIVERSES:
            s = pd.read_csv(root / "main" / "tables" / "spa.csv").query("h == @h")
            worst = s.loc[s.p_consistent.idxmax()]
            row += [f"{int((s.p_consistent < 0.05).sum())}/{len(s)}", f"{nm2(worst.benchmark)} ({f(worst.p_consistent, 3)})"]
        body.append(f"$h={h}$ & " + " & ".join(row) + "\\\\")
    write("supp_spa", "K13, test of superior predictive ability (\\citealp{hansen2005spa}; studentized, stationary bootstrap, 5,000 "
          "replications) with each classical model as the benchmark and the five symmetric networks as the "
          "alternatives: benchmarks for which the null hypothesis that no network beats it is rejected at 5\\%, and "
          "the benchmark with the largest $p$-value (consistent $p$).", "tab:supp_spa", "lllll",
          two_universe_header("", ["rejected", "hardest benchmark ($p$)"]), body)

    groups = [("N5", "$N=5$"), ("N10", "$N=10$"), ("N20", "$N=20$"), ("N10-conc", "conc."), ("N10-EW", "equal")]
    body = []
    for h in (1, 5):
        body.append(f"\\multicolumn{{11}}{{l}}{{\\emph{{$h={h}$}}}}\\\\")
        L = {u: pd.read_csv(root / "main" / "tables" / f"losses_h{h}.csv") for u, root, _ in UNIVERSES}
        for m in ("HEAVY", "Linear", "MLP", "MLP-sorted", "DeepSets", "PENet", "SetTransformer", "EQCov"):
            row = []
            for u in L:
                d = L[u].set_index(["block", "model"]).qlike_rel_ewma
                row += [f(d.get((g, m)), 3) for g, _ in groups]
            body.append(nm2(m) + " & " + " & ".join(row) + "\\\\")
    write("supp_groups", "In-range QLIKE relative to EWMA by portfolio group: ten-asset portfolios with concentrated and "
          "with equal weights and portfolios of 5, 10 and 20 assets (the MLP fills its first $N$ slots, so the last "
          "slots are trained only on the larger portfolios).", "tab:supp_groups", "l" + "r" * 10,
          two_universe_header("", [g for _, g in groups]), body)

    fy = pd.read_csv(BASE / "main" / "tables" / "fy_quantiles.csv")
    body = [nm2(r.model) + " & " + " & ".join(f(getattr(r, c), 2) for c in ("q1", "q50", "q99", "q99_9", "max"))
            + f" & {f(100 * r.share_above_10, 2)}\\\\" for r in fy.itertuples()]
    write("supp_fy", "Ratio of the forecast to the realized variance one month ahead, in-range portfolios, ETF universe: "
          "quantiles, maximum and share of ratios above 10 (\\%). Under the squared error the loss is dominated by "
          "the few forecasts far above the realized variance.", "tab:supp_fy", "lrrrrrr",
          ["Model & 1\\% & 50\\% & 99\\% & 99.9\\% & max & $>10$ (\\%)\\\\"], body)

    E = {u: pd.read_csv(root / "main" / "tables" / "kept_epochs.csv") for u, root, _ in UNIVERSES}
    body = []
    for m in ("MLP", "DeepSets", "PENet", "SetTransformer", "EQCov", "Linear"):
        row = []
        for u in E:
            d = E[u].set_index(["h", "model"])
            row += [f"{d.loc[(h, m), 'median_epoch']:.0f} ({100 * d.loc[(h, m), 'share_epoch_1_2']:.0f})" for h in HS]
        body.append(nm2(m) + " & " + " & ".join(row) + "\\\\")
    write("supp_epochs", "Kept epoch over the 21 runs (three seeds, seven test years): median and, in parentheses, the "
          "share of runs stopped after the first or second epoch (\\%).", "tab:supp_epochs", "l" + "c" * 6,
          two_universe_header("", [f"$h={h}$" for h in HS]), body)


def table_secondary_and_all():
    sb = load_script("20_scoreboard.py")
    sb.OUT = OUT
    sb.write(sb.ROWS_SUPP, "supp_scoreboard", "tab:supp_scoreboard",
             "Verdicts of the main comparisons (shown as a forest plot in the article) and of the secondary families, both "
             "universes.")
    adapt("supp_scoreboard")
    tt = load_script("10_tables_tex.py")
    tt.OUT = OUT
    tt.main_table(full=True)
    (OUT / "main_all.tex").replace(OUT / "supp_all_models.tex")
    adapt("supp_all_models", "tab:supp_all_models")


def table_other_losses():
    """MSE and the Patton loss with b=-1 relative to EWMA, in-range portfolios."""
    models = ["HIST", "HAR", "HAR-WLS", "HARX", "LogHAR-X", "HEAVY", "GARCH", "GJR", "DCC", "DCCNL", "LW", "EWMANLS",
              "PCA-F", "PortMLP", "MLP", "MLP-sorted", "GRU", "DeepSets", "PENet", "SetTransformer", "SetGRU", "EQCov",
              "Linear", "GBM"]
    cols = []
    for loss in ("MSE", "Patton b=-1"):
        for u, root, _ in UNIVERSES:
            v = pd.read_csv(root / "main" / "tables" / "losses_variants.csv")
            v = v[v.loss == loss].set_index(["h", "model"]).rel_ewma.to_dict()  # Patton loss (MSE: losses_h*.csv)
            for h in (1, 5, 22):
                if loss == "MSE":
                    d = pd.read_csv(root / "main" / "tables" / f"losses_h{h}.csv")
                    d = d[d.block == "in-range"].set_index("model").mse
                    v.update({(h, m): d[m] / d["EWMA"] for m in d.index})
                cols.append({m: v.get((h, m), np.nan) for m in models})
    label = {"HARX": "HAR-X", "DCCNL": "DCC-NL"}
    body = [f"{label.get(m, nm(m))} & " + " & ".join(f(c[m], 3) for c in cols) + "\\\\" for m in models]
    hs = " & ".join(["$h=1$ & $h=5$ & $h=22$"] * 4)
    write("supp_other_losses", "Out-of-sample MSE and the robust loss of \\citet{patton2011} with $b=-1$ relative to EWMA, "
          "in-range portfolios, 2020--2026. `--': not available.", "tab:supp_other_losses", "l" + "rrr" * 4,
          [" & \\multicolumn{6}{c}{MSE} & \\multicolumn{6}{c}{Patton, $b=-1$}\\\\",
           " & \\multicolumn{3}{c}{ETF} & \\multicolumn{3}{c}{Stocks} & \\multicolumn{3}{c}{ETF} & "
           "\\multicolumn{3}{c}{Stocks}\\\\", f"Model & {hs}\\\\"], body)


if __name__ == "__main__":
    table_other_losses()
    T = {u: tests(root) for u, root, _ in UNIVERSES}
    table_ties(T)
    K6 = ["Linear-noasset", "Linear-noeuler", "Linear-nopairs", "Linear-noport", "Linear-portonly"]
    table_tests(T, "K6", [(m, "Linear") for m in K6], "supp_ablation",
                "K6, feature ablation of Linear: difference from Linear with all features ($\\Delta>0$: the variant is "
                "less accurate), in pp, with 95\\% confidence interval and Holm-adjusted $p$ "
                "(Holm over the five variants at each horizon).", "tab:supp_ablation", extra=lambda a, b: "")
    table_tests(T, "K7", [("DeepSets", "MLP-small"), ("DeepSets-L", "MLP")], "supp_size",
                "K7, networks of matched size: difference of the Deep Set from the MLP ($\\Delta<0$: Deep Sets more "
                "accurate), in pp, with 95\\% confidence interval and Holm-adjusted $p$ (Holm "
                "over the two pairs at each horizon).", "tab:supp_size")
    table_size_params()
    table_minvar_check()
    table_yearly()
    table_search()
    table_economic()
    table_costs()
    table_volcorr()
    table_equivalence()
    table_cost_hp()
    table_secondary_and_all()
    table_dropout()
    table_raw()
    print("written:", ", ".join(sorted(p.name for p in OUT.glob("*.tex"))))
