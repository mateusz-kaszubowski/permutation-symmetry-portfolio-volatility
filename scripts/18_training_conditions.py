"""Training conditions of A1-A3, h = 5.

A1 (tag -RAW):   raw inputs: constant anchor, no portfolio-level features, Euler contributions or correlations with
                 the portfolio.
A2 (tag -P50/-P500): training and validation on a fixed pool of 50 / 500 portfolios with a fixed asset order.
A3 (tag -N54):   training on portfolios with up to 54 assets; the MLP gets 54 slots.

Usage: python scripts/18_training_conditions.py pool raw n54      (PVOL_UNIVERSE=stocks for the stock universe)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pvol.data import load_panel  # noqa: E402
from pvol.experiment import Experiment, load_tuned_hp  # noqa: E402
from pvol.train import TrainCfg  # noqa: E402

H = 5
RAW_MODELS = ["MLP", "DeepSets", "PENet", "SetTransformer", "GRU", "SetGRU"]
POOL_MODELS = ["MLP", "DeepSets", "PENet", "SetTransformer", "GRU", "SetGRU", "EQCov"]
N54_MODELS = ["MLP", "DeepSets", "PENet", "SetTransformer", "EQCov"]

if __name__ == "__main__":
    todo = sys.argv[1:] or ["pool", "raw", "n54"]
    hp = load_tuned_hp()
    ex = Experiment("main", load_panel())
    for e in todo:
        if e == "pool":
            for P in (50, 500):
                for seed in (0, 1):
                    for m in POOL_MODELS:
                        ex.run_nn(m, H, [seed], hp=hp[m], cfg=TrainCfg(portfolio_pool=P), tag=f"{m}-P{P}")
        elif e == "raw":
            for seed in (0, 1, 2):
                for m in RAW_MODELS:
                    ex.run_nn(m, H, [seed], hp=hp[m], cfg=TrainCfg(raw_inputs=True), tag=f"{m}-RAW")
        elif e == "n54":
            for seed in (0, 1):
                for m in N54_MODELS:
                    hp_m = {**hp[m], "n_slots": 54} if m == "MLP" else hp[m]
                    ex.run_nn(m, H, [seed], hp=hp_m, cfg=TrainCfg(n_max=54), tag=f"{m}-N54")
        else:
            raise SystemExit(f"unknown experiment {e}")
        print(f"{e} finished", flush=True)
