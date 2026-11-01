"""Routes to invariance (B1-B4) and Deep Sets with pair summaries (A1-pairs).

shuffle   B1 (tags MLP-P50s, GRU-P500s, ...): fixed pools of A2, but a fresh random order of the assets in every draw.
sorted    B2 (tags MLP-sorted-RAW, MLP-sorted-P50, MLP-sorted-P500): canonicalisation under the training conditions.
tta       B3 (tags MLP-TTA, GRU-TTA): main-design networks whose forecast averages 8 random orderings of the constituents.
pairs     A1-pairs (tags DeepSetsP, DeepSetsP-RAW): Deep Sets with the weight-averaged correlation of each constituent as extra input.
subsample B4 (PVOL_UNIVERSE=stocks54): the main networks on a 54-stock subsample, h = 1 and 5.

Usage: python scripts/23_routes_to_invariance.py shuffle sorted tta pairs        (PVOL_UNIVERSE=stocks ... for the stock universe)
       PVOL_UNIVERSE=stocks54 python scripts/23_routes_to_invariance.py subsample
Resumable: finished (tag, seed) pairs are skipped.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pvol.data import load_panel  # noqa: E402
from pvol.experiment import Experiment, load_tuned_hp  # noqa: E402
from pvol.train import TrainCfg  # noqa: E402

SUBSAMPLE_MODELS = ["MLP", "GRU", "DeepSets", "PENet", "SetTransformer", "SetGRU", "EQCov"]

if __name__ == "__main__":
    todo = sys.argv[1:] or ["shuffle", "sorted", "tta", "pairs"]
    hp = load_tuned_hp()
    hp.setdefault("DeepSetsP", hp["DeepSets"])  # same configuration as Deep Sets, no separate tuning
    ex = Experiment("main", load_panel())
    for e in todo:
        if e == "shuffle":
            for P in (50, 500):
                for seed in (0, 1):
                    for m in ("MLP", "GRU"):
                        ex.run_nn(m, 5, [seed], hp=hp[m], cfg=TrainCfg(portfolio_pool=P, pool_shuffle=True),
                                  tag=f"{m}-P{P}s")
        elif e == "sorted":
            for seed in (0, 1, 2):
                ex.run_nn("MLP-sorted", 5, [seed], hp=hp["MLP-sorted"], cfg=TrainCfg(raw_inputs=True),
                          tag="MLP-sorted-RAW")
            for P in (50, 500):
                for seed in (0, 1):
                    ex.run_nn("MLP-sorted", 5, [seed], hp=hp["MLP-sorted"], cfg=TrainCfg(portfolio_pool=P),
                              tag=f"MLP-sorted-P{P}")
        elif e == "tta":
            for h in (5, 1):
                for seed in (0, 1, 2):
                    for m in ("MLP", "GRU"):
                        ex.run_nn(m, h, [seed], hp=hp[m], tag=f"{m}-TTA", tta_k=8)
        elif e == "pairs":
            for seed in (0, 1, 2):
                ex.run_nn("DeepSetsP", 5, [seed], hp=hp["DeepSetsP"], cfg=TrainCfg(raw_inputs=True), tag="DeepSetsP-RAW")
                ex.run_nn("DeepSetsP", 5, [seed], hp=hp["DeepSetsP"], tag="DeepSetsP")
        elif e == "subsample":
            for h in (1, 5):
                for seed in (0, 1, 2):
                    for m in SUBSAMPLE_MODELS:
                        ex.run_nn(m, h, [seed], hp=hp[m])
        else:
            raise SystemExit(f"unknown experiment {e}")
        print(f"{e} finished", flush=True)
