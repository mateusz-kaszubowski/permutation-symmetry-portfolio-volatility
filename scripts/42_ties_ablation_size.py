"""Checks K5-K7: ties of the sort, feature ablation of Linear, networks of matched size.

K5: MLP sorted by descending weight with ties broken by asset index (MLP-tsort, settings of MLP-sorted).
K6: Linear without one group of features (Linear-noasset, -noeuler, -nopairs, -noport) and with the portfolio-level
    backcasts only (Linear-portonly).
K7: capacity-matched pairs: MLP-small (MLP with hidden (16, 16), about the size of Deep Sets) and DeepSets-L (Deep Sets
    with d = 400, about the size of the MLP); all other settings as tuned.
Main design, h = 1, 5, 22, seeds 0-2, every test year.

Usage: [PVOL_UNIVERSE=stocks] python scripts/42_ties_ablation_size.py ties ablation size
Resumable: finished (tag, seed) pairs are skipped.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pvol.data import load_panel  # noqa: E402
from pvol.experiment import Experiment, load_tuned_hp  # noqa: E402

ABLATION_MODELS = ["Linear-noasset", "Linear-noeuler", "Linear-nopairs", "Linear-noport", "Linear-portonly"]

if __name__ == "__main__":
    todo = sys.argv[1:] or ["ties", "ablation", "size"]
    ex = Experiment("main", load_panel())
    hp = load_tuned_hp()
    for e in todo:
        for h in (5, 1, 22):
            for seed in (0, 1, 2):
                if e == "ties":
                    ex.run_nn("MLP-tsort", h, [seed], hp=hp["MLP-sorted"], tag="MLP-tsort")
                elif e == "ablation":
                    for m in ABLATION_MODELS:
                        ex.run_nn(m, h, [seed], hp={})
                elif e == "size":
                    ex.run_nn("MLP", h, [seed], hp={**hp["MLP"], "hidden": [16, 16]}, tag="MLP-small")
                    ex.run_nn("DeepSets", h, [seed], hp={**hp["DeepSets"], "d": 400, "phi": [400], "rho": [400]},
                              tag="DeepSets-L")
                else:
                    raise SystemExit(f"unknown experiment {e}")
        print(f"{e} finished", flush=True)
