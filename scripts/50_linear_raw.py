"""Linear model with raw inputs (C2).

C2 (tag Linear-RAW): the invariant linear model with raw inputs (no engineered portfolio-level summaries, constant
    anchor), as the networks of A1; settings of Linear, h = 5, seeds 0-2, every test year.
The tests are in scripts/51_raw_spa_ewc.py.

Usage: [PVOL_UNIVERSE=stocks] python scripts/50_linear_raw.py
Resumable: finished (tag, seed) pairs are skipped.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pvol.data import load_panel  # noqa: E402
from pvol.experiment import Experiment  # noqa: E402
from pvol.train import TrainCfg  # noqa: E402

if __name__ == "__main__":
    ex = Experiment("main", load_panel())
    for seed in (0, 1, 2):
        ex.run_nn("Linear", 5, [seed], hp={}, cfg=TrainCfg(raw_inputs=True), tag="Linear-RAW")
    print("Linear-RAW finished", flush=True)
