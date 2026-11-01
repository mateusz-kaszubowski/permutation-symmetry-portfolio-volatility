"""Per-asset volatility network with the EWMA correlation (K15).

K15 (tags VolCorr-asset, VolCorr): w' D R_ewma D w with D = diag(sigma_ewma * exp(delta)) and delta from an MLP shared
    across assets that sees only the features of each asset (src/pvol/models.py, VolCorr). VolCorr-asset is trained on
    single assets (a univariate volatility model combined with the EWMA correlation), VolCorr on the portfolios of the
    main design. Hidden layers (64, 64), not tuned; training protocol of the main design; h = 1, 5, 22, seeds 0-2.
The tests are in scripts/57_volcorr_tests.py.

Usage: [PVOL_UNIVERSE=stocks] python scripts/56_volcorr.py
Resumable: finished (tag, seed) pairs are skipped.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pvol.config import HORIZONS  # noqa: E402
from pvol.data import load_panel  # noqa: E402
from pvol.experiment import Experiment  # noqa: E402
from pvol.train import TrainCfg  # noqa: E402

if __name__ == "__main__":
    ex = Experiment("main", load_panel())
    for h in HORIZONS:
        for seed in (0, 1, 2):
            ex.run_nn("VolCorr", h, [seed], hp={}, cfg=TrainCfg(n_min=1, n_max=1), tag="VolCorr-asset")
            ex.run_nn("VolCorr", h, [seed], hp={}, tag="VolCorr")
    print("VolCorr finished", flush=True)
