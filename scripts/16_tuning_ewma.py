"""Validation QLIKE of EWMA on the tuning fold (training origins up to the end of 2016, validation 2017-2018, h = 5).

EWMA is scored on the same fixed validation portfolio-date pairs on which the networks are selected (train.fixed_batches
with the seed of train_model), so its loss is the reference for the validation losses of scripts/03_tune.py.
Writes results*/tuning/ewma_validation.csv.

Usage: [PVOL_UNIVERSE=stocks] python scripts/16_tuning_ewma.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from pvol.config import RESULTS  # noqa: E402
from pvol.data import load_panel  # noqa: E402
from pvol.dataset import SEQ_MARGIN, PanelGPU, Split, year_bounds  # noqa: E402
from pvol.train import TrainCfg, evaluate_batches, fixed_batches  # noqa: E402

H = 5


class EWMA(torch.nn.Module):
    """The anchor of the networks: the log EWMA forecast of the portfolio variance."""

    def forward(self, inp):
        return inp["base"]


if __name__ == "__main__":
    panel = PanelGPU(load_panel())
    v0, v1 = year_bounds(panel.dates, 2017)[0], year_bounds(panel.dates, 2018)[1]
    split = Split(train=(panel.first_valid + SEQ_MARGIN, v0 - 1 - H), val=(v0, v1 - H))
    cfg = TrainCfg()
    panel.fit_scalers(split.train, torch.Generator(device=panel.device).manual_seed(0))
    val = fixed_batches(panel, split.val, cfg.val_batches, cfg, seed=987 + H)
    loss = evaluate_batches(EWMA(), panel, val, H, gru_len=0)
    out = RESULTS / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(h=H, pairs=cfg.val_batches * cfg.batch, qlike=loss)]).to_csv(out / "ewma_validation.csv", index=False)
    print(f"EWMA validation QLIKE on the tuning fold (h = {H}): {loss:.4f}")
