"""Comparison of four training settings on the tuning fold (training to 2016, validation 2017-2018, h = 5).

A: learning rate 1e-3, weight average 0.995; B: 3e-4, 0.995; C: 3e-4 with weight decay 1e-2, 0.995;
D: 1e-4, 0.99 (the setting of TrainCfg). Deep Sets, the MLP and the Set Transformer with their default sizes, 30 epochs
without early stopping, seeds 0 and 1; the validation QLIKE per epoch is averaged over the seeds. Writes
results*/tuning/protocol_study.txt.

Usage: [PVOL_UNIVERSE=stocks] python scripts/55_protocol_study.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402

from pvol.config import RESULTS  # noqa: E402
from pvol.data import load_panel  # noqa: E402
from pvol.dataset import SEQ_MARGIN, PanelGPU, Split, year_bounds  # noqa: E402
from pvol.train import TrainCfg, train_model  # noqa: E402

H = 5
SETTINGS = {"A lr1e-3 ema.995": dict(lr=1e-3, ema_decay=0.995), "B lr3e-4 ema.995": dict(lr=3e-4, ema_decay=0.995),
            "C lr3e-4 wd1e-2": dict(lr=3e-4, weight_decay=1e-2, ema_decay=0.995), "D lr1e-4 ema.99": dict(lr=1e-4, ema_decay=0.99)}
EPOCHS = 30

if __name__ == "__main__":
    panel = PanelGPU(load_panel())
    v0 = year_bounds(panel.dates, 2017)[0]
    v1 = year_bounds(panel.dates, 2018)[1]
    split = Split(train=(panel.first_valid + SEQ_MARGIN, v0 - 1 - H), val=(v0, v1 - H))
    lines = ["# Best validation QLIKE (at the epoch shown) and QLIKE at epochs 5, 10, 20 and 29 (epochs counted from 0) on "
             "the tuning fold (training to 2016, validation 2017-2018, h = 5), mean of seeds 0 and 1,",
             "# for four training settings: learning rate and decay of the weight average (ema) or weight decay (wd). "
             "D is the setting used."]
    for name in ("DeepSets", "MLP", "SetTransformer"):
        for label, kw in SETTINGS.items():
            cfg = TrainCfg(max_epochs=EPOCHS, patience=EPOCHS, **kw)
            hist = [np.array([r[2] for r in train_model(name, panel, split, H, seed=s, cfg=cfg)[1]["history"]]) for s in (0, 1)]
            n = min(len(x) for x in hist)
            v = np.mean([x[:n] for x in hist], 0)
            at = " ".join(f"ep{e} {v[e]:.4f}" for e in (5, 10, 20, 29) if e < n)
            lines.append(f"{name:14s} {label:18s} best {v.min():.4f} @ep {v.argmin():2d} | {at}")
            print(lines[-1], flush=True)
    (RESULTS / "tuning").mkdir(parents=True, exist_ok=True)
    (RESULTS / "tuning" / "protocol_study.txt").write_text("\n".join(lines) + "\n")
