"""Broader search (K3, K14) and sort key (K4).

wide:    main design with the settings of the broader search (scripts/36_tune_wide.py, best_hp_wide.json), tag '<model>-wide';
         networks and Linear, h = 1, 5, 22, seeds 0-2.
wideh:   the same with the settings selected at the horizon of evaluation (best_hp_wide_h1.json, best_hp_wide_h22.json),
         tag '<model>-wideh', h = 1 and 22, seeds 0-2 (K14; at h = 5 these are the settings of K3).
sortkey: MLP with slots ordered by asset index (MLP-idsort, settings of MLP-sorted), h = 5: main design (tag MLP-idsort,
         seeds 0-2), raw inputs (MLP-idsort-RAW, seeds 0-2), fixed pool of 50 portfolios (MLP-idsort-P50, seeds 0-1).

Usage: [PVOL_UNIVERSE=stocks] python scripts/37_search_sort_key.py sortkey wide wideh
Resumable: finished (tag, seed) pairs are skipped.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pvol.config import RESULTS  # noqa: E402
from pvol.data import load_panel  # noqa: E402
from pvol.experiment import Experiment, load_tuned_hp  # noqa: E402
from pvol.train import TrainCfg  # noqa: E402

WIDE_MODELS = ["MLP", "GRU", "DeepSets", "PENet", "SetTransformer", "SetGRU", "EQCov", "Linear"]

if __name__ == "__main__":
    todo = sys.argv[1:] or ["sortkey", "wide"]
    ex = Experiment("main", load_panel())
    for e in todo:
        if e == "sortkey":
            hp = load_tuned_hp()["MLP-sorted"]
            for seed in (0, 1, 2):
                ex.run_nn("MLP-idsort", 5, [seed], hp=hp, tag="MLP-idsort")
                ex.run_nn("MLP-idsort", 5, [seed], hp=hp, cfg=TrainCfg(raw_inputs=True), tag="MLP-idsort-RAW")
            for seed in (0, 1):
                ex.run_nn("MLP-idsort", 5, [seed], hp=hp, cfg=TrainCfg(portfolio_pool=50), tag="MLP-idsort-P50")
        elif e == "wide":
            best = json.loads((RESULTS / "tuning" / "best_hp_wide.json").read_text())
            for h in (5, 1, 22):
                for seed in (0, 1, 2):
                    for m in WIDE_MODELS:
                        ex.run_nn(m, h, [seed], hp=best[m]["hp"], cfg=TrainCfg(**best[m]["cfg"]), tag=f"{m}-wide")
        elif e == "wideh":
            for h in (1, 22):
                best = json.loads((RESULTS / "tuning" / f"best_hp_wide_h{h}.json").read_text())
                for seed in (0, 1, 2):
                    for m in WIDE_MODELS:
                        ex.run_nn(m, h, [seed], hp=best[m]["hp"], cfg=TrainCfg(**best[m]["cfg"]), tag=f"{m}-wideh")
        else:
            raise SystemExit(f"unknown experiment {e}")
        print(f"{e} finished", flush=True)
