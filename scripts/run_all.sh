#!/usr/bin/env bash
# Full pipeline in the order used for the paper. The neural steps need a CUDA GPU and take several days in total;
# the main training scripts skip finished (tag, seed) pairs, so the script can be re-run after an interruption
# (see README.md for the steps that are recomputed in full). Tables and figures are written to paper/tables and paper/figures.
set -eo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python}
mkdir -p logs results/{main,heldout}/tables results_stocks/{main,heldout}/tables results_stocks54/main/tables

universe() {  # $1 = etf | stocks, $2 = horizons of the held-out-assets experiment, $3 = parts of 25_linear_seeds_refit.py
    export PVOL_UNIVERSE=$1
    $PY scripts/01_download.py
    $PY scripts/02_build_panel.py
    $PY scripts/07_descriptives.py
    if [ "$1" = etf ]; then
        $PY scripts/03_tune.py
        $PY scripts/55_protocol_study.py
    else  # the hyper-parameters are tuned on the ETF universe
        mkdir -p results_stocks/tuning
        cp results/tuning/best_hp.json results_stocks/tuning/
    fi
    $PY scripts/04_baselines.py
    $PY scripts/04c_shrinkage.py
    $PY scripts/26_range_benchmarks.py
    $PY scripts/27_gbm.py
    $PY scripts/05_train.py --exp main
    $PY scripts/05_train.py --exp heldout --h $2
    $PY scripts/08_ablation_datasize.py
    $PY scripts/13_minvar.py
    $PY scripts/18_training_conditions.py pool raw n54
    $PY scripts/23_routes_to_invariance.py shuffle sorted tta pairs
    $PY scripts/25_linear_seeds_refit.py $3
    $PY scripts/06_evaluate.py main
    $PY scripts/06_evaluate.py heldout
    $PY scripts/12_vol_targeting.py
    $PY scripts/15_hypotheses.py main heldout
    $PY scripts/15_hypotheses.py main heldout --no-range
    $PY scripts/17_economic_tests.py
    $PY scripts/19_training_conditions_tests.py
    $PY scripts/24_routes_tests.py
    $PY scripts/29_linear_tests.py
    $PY scripts/28_robustness_analyses.py
    $PY scripts/22_compute_cost.py
}

universe etf "5 1 22" linear 2>&1 | tee -a logs/etf.log
# alternative training protocol and refit protocol (ETF: protocol B at h = 5; stocks: at h = 1), tuning at other horizons
PVOL_UNIVERSE=etf $PY scripts/14_robust_protocol.py 5
PVOL_UNIVERSE=etf $PY scripts/25_linear_seeds_refit.py refit
PVOL_UNIVERSE=etf $PY scripts/15_hypotheses.py main --suffix=-P2
PVOL_UNIVERSE=etf $PY scripts/15_hypotheses.py main --suffix=-RF
PVOL_UNIVERSE=etf $PY scripts/03_tune.py --h 1 --tag h1
PVOL_UNIVERSE=etf $PY scripts/03_tune.py --h 22 --tag h22
PVOL_UNIVERSE=etf $PY scripts/31_determinism.py
# ETF universe only: EQ-Cov diagnostics, EWMA loss on the tuning fold
PVOL_UNIVERSE=etf $PY scripts/11_eqcov_diagnostics.py
PVOL_UNIVERSE=etf $PY scripts/16_tuning_ewma.py

universe stocks 5 "linear seeds" 2>&1 | tee -a logs/stocks.log
PVOL_UNIVERSE=stocks $PY scripts/14_robust_protocol.py 1
PVOL_UNIVERSE=stocks $PY scripts/25_linear_seeds_refit.py refit
PVOL_UNIVERSE=stocks $PY scripts/15_hypotheses.py main --suffix=-P2
PVOL_UNIVERSE=stocks $PY scripts/15_hypotheses.py main --suffix=-RF
PVOL_UNIVERSE=stocks $PY scripts/03_tune.py --h 5 --tag h5
PVOL_UNIVERSE=stocks $PY scripts/03_tune.py --h 1 --tag h1

# 54-stock subsample (B4 and its seeds 3-9); prices are read from the stock universe
export PVOL_UNIVERSE=stocks54
$PY scripts/02_build_panel.py
mkdir -p results_stocks54/tuning
cp results/tuning/best_hp.json results_stocks54/tuning/
$PY scripts/04_baselines.py main
$PY scripts/23_routes_to_invariance.py subsample
$PY scripts/25_linear_seeds_refit.py seeds
$PY scripts/24_routes_tests.py
$PY scripts/29_linear_tests.py
unset PVOL_UNIVERSE

# checks over both universes: second data source, tuning at other horizons
PVOL_UNIVERSE=etf $PY scripts/33_data_crosscheck.py
PVOL_UNIVERSE=stocks $PY scripts/33_data_crosscheck.py
$PY scripts/34_tuning_check.py
PVOL_UNIVERSE=etf $PY scripts/35_refit_summary.py
PVOL_UNIVERSE=stocks $PY scripts/35_refit_summary.py
# sort key (K4) and broader search with validation over three years at h = 5 (K3) and at h = 1 and 22 (K14)
for u in etf stocks; do
    PVOL_UNIVERSE=$u $PY scripts/37_search_sort_key.py sortkey
    PVOL_UNIVERSE=$u $PY scripts/36_tune_wide.py
    PVOL_UNIVERSE=$u $PY scripts/37_search_sort_key.py wide
    PVOL_UNIVERSE=$u $PY scripts/36_tune_wide.py 1
    PVOL_UNIVERSE=$u $PY scripts/36_tune_wide.py 22
    PVOL_UNIVERSE=$u $PY scripts/37_search_sort_key.py wideh
    PVOL_UNIVERSE=$u $PY scripts/38_search_sort_key_tests.py
    PVOL_UNIVERSE=$u $PY scripts/39_equivalence.py
    PVOL_UNIVERSE=$u $PY scripts/49_mcs_tmax.py
    PVOL_UNIVERSE=$u $PY scripts/40_anchor_distance.py
    PVOL_UNIVERSE=$u $PY scripts/41_sorted_ties.py
done
# K5 (ties of the sort), K6 (feature ablation of Linear), K7 (matched size), K9 (min-var optimizer),
# K10 (dropout of MLP-sorted); K8 (results by test year) reads both universes
for u in etf stocks; do
    PVOL_UNIVERSE=$u $PY scripts/42_ties_ablation_size.py ties size ablation
    PVOL_UNIVERSE=$u $PY scripts/41_sorted_ties.py MLP-tsort
    PVOL_UNIVERSE=$u $PY scripts/44_minvar_check.py
    PVOL_UNIVERSE=$u $PY scripts/45_ties_ablation_size_tests.py
    PVOL_UNIVERSE=$u $PY scripts/47_dropout_control.py
    PVOL_UNIVERSE=$u $PY scripts/48_dropout_tests.py
done
$PY scripts/43_yearly.py
# C2 (Linear with raw inputs), K13 (SPA), K12 (fixed-smoothing tests), K11 (EQ-Cov min-var exact)
for u in etf stocks; do
    PVOL_UNIVERSE=$u $PY scripts/50_linear_raw.py
    PVOL_UNIVERSE=$u $PY scripts/51_raw_spa_ewc.py
    PVOL_UNIVERSE=$u $PY scripts/52_minvar_eqcov_exact.py
done
# K15 (per-asset volatility network with the EWMA correlation), K16 (volatility targeting with costs and leverage limit)
for u in etf stocks; do
    PVOL_UNIVERSE=$u $PY scripts/56_volcorr.py
    PVOL_UNIVERSE=$u $PY scripts/57_volcorr_tests.py
    PVOL_UNIVERSE=$u $PY scripts/58_voltarget_costs.py
done
# precision of every comparison and global multiplicity over all families
$PY scripts/32_precision_multiplicity.py

# tables and figures for both universes
$PY scripts/10_tables_tex.py
$PY scripts/20_scoreboard.py
$PY scripts/21_hypothesis_tables.py
$PY scripts/30_robustness_tables.py
$PY scripts/09_figures.py
$PY scripts/53_forest_plot.py
$PY scripts/54_learning_curves.py
$PY scripts/46_supplement_tables.py
bash scripts/collect_results.sh
