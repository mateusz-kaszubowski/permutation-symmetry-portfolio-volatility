# Permutation-symmetric neural networks for forecasting the volatility of multi-asset portfolios

[![tests](https://github.com/mateusz-kaszubowski/permutation-symmetry-portfolio-volatility/actions/workflows/tests.yml/badge.svg)](https://github.com/mateusz-kaszubowski/permutation-symmetry-portfolio-volatility/actions/workflows/tests.yml)

Code for the paper. It downloads the data, builds the panels, estimates every model and writes every table and figure
of the paper and its supplementary material. All data are public daily prices from Yahoo Finance.

## Requirements
Python 3.10 and an NVIDIA GPU with CUDA 12.4 (the experiment runner uses it by default for all models). The
results in the paper were computed on Windows 11 with one GPU.
```
py -3.10 -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python -m pip install -r requirements.txt
```
On Linux, use `python3.10` and `.venv/bin/python` (set `PY=.venv/bin/python` for `run_all.sh`).

## Reproducing the paper
```
bash scripts/run_all.sh
```
runs the whole pipeline. Training all networks takes several days on one GPU. Forecasts are saved per (model, seed)
and the scripts that train them (05, 08, 14, 18, 23, 25, 37, 42, 47, 50, 56) skip finished pairs, so an interrupted run
can be restarted with the same command; 03 and 36 resume from their logs and 44 from its yearly files. An existing price download is kept (`01_download.py --force` downloads again), so a restart does not mix
forecasts from two versions of the data. Everything else is recomputed in full on a restart, including the scripts that retrain networks for one analysis
(11, 13, 31, 41, 52, 55).

`PVOL_SMOKE=1 bash scripts/run_all.sh` on a fresh clone runs the same pipeline on simulated prices of the same
tickers (from 2012, test years 2020 and 2026, two training epochs of two steps). It checks that every script runs
and writes its outputs, not the results, and takes about five hours on one GPU; its results overwrite those of a
full run, so use a separate clone.

The universe is chosen by the environment variable `PVOL_UNIVERSE`:

| value | universe | data | results |
|---|---|---|---|
| `etf` (default) | 54 exchange-traded funds | `data/raw`, `data/processed` | `results/` |
| `stocks` | 122 large US stocks | `data/raw_stocks`, `data/processed_stocks` | `results_stocks/` |
| `stocks54` | stratified subsample of 54 of the stocks | `data/processed_stocks54` | `results_stocks54/` |

The sample runs from 2007-12-19 to 2026-09-18 (`src/pvol/config.py`); the test years are 2020-2026. Yahoo Finance
revises adjusted prices after dividends and splits, so a later download can differ slightly from the one used in the
paper. The price files are not redistributed because of the provider's terms of use. The snapshot used in the paper
was downloaded on 2026-09-21; its SHA-256 checksums are:

```
8c8af11acaa49e9c10ab35155d9d9c0e366676031ff62590e966e70e76491aca  data/raw/ohlcv_adjusted.parquet
2e40d6c8bc2ca99dfda49ce9455493ffafe41832cdb7b61f566591d89c866d53  data/raw_stocks/ohlcv_adjusted.parquet
c1016bac854a0eb7810a76155564b837efeb0b56eeefb94c900c255fe4cd48b2  data/processed/panel.npz
e4d50d89cdf83ec38ce90de37617b2322e4555371907c550b0c5ffa98d625634  data/processed_stocks/panel.npz
b20fc3c3445ca0d51accb63dec8b30a33b1f4dcad4aa521a5ea8e0b5bf9bee52  data/processed_stocks54/panel.npz
```

`replication/` holds the result tables behind the paper and does not depend on a new download. The LaTeX tables and
figures are rebuilt from `results*/`, which also needs the forecasts: rerun the pipeline, or ask the authors for the
forecasts (about 16 GB).

## Scripts

Scripts and result files use the labels of the article: H1, H2, A1-A3, B1-B4, C1-C3, G1 (G1a-G1c), G2 and the
robustness checks K1-K16 of Table S1 of the supplementary material. A1-pairs is the comparison of Deep Sets with pair
summaries against the MLP with raw inputs; rows marked "aux" are auxiliary comparisons reported with a family.

Data, tuning and benchmarks

| script | purpose |
|---|---|
| `01_download.py` | daily adjusted OHLCV from Yahoo Finance |
| `02_build_panel.py` | returns, variance proxies, asset features, correlations, EWMA covariance (`panel.npz`) |
| `55_protocol_study.py` | comparison of four training settings on the tuning fold (`protocol_study.txt`; the setting used is D) |
| `03_tune.py` | hyper-parameter selection on a fold before the test period (four configurations per network; `--h`, `--tag` repeat the search at another horizon) |
| `16_tuning_ewma.py` | validation QLIKE of EWMA on the same fold, the reference for the validation losses of 03 (`ewma_validation.csv`) |
| `04_baselines.py`, `04c_shrinkage.py` | targets and classical benchmarks: HIST, EWMA, HAR, HAR-X, GARCH, GJR-GARCH, DCC, Ledoit-Wolf, nonlinear shrinkage, DCC-NL |
| `26_range_benchmarks.py` | HAR-WLS, LogHAR-X, HEAVY and a PCA factor correlation model |
| `27_gbm.py` | gradient-boosting benchmark on the features of the linear set model |
| `07_descriptives.py` | descriptive statistics of the universe |

Experiments (forecasts in `results*/<experiment>/h<h>/preds/<tag>__s<seed>.npy`)

| script | experiment | tags |
|---|---|---|
| `05_train.py` | main design (`--exp main`) and assets never seen in training (`--exp heldout`) | model names |
| `08_ablation_datasize.py` | shortened training histories | `-K2`, `-K4` |
| `14_robust_protocol.py` | alternative training protocol (ETF at h = 5, stocks at h = 1) | `-P2` |
| `18_training_conditions.py` | raw inputs, fixed portfolio pools, portfolios of up to 54 assets | `-RAW`, `-P50`, `-P500`, `-N54` |
| `23_routes_to_invariance.py` | fixed pools with random orderings, sorted inputs, averaging over orderings, Deep Sets with pair summaries, 54-stock subsample | `-P50s`, `-P500s`, `MLP-sorted-*`, `-TTA`, `DeepSetsP*` |
| `25_linear_seeds_refit.py` | `linear`: linear set model (C1); `seeds`: seeds 3-9 at h = 1 (K1); `refit`: refit on training + validation data (K2) | `Linear`, seeds 3-9, `-RF` |
| `13_minvar.py` | minimum-variance portfolios built with each model | |
| `11_eqcov_diagnostics.py` | correlation weights and volatility adjustments of EQ-Cov | |

Evaluation and tests (tables in `results*/<experiment>/tables/`)

| script | output |
|---|---|
| `06_evaluate.py` | losses by portfolio group and block; for the main experiment the model confidence set (`mcs_h*.csv`), permutation consistency and value-at-risk |
| `12_vol_targeting.py`, `17_economic_tests.py` | volatility targeting and the tests of the economic evaluation |
| `15_hypotheses.py` | H1, H2, G1 and G2 (`--no-range` writes `hypotheses_h2norange.csv`, H2 against the classical models without range-based variances; `--suffix=-P2` / `-RF` the alternative protocols, `hypotheses_P2.csv`, `hypotheses_RF.csv`) |
| `19_training_conditions_tests.py`, `24_routes_tests.py`, `29_linear_tests.py` | tests of the training conditions, routes to invariance and linear model |
| `28_robustness_analyses.py` | robustness: Patton loss, test samples without 2020 and without 2026, Giacomini-White test, agreement of seeds, value-at-risk with Student-t and filtered historical simulation |
| `22_compute_cost.py` | parameters, training and forecast time |
| `31_determinism.py` | retrains three networks twice and compares with the stored forecasts (`determinism.csv`) |
| `32_precision_multiplicity.py` | confidence intervals and detectable effects of every comparison; Holm and Benjamini-Hochberg over all comparisons |
| `33_data_crosscheck.py` | prices against a second source (Nasdaq, unadjusted) |
| `34_tuning_check.py` | hyper-parameters in use against the search repeated at other horizons and on the stocks |
| `35_refit_summary.py` | accuracy of the networks under the main, B and refit protocols; Linear against the symmetric networks under each |
| `36_tune_wide.py` | K3: random search (21 configurations per learned model) with validation over 2017-2019 at h = 5; with argument 1 or 22, the same search at that horizon (K14) |
| `37_search_sort_key.py` | `sortkey`: MLP with slots ordered by asset index (K4); `wide`: main design with the settings of the search (K3, tag `-wide`); `wideh`: with the settings selected at h = 1 and 22 (K14, tag `-wideh`) |
| `38_search_sort_key_tests.py` | tests of the broader search (K3a-K3d, K14a-K14c) and the sort key (K4a, K4b), best validation epochs |
| `39_equivalence.py` | confidence intervals and equivalence tests (TOST, margins of 1 and 2 points; Bonferroni over the five networks) for C1 and H1 |
| `49_mcs_tmax.py` | Model Confidence Set with the T_max statistic for the in-range portfolios (the main tables use the range statistic) and the Diebold-Mariano test of LW against HEAVY |
| `40_anchor_distance.py` | distance of the learned forecasts from the EWMA anchor and from the Linear correction |
| `41_sorted_ties.py` | permutation consistency of MLP-sorted (or of the model given as argument, e.g. `MLP-tsort`) on equally weighted portfolios (retrains it, h=5) |
| `42_ties_ablation_size.py` | `ties`: MLP sorted by weight, ties broken by asset index (K5); `ablation`: Linear without one feature group (K6); `size`: networks of matched size (K7) |
| `43_yearly.py` | K8: losses and Diebold-Mariano tests per test year |
| `44_minvar_check.py` | K9: minimum-variance weights from the exact solution (covariance models) and from several starts (networks) against the stored weights |
| `45_ties_ablation_size_tests.py` | tests of K5-K7 |
| `47_dropout_control.py` | K10: MLP-sorted retrained with the dropout of the MLP under the conditions of B2 |
| `48_dropout_tests.py` | tests of K10 (`hypotheses_dropout.csv`) |
| `50_linear_raw.py` | C2: Linear with raw inputs, h=5 |
| `51_raw_spa_ewc.py` | tests: C2, fixed-smoothing Diebold-Mariano tests (K12), SPA test (K13), forecast-to-realized ratios |
| `52_minvar_eqcov_exact.py` | K11: minimum-variance weights of EQ-Cov from the exact solution of its convex problem |
| `56_volcorr.py` | K15: per-asset volatility network with the EWMA correlation, trained on single assets (`VolCorr-asset`) and on portfolios (`VolCorr`), h = 1, 5, 22 |
| `57_volcorr_tests.py` | tests of K15 (`hypotheses_volcorr.csv`) |
| `58_voltarget_costs.py` | K16: volatility targeting with a limit on leverage and transaction costs (`voltarget_costs.csv`, `voltarget_costs_tests.csv`) |
| `46_supplement_tables.py` | tables of the supplementary material (all models, secondary families, equivalence, robustness checks, economic evaluation, cost and hyper-parameters) in `paper/supplement/tables/` |

Every hypothesis is decided by the same rule: model a is more accurate than model b if its mean QLIKE is lower and the
Diebold-Mariano test with the Harvey-Leybourne-Newbold correction rejects at 5% after a Holm correction within the
family of comparisons. The status of the hypotheses is described in Section III-C of the article.

Tables and figures (read both universes, run once)

| script | output |
|---|---|
| `10_tables_tex.py`, `20_scoreboard.py`, `21_hypothesis_tables.py`, `30_robustness_tables.py` | LaTeX tables in `paper/tables/` |
| `09_figures.py`, `53_forest_plot.py`, `54_learning_curves.py` | PDF figures in `paper/figures/`, including the forest plot of H1, C1 and H2 and the learning curves from the stored training histories (need a LaTeX installation, the figures use `usetex`; `TEXBIN` can point to its `bin` directory) |

The result tables are generated from the CSV files above; Table 1 and Table S1 (hypotheses and rules) and a few
counts in captions are written by hand.

`scripts/collect_results.sh` copies the result tables, tuning logs and descriptive statistics to `replication/`, which
is under version control. Not copied: the forecasts, the training histories (`info/`), targets and origins, and the
per-subset minimum-variance records of `13_minvar.py` and `44_minvar_check.py` (those of `52_minvar_eqcov_exact.py` are
copied).

Training is deterministic on a given GPU and software stack: retraining with the same seed reproduces the stored
forecasts to within 1e-6 (`31_determinism.py`); other hardware or library versions can give slightly different
forecasts.

## Code
- `src/pvol/config.py`: paths, sample, horizons, test years
- `src/pvol/universe.py`: ETF and stock universes with asset classes and sectors
- `src/pvol/data.py`: variance proxies, asset features, rolling and EWMA correlations
- `src/pvol/dataset.py`: GPU panel, random portfolio sampler, features, time splits
- `src/pvol/portfolios.py`: fixed evaluation portfolios and portfolio blocks
- `src/pvol/models.py`: portfolio-level MLP, MLP, GRU, sorted MLP, Deep Sets, PE-Net, Set Transformer, Set-GRU, EQ-Cov, linear set model
- `src/pvol/train.py`: QLIKE training with EMA weights and early stopping
- `src/pvol/experiment.py`: expanding-window protocol, one refit per test year
- `src/pvol/baselines.py`, `src/pvol/baselines_range.py`: classical benchmarks (GBM: `scripts/27_gbm.py`)
- `src/pvol/evaluate.py`: losses, Diebold-Mariano tests, model confidence set, value-at-risk backtests

Tests: `.venv\Scripts\python -m pytest tests --cov=src/pvol` (every line of `src/pvol` is covered; about 30 s on
one GPU, the CPU works too). They check the permutation symmetry of the models, the alignment of the targets and the
absence of look-ahead, the benchmarks and the Diebold-Mariano, fixed-smoothing and Giacomini-White tests against a direct computation, a reference
implementation or a simulation (the SPA and equivalence tests and most numbered scripts are not unit-tested), short training runs of the networks and of the experiment runner, the Holm and Benjamini-Hochberg
corrections of the scripts, and that the verdicts and corrected p-values stored in `replication/` follow the
decision rule. They use `data/processed/panel.npz` (`scripts/02_build_panel.py`) when it exists; without the data they
build a panel of the same shape from simulated prices, so they also run on a fresh clone and on the CPU, as in the
continuous integration (`.github/workflows/tests.yml`, Ubuntu, CPU build of torch).

## License
MIT (see `LICENSE`), for the code and the result tables in `replication/`. The price data are not covered: they
are downloaded from Yahoo Finance under its terms of use.
