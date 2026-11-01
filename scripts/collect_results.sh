#!/usr/bin/env bash
# Copy the small result files (tables of the evaluation, tuning choices, evaluation portfolios) from the results*
# directories, which are not under version control, to replication/, which is. Not copied: the forecasts
# (results*/*/h*/preds, about 16 GB), the training histories (info/), targets and origins, and the per-subset
# minimum-variance records (minvar_h5.csv, minvar_check*.csv except the summary). replication/ is replaced only
# when the tables folders of all three result directories exist.
set -eo pipefail
cd "$(dirname "$0")/.."
for R in results results_stocks results_stocks54; do
    [ -d "$R/main/tables" ] || { echo "$R/main/tables not found: replication/ not changed"; exit 1; }
done
[ -d results/heldout/tables ] && [ -d results_stocks/heldout/tables ] || { echo "held-out tables missing: replication/ not changed"; exit 1; }
TMP=$(mktemp -d replication.tmp.XXXX)
trap 'rm -rf "$TMP"' EXIT
for R in results results_stocks results_stocks54; do
    for E in main heldout; do
        [ -d "$R/$E" ] || continue
        mkdir -p "$TMP/$R/$E"
        cp "$R/$E"/*.json "$TMP/$R/$E/"
        if [ -d "$R/$E/tables" ]; then
            mkdir -p "$TMP/$R/$E/tables"
            find "$R/$E/tables" -maxdepth 1 -name '*.csv' -size +2c ! -name 'minvar_h5.csv' ! -name 'minvar_check.csv' \
                ! -name 'minvar_check_20*.csv' -exec cp {} "$TMP/$R/$E/tables/" \;
        fi
    done
    for D in tuning descriptives; do
        [ -d "$R/$D" ] || continue
        mkdir -p "$TMP/$R/$D"
        find "$R/$D" -maxdepth 1 \( -name '*.csv' -o -name '*.json' -o -name '*.jsonl' -o -name '*.txt' \) \
            -exec cp {} "$TMP/$R/$D/" \;
    done
done
rm -rf replication
mv "$TMP" replication
trap - EXIT
du -sh replication
