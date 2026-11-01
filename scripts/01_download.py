"""Download daily OHLCV (dividend/split adjusted) for the current universe (PVOL_UNIVERSE) from Yahoo Finance.

An existing download is kept (Yahoo revises adjusted prices, so a new download can differ from the one behind stored
forecasts); --force downloads again. The checksums of the files used for the paper are in README.md.
"""
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pvol.config import RAW, SAMPLE_END, SAMPLE_START, SMOKE  # noqa: E402
from pvol.data import simulate_prices  # noqa: E402
from pvol.universe import UNIVERSE  # noqa: E402

RAW.mkdir(parents=True, exist_ok=True)


def main():
    if SMOKE:  # simulated prices of the same tickers
        simulate_prices(list(UNIVERSE), SAMPLE_START, SAMPLE_END).to_parquet(RAW / "ohlcv_adjusted.parquet")
        return
    if (RAW / "ohlcv_adjusted.parquet").exists() and "--force" not in sys.argv:
        print(f"{RAW / 'ohlcv_adjusted.parquet'} exists; kept (--force to download again)")
        return
    tickers = list(UNIVERSE)
    df = yf.download(
        tickers,
        start="2000-01-01",
        end="2026-09-19",
        auto_adjust=True,
        actions=False,
        group_by="column",
        progress=False,
        threads=True,
    )
    df.to_parquet(RAW / "ohlcv_adjusted.parquet")
    close = df["Close"]
    summary = pd.DataFrame(
        {
            "asset_class": pd.Series(UNIVERSE),
            "first": close.apply(lambda s: s.first_valid_index()),
            "last": close.apply(lambda s: s.last_valid_index()),
            "n_obs": close.notna().sum(),
        }
    ).sort_values("first")
    summary.to_csv(RAW / "coverage.csv")
    print(summary.to_string())
    print("shape", df.shape)


if __name__ == "__main__":
    main()
