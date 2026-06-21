"""
download_data.py — build the price dataset this project expects (df_2010.csv).

The original research used Bloomberg data, which is licensed and is therefore NOT
redistributed in this repository. This script reproduces a comparable, free, and
redistributable dataset from Yahoo Finance so the backtest is runnable end-to-end.

It downloads daily split/dividend-adjusted OHLCV for the same ~98 large-cap U.S.
equities, over 2010-01-01 .. 2024-12-31, and writes `df_2010.csv` in the exact
schema the backtest loader expects:

    date, PX_OPEN, PX_HIGH, PX_LOW, PX_LAST, VOLUME, ticker

Usage:
    pip install -r requirements.txt        # (includes yfinance)
    python download_data.py
    python momentum_ml_framework.py        # then run the backtest

Notes:
- Yahoo occasionally rate-limits bulk requests; this script downloads in small
  batches and retries with exponential backoff.
- Adjusted prices and survivorship differ from the original Bloomberg vendor data,
  so headline numbers will differ slightly. The methodology is identical.
"""

import sys
import time

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    sys.exit("yfinance is required. Install it with: pip install yfinance")

START = "2010-01-01"
END = "2025-01-01"          # yfinance end is exclusive; this includes 2024-12-31
OUT = "df_2010.csv"
BATCH_SIZE = 15
MAX_RETRIES = 5

# Same large-cap U.S. equity universe used in the original study (98 names).
TICKERS = [
    "AA", "AAPL", "ABT", "ADBE", "ADP", "AEP", "AMD", "AMGN", "AMT", "AMZN",
    "APA", "APD", "BA", "BDX", "BKNG", "BMY", "CAT", "CL", "CMCSA", "COP",
    "COST", "CRM", "CSCO", "CVX", "DE", "DIS", "DUK", "DVN", "ECL", "EOG",
    "EXC", "F", "FDX", "GD", "GE", "GILD", "GIS", "GOOGL", "HAL", "HD",
    "HON", "HPQ", "IBM", "INTC", "INTU", "JNJ", "KLAC", "KMB", "KO", "KR",
    "LLY", "LMT", "LOW", "MCD", "MDT", "MLM", "MMM", "MO", "MRK", "MSFT",
    "MU", "NEE", "NFLX", "NKE", "NOC", "NUE", "NVDA", "O", "OMC", "ORCL",
    "OXY", "PEP", "PFE", "PG", "PLD", "PPG", "PSA", "QCOM", "RCL", "SBUX",
    "SHW", "SLB", "SO", "SPG", "SYK", "T", "TGT", "TM", "TMO", "TXN",
    "UNP", "UPS", "VLO", "VMC", "VZ", "WM", "WMT", "XOM",
]


def _download_batch(tickers):
    """Download one batch of tickers, retrying on transient/rate-limit errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            data = yf.download(
                tickers,
                start=START,
                end=END,
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="column",
            )
            if data is not None and not data.empty:
                return data
            print(f"  empty response, retry {attempt}/{MAX_RETRIES}...")
        except Exception as exc:  # noqa: BLE001 - yfinance raises a variety of errors
            print(f"  error ({exc.__class__.__name__}), retry {attempt}/{MAX_RETRIES}...")
        time.sleep(min(60, 2 ** attempt))
    return None


def _to_long(data, tickers):
    """Reshape a yfinance frame into long rows: date, ticker, OHLCV."""
    frames = []
    single = len(tickers) == 1
    for ticker in tickers:
        try:
            sub = data if single else data.xs(ticker, axis=1, level=1)
        except KeyError:
            continue
        sub = sub.dropna(how="all")
        if sub.empty:
            continue
        out = pd.DataFrame({
            "date": sub.index,
            "PX_OPEN": sub.get("Open"),
            "PX_HIGH": sub.get("High"),
            "PX_LOW": sub.get("Low"),
            "PX_LAST": sub.get("Close"),
            "VOLUME": sub.get("Volume"),
            "ticker": ticker,
        })
        frames.append(out)
    return frames


def main():
    all_frames = []
    for i in range(0, len(TICKERS), BATCH_SIZE):
        batch = TICKERS[i:i + BATCH_SIZE]
        print(f"Downloading {i + 1}-{i + len(batch)} of {len(TICKERS)}: {', '.join(batch)}")
        data = _download_batch(batch)
        if data is None:
            print(f"  !! giving up on this batch after {MAX_RETRIES} retries")
            continue
        all_frames.extend(_to_long(data, batch))
        time.sleep(1)  # be polite to Yahoo between batches

    if not all_frames:
        sys.exit("No data downloaded — Yahoo may be rate-limiting. Try again later.")

    df = pd.concat(all_frames, ignore_index=True)
    df = df.dropna(subset=["PX_LAST"]).sort_values(["ticker", "date"])
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df[["date", "PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "VOLUME", "ticker"]]
    df.to_csv(OUT, index=False)

    n_tickers = df["ticker"].nunique()
    print(f"\nWrote {OUT}: {len(df):,} rows, {n_tickers} tickers, "
          f"{df['date'].min()} .. {df['date'].max()}")
    if n_tickers < len(TICKERS):
        print(f"Note: {len(TICKERS) - n_tickers} tickers returned no data "
              f"(delisted or rate-limited). Re-run to fill gaps if needed.")


if __name__ == "__main__":
    main()
