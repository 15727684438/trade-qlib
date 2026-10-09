"""Convert raw Yahoo MES/MNQ CSVs into freqtrade data files (spot + futures).

Pipeline:
  1. Daily: reads data/raw/<SYM>.csv (Date = ISO market date) and writes
     data/freqtrade/binance/<PAIR>-1d.feather with freqtrade schema
     [date, open, high, low, close, volume] (date = tz-aware UTC midnight,
     OHLCV float, lz4 compression).
  2. Intraday: fetches ~last 60 trading days of 1h OHLCV for both symbols
     (yfinance first, direct Yahoo chart API as fallback — reusing the logic
     in download_data.py), saves data/raw/<SYM>-1h.csv, normalizes timestamps
     to UTC, dedupes + sorts, and writes <PAIR>-1h.feather (same schema).
  3. Futures mode: for every OHLCV frame (1d always, 1h when available) also
     writes into data/freqtrade/binance/futures/:
       - <FUT>-<tf>-futures.feather      = same OHLCV (main candles)
       - <FUT>-<tf>-mark.feather         = same OHLCV (mark ~= futures price)
       - <FUT>-<tf>-funding_rate.feather = date + open=0.0 + rest 0.0
         (CME micro futures have NO funding fee)
     with <FUT> = "MNQ_USD_USD" for whitelist pair "MNQ/USD:USD" (freqtrade
     futures naming: "/" and ":" both become "_").

If the 1h fetch fails entirely, a clear warning is printed and the daily
files (spot + futures) are still produced; the script still exits 0.

Usage:
    & "D:\\freqtrade\\.venv\\Scripts\\python.exe" mnq_mes/scripts/convert_to_freqtrade.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import download_data as dd  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
FT_DIR = ROOT / "data" / "freqtrade" / "binance"
FUT_DIR = FT_DIR / "futures"

SYMBOLS = ["MES=F", "MNQ=F"]
FT_NAMES = {"MES=F": "MES_USD", "MNQ=F": "MNQ_USD"}           # spot filenames
FUT_PAIRS = {"MES=F": "MES/USD:USD", "MNQ=F": "MNQ/USD:USD"}  # futures whitelist pairs
OHLCV_COLS = ["open", "high", "low", "close", "volume"]
FEATHER_COLS = ["date", *OHLCV_COLS]
H1_FETCH_DAYS = 90  # calendar days back -> ~60 trading days
FUT_KINDS = ("futures", "mark", "funding_rate")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def fut_basename(sym: str) -> str:
    """Futures filename stem: pair 'MNQ/USD:USD' -> 'MNQ_USD_USD'."""
    return FUT_PAIRS[sym].replace("/", "_").replace(":", "_")


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize to freqtrade schema: [date(UTC), open..volume(float)], sorted."""
    out = df[FEATHER_COLS].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_convert("UTC")
    for col in OHLCV_COLS:
        out[col] = out[col].astype(float)
    return out.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)


def print_summary(path: Path, df: pd.DataFrame) -> None:
    first = df["date"].iloc[0]
    last = df["date"].iloc[-1]
    print(
        f"{path.name}: {len(df)} rows, first={first:%Y-%m-%d %H:%M:%S} UTC, "
        f"last={last:%Y-%m-%d %H:%M:%S} UTC, {path.stat().st_size} bytes"
    )


def build_ohlcv_df(sym: str, tf: str) -> pd.DataFrame:
    """Rebuild the normalized OHLCV frame from the raw CSV (daily or 1h)."""
    if tf == "1d":
        src = pd.read_csv(RAW_DIR / f"{sym}.csv")
        date_col = pd.to_datetime(src["Date"]).dt.tz_localize("UTC")  # UTC midnight
    else:
        src = pd.read_csv(RAW_DIR / f"{sym}-1h.csv")
        date_col = pd.to_datetime(src["Date"], utc=True)  # handles DST offset mixes
    return pd.DataFrame(
        {
            "date": date_col,
            "open": src["Open"].astype(float),
            "high": src["High"].astype(float),
            "low": src["Low"].astype(float),
            "close": src["Close"].astype(float),
            "volume": src["Volume"].astype(float),
        }
    )


# --------------------------------------------------------------------------- #
# Feather writing
# --------------------------------------------------------------------------- #
def write_feather(df: pd.DataFrame, sym: str, timeframe: str) -> Path:
    """Write the spot feather (freqtrade schema, lz4)."""
    out = normalize_ohlcv(df)
    path = FT_DIR / f"{FT_NAMES[sym]}-{timeframe}.feather"
    out.to_feather(path, compression="lz4")
    print_summary(path, out)
    return path


def write_futures_trio(df: pd.DataFrame, sym: str, timeframe: str, emit_funding: bool = True) -> None:
    """Write futures/mark/funding_rate feathers for one pair+timeframe.

    ``emit_funding=False`` skips the funding_rate file. It must be used for
    intraday timeframes when the configured ``funding_fee_timeframe`` is larger
    (e.g. 1d): freqtrade's ``fix_funding_fee_timeframe`` migration would
    otherwise DELETE the 1d funding file and rename the 1h one over it.
    """
    ohlcv = normalize_ohlcv(df)
    for kind in ("futures", "mark"):
        path = FUT_DIR / f"{fut_basename(sym)}-{timeframe}-{kind}.feather"
        ohlcv.to_feather(path, compression="lz4")
        print_summary(path, ohlcv)
    if not emit_funding:
        return
    # funding_rate: only the date column carries data; open=0.0 (no funding on
    # CME micros), high/low/close/volume zeroed as well.
    fund = ohlcv[["date"]].copy()
    for col in OHLCV_COLS:
        fund[col] = 0.0
    path = FUT_DIR / f"{fut_basename(sym)}-{timeframe}-funding_rate.feather"
    fund.to_feather(path, compression="lz4")
    print_summary(path, fund)


# --------------------------------------------------------------------------- #
# Daily
# --------------------------------------------------------------------------- #
def convert_daily(sym: str) -> Path:
    """Build the daily spot feather + futures trio from the raw daily CSV."""
    df = build_ohlcv_df(sym, "1d")
    write_feather(df, sym, "1d")
    write_futures_trio(df, sym, "1d")
    return FT_DIR / f"{FT_NAMES[sym]}-1d.feather"


# --------------------------------------------------------------------------- #
# 1h fetch + conversion
# --------------------------------------------------------------------------- #
def fetch_1h_frames(start: date) -> tuple[dict[str, pd.DataFrame] | None, str | None]:
    """Fetch 1h frames for both symbols (yfinance first, direct API fallback).

    Returns (frames, method); (None, None) if both methods fail.
    """
    try:
        if dd.ensure_yfinance():
            frames = dd._download_with_retries(
                lambda: dd.download_yfinance(start, interval="1h"), "yfinance-1h"
            )
            return frames, "yfinance"
        raise RuntimeError("yfinance import failed after install attempt")
    except Exception as exc:  # noqa: BLE001
        print(f"[convert] yfinance 1h failed ({exc}); trying direct chart API...", file=sys.stderr)
        try:
            frames = {
                sym: dd.download_direct(sym, start, attempts=3, base_sleep=30, interval="1h")
                for sym in SYMBOLS
            }
            return frames, "direct chart API"
        except Exception as exc2:  # noqa: BLE001
            print(
                f"[convert] WARNING: 1h fetch failed entirely ({exc2}). "
                f"Skipping 1h files; daily files are still produced.",
                file=sys.stderr,
            )
            return None, None


def convert_1h_csv_to_feather(sym: str) -> Path:
    """Read data/raw/<SYM>-1h.csv, normalize to UTC, write spot + futures 1h.

    Funding for 1h is intentionally skipped: the MNQ/MES backtests use
    ``funding_fee_timeframe: 1d``, and a 1h funding file would be renamed over
    the 1d file by freqtrade's ``fix_funding_fee_timeframe`` migration.
    """
    df = build_ohlcv_df(sym, "1h")
    write_feather(df, sym, "1h")
    write_futures_trio(df, sym, "1h", emit_funding=False)
    return FT_DIR / f"{FT_NAMES[sym]}-1h.feather"


def process_1h(frames: dict[str, pd.DataFrame]) -> None:
    """Save 1h CSVs (ISO with offset) then convert each to spot + futures."""
    for sym in SYMBOLS:
        df = frames[sym]
        out = df.copy()
        out["Date"] = pd.to_datetime(out["Date"])
        # trim to the last ~60 trading days (calendar dates in the data's tz)
        local_dates = sorted(out["Date"].dt.date.unique())
        if len(local_dates) > 60:
            cutoff = local_dates[-60]
            out = out[out["Date"].dt.date >= cutoff].reset_index(drop=True)
        # prefer ISO timestamps with timezone offset (pandas str keeps e.g.
        # "2026-08-16 18:00:00-04:00"); utc=True on read-back handles DST mixes
        out["Date"] = out["Date"].astype(str)
        out = out[dd.COLUMNS]
        csv_path = RAW_DIR / f"{sym}-1h.csv"
        out.to_csv(csv_path, index=False)
        print(
            f"{sym}-1h.csv: {len(out)} rows, first={out['Date'].iloc[0]}, "
            f"last={out['Date'].iloc[-1]} -> {csv_path}"
        )
        convert_1h_csv_to_feather(sym)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="Convert MES/MNQ data to freqtrade feathers.")
    parser.add_argument(
        "--skip-1h", action="store_true", help="skip the 1h fetch (re-emit 1h from CSV if present)"
    )
    args = parser.parse_args()

    FT_DIR.mkdir(parents=True, exist_ok=True)
    FUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # ---- daily ---- #
    for sym in SYMBOLS:
        daily_csv = RAW_DIR / f"{sym}.csv"
        if not daily_csv.exists():
            print(f"[convert] ERROR: {daily_csv} not found — run download_data.py first", file=sys.stderr)
            return 1
        convert_daily(sym)

    # ---- 1h (robust: failure -> warning, still exit 0) ---- #
    if not args.skip_1h:
        proxy = dd.apply_proxy()
        if proxy:
            print(f"[convert] using proxy: {proxy}", file=sys.stderr)
        start = datetime.now(timezone.utc).date() - timedelta(days=H1_FETCH_DAYS)
        frames, method = fetch_1h_frames(start)
        if frames is None:
            print("[convert] 1h data skipped (see warning above); daily feathers written.", file=sys.stderr)
        else:
            process_1h(frames)
            print(f"1h fetch method: {method}")
    else:
        # no fetch: re-emit 1h from the CSV if it exists (keeps futures 1h files)
        for sym in SYMBOLS:
            if (RAW_DIR / f"{sym}-1h.csv").exists():
                convert_1h_csv_to_feather(sym)
        print("[convert] --skip-1h: 1h fetch skipped; 1h re-emitted from CSV where present.")

    print(f"spot feather dir: {FT_DIR}")
    print(f"futures feather dir: {FUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
