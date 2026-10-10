# -*- coding: utf-8 -*-
"""Convert local Freqtrade/futures OHLCV sources into qlib ``dump_bin`` staging CSVs.

Produces, under ``D:\\qlib_research\\data_stage\\``:
    day\\MNQ.csv, day\\MES.csv, day\\BTCUSDT.csv, day\\ETHUSDT.csv
    1h\\MNQ.csv,  1h\\MES.csv,  1h\\BTCUSDT.csv,  1h\\ETHUSDT.csv

Each staging CSV has lowercase columns: date,open,high,low,close,volume
(the format expected by ``tools/dump_bin.py``). Run ``dump_bin.py`` afterwards.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:\freqtrade")
RAW = ROOT / "mnq_mes" / "data" / "raw"
FT = ROOT / "mnq_mes" / "data" / "freqtrade"
STAGE = Path(r"D:\qlib_research\data_stage")

COLS = ["date", "open", "high", "low", "close", "volume"]


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.lower() for c in df.columns})
    df = df[COLS].copy()
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df = df.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")
    for c in COLS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)


def _write(df: pd.DataFrame, freq: str, code: str) -> None:
    out_dir = STAGE / freq
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{code}.csv"
    df.to_csv(path, index=False)
    print(f"  {freq:>3}  {code:<8} {len(df):>6} rows  {df['date'].min().date()}..{df['date'].max().date()}  -> {path}")


def main() -> None:
    print("Futures (yfinance raw CSV):")
    for freq, fname, code in [
        ("day", "MNQ=F.csv", "MNQ"),
        ("day", "MES=F.csv", "MES"),
        ("1h", "MNQ=F-1h.csv", "MNQ"),
        ("1h", "MES=F-1h.csv", "MES"),
    ]:
        src = RAW / fname
        if src.exists():
            _write(_norm(pd.read_csv(src)), freq, code)
        else:
            print(f"  skip missing {src}")

    print("Crypto (freqtrade feather, OKX USDT swaps):")
    okx = FT / "okx" / "futures"
    for freq, suffix, code in [
        ("day", "1d", "BTCUSDT"),
        ("day", "1d", "ETHUSDT"),
        ("1h", "1h", "BTCUSDT"),
        ("1h", "1h", "ETHUSDT"),
    ]:
        src = okx / f"{code[:3]}_USDT_USDT-{suffix}-futures.feather"
        if src.exists():
            _write(_norm(pd.read_feather(src)), freq, code)
        else:
            print(f"  skip missing {src}")


if __name__ == "__main__":
    main()
