# -*- coding: utf-8 -*-
"""Ingest the @MNQ#C 1-minute futures CSV into qlib binary format.

Reads the raw 1-minute CSV, builds lowercase OHLCV frames, derives
1min/5min/15min/60min/day bars, writes one staging CSV per freq under
``D:\\qlib_research\\data_stage\\<freq>\\MNQ.csv`` and then runs the official
``tools/dump_bin.py dump_all`` for each freq into ``D:\\qlib_research\\data\\mnq``.

Run with the dedicated venv python:
    D:\\qlib_research\\.venv\\Scripts\\python.exe D:\\qlib_research\\tools\\ingest_mnq_1min.py
"""
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

# Portable paths: env override first, else repo-relative defaults.
#   MNQ_RAW_CSV    raw 1-min CSV (default: Baidu Netdisk download path)
#   MNQ_STAGE_DIR  staging CSV dir
#   MNQ_QLIB_DIR   qlib bin output dir
_HERE = Path(__file__).resolve().parent          # .../qlib_research/tools
_ROOT = _HERE.parent                             # .../qlib_research
SRC = Path(os.environ.get("MNQ_RAW_CSV", r"D:\BaiduNetdiskDownload\@MNQ#C_1min_20261003.csv"))
STAGE = Path(os.environ.get("MNQ_STAGE_DIR", _ROOT / "data_stage"))
QLIB_DIR = Path(os.environ.get("MNQ_QLIB_DIR", _ROOT / "data" / "mnq"))
TOOLS = _HERE
DUMP_BIN = TOOLS / "dump_bin.py"

CODE = "MNQ"
MIN_FREQS = ["1min", "5min", "15min", "60min"]
ALL_FREQS = MIN_FREQS + ["day"]

AGG = dict(open=("open", "first"), high=("high", "max"), low=("low", "min"),
           close=("close", "last"), volume=("volume", "sum"))


def load_base() -> pd.DataFrame:
    """Load the raw CSV and normalise to lowercase tz-naive OHLCV."""
    df = pd.read_csv(SRC, usecols=["DateTime", "Open", "High", "Low", "Close", "Inc Volume"])
    df = df.rename(columns={
        "DateTime": "date", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Inc Volume": "volume",
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df.reset_index(drop=True)


def resample_minute(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    grouped = df.groupby(df["date"].dt.floor(freq), sort=True)
    return grouped.agg(**AGG).reset_index()


def resample_day(df: pd.DataFrame) -> pd.DataFrame:
    """CME Globex session date: (date + 6h) normalised to midnight."""
    session = (df["date"] + pd.Timedelta(hours=6)).dt.normalize()
    grouped = df.groupby(session, sort=True)
    out = grouped.agg(**AGG).reset_index()
    out = out.rename(columns={"index": "date"})
    return out


def write_stage(df: pd.DataFrame, freq: str) -> Path:
    out_dir = STAGE / freq
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{CODE}.csv"
    out = df.copy()
    if freq == "day":
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    else:
        out["date"] = out["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(path, index=False)
    return path


def dump_freq(freq: str) -> None:
    # NOTE: this dump_bin.py exposes the source dir as `data_path` (positional),
    # not `csv_path` as stated in the task brief.
    cmd = [sys.executable, str(DUMP_BIN), "dump_all",
           "--data_path", str(STAGE / freq), "--qlib_dir", str(QLIB_DIR),
           "--freq", freq, "--date_field_name", "date"]
    print("\n[run] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    print(f"Loading {SRC} ...", flush=True)
    base = load_base()
    print(f"Base rows: {len(base)}")

    frames = {}
    for freq in MIN_FREQS:
        if freq == "1min":
            frames[freq] = base.copy()
        else:
            frames[freq] = resample_minute(base, freq)
    frames["day"] = resample_day(base)

    for freq in ALL_FREQS:
        df = frames[freq]
        path = write_stage(df, freq)
        dmin, dmax = df["date"].min(), df["date"].max()
        print(f"  {freq:>5}: {len(df):>8} rows  {dmin} .. {dmax}  -> {path}")

    print("\nRow counts:", {f: len(frames[f]) for f in ALL_FREQS})

    for freq in ALL_FREQS:
        dump_freq(freq)

    print("\nDump complete. qlib_dir =", QLIB_DIR)


if __name__ == "__main__":
    main()
