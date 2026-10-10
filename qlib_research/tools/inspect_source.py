# -*- coding: utf-8 -*-
"""Fast integrity check of the MNQ 1-min continuous-contract CSV."""
import sys

import pandas as pd

SRC = r"D:\BaiduNetdiskDownload\@MNQ#C_1min_20261003.csv"

df = pd.read_csv(SRC, usecols=["DateTime", "Symbol", "Open", "High", "Low", "Close", "Inc Volume", "Total Volume"])
print("rows:", len(df))
print("symbols:", df["Symbol"].unique().tolist())
dt = pd.to_datetime(df["DateTime"])
print("range:", dt.min(), "->", dt.max())
print("dtypes:\n", df.dtypes)
print("NaNs per col:\n", df.isna().sum())
print("dup timestamps:", int(df["DateTime"].duplicated().sum()))
print("monotonic increasing:", dt.is_monotonic_increasing)
zero_vol = int((df["Inc Volume"] == 0).sum())
print("zero Inc Volume rows:", zero_vol)

# bad OHLC sanity
bad = (
    (df["High"] < df["Low"])
    | (df["High"] < df["Open"])
    | (df["High"] < df["Close"])
    | (df["Low"] > df["Open"])
    | (df["Low"] > df["Close"])
)
print("OHLC-inconsistent rows:", int(bad.sum()))

span_days = (dt.max() - dt.min()).days
print(f"span_days={span_days}  avg_bars/day={len(df)/span_days:.1f}")
print("bars per weekday:\n", dt.dt.dayofweek.value_counts().sort_index())
print("sample head:\n", df.head(2).to_string())
print("sample tail:\n", df.tail(2).to_string())
