"""Cached qlib bar loading for the MNQ dataset.

Only this module talks to qlib. Bars are cached to ``tf_mnq/cache/{freq}.parquet``
on first load; subsequent calls read the cache. The returned frame uses lowercase
OHLCV columns, a tz-naive sorted DatetimeIndex and is NaN-checked.

NOTE: ``load_bars`` may read the *full* series into the cache; every caller is
responsible for slicing to IN-SAMPLE (<= 2023-12-31) before using it as a
feature/label source.
"""

from __future__ import annotations

import os

import pandas as pd

from qlib.constant import REG_CN

# qlib bin dir: env override, else <repo>/qlib_research/data/mnq (portable).
_DATA_URI = os.environ.get("MNQ_QLIB_URI") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "mnq"
)
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
_INITED = False

_COLUMNS = ["open", "high", "low", "close", "volume"]


def _ensure_init() -> None:
    global _INITED
    if _INITED:
        return
    import qlib

    qlib.init(provider_uri=_DATA_URI, region=REG_CN, logging_level=40)
    _INITED = True


def _cache_path(freq: str) -> str:
    return os.path.join(_CACHE_DIR, f"{freq}.parquet")


def _fetch_full(freq: str) -> pd.DataFrame:
    """Query qlib once for the full series (all dates)."""
    _ensure_init()
    from qlib.data import D

    df = D.features(["MNQ"], ["$open", "$high", "$low", "$close", "$volume"], freq=freq)
    df = df.droplevel("instrument")
    df.columns = [str(c).lstrip("$").lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df = df[_COLUMNS].sort_index()
    return df


def load_bars(freq: str, start=None, end=None) -> pd.DataFrame:
    """Return OHLCV bars for ``freq`` sliced to ``[start, end]`` (inclusive).

    Parameters
    ----------
    freq : {"1min","5min","15min","60min","day"}
    start, end : optional date-likes (inclusive bounds on the DatetimeIndex)
    """
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = _cache_path(freq)
    if os.path.exists(path):
        df = pd.read_parquet(path)
    else:
        df = _fetch_full(freq)
        df.to_parquet(path)

    df = df[_COLUMNS].sort_index().copy()
    if start is not None:
        df = df.loc[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df.loc[df.index <= pd.Timestamp(end)]
    df = df.sort_index().copy()

    if df.isna().any().any():
        bad = df.columns[df.isna().any()].tolist()
        raise ValueError(f"NaNs found in {freq} bars for columns {bad}")
    if df.empty:
        raise ValueError(f"no bars for freq={freq} in [{start}, {end}]")
    return df
