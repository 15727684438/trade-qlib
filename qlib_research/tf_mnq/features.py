"""Causal feature/gate layer.

Every series here at index ``t`` may only depend on bars ``<= t`` (a single
forward pass, no future peeking). Wilder ATR/ADX are ported from the canonical
reference ``D:\\freqtrade\\mnq_mes\\src\\mrind.py`` (TA-Lib conventions: wilder
smoothing seeded with a simple average of the first ``n`` values) because
``talib`` is not installed in this venv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _as_series(x, index=None) -> pd.Series:
    if isinstance(x, pd.Series):
        s = x.astype(float)
    else:
        s = pd.Series(np.asarray(x, dtype=float))
    if index is not None:
        s.index = index
    return s


def _wilder_smooth(values: pd.Series, n: int) -> pd.Series:
    """Wilder smoothing with a simple-average seed (TA-Lib ``RMA`` convention).

    First valid output sits at ``first_valid + n - 1`` and equals the mean of the
    first ``n`` valid inputs; afterwards ``y = (y_prev*(n-1) + x) / n``.
    """
    x = values.to_numpy(dtype=float)
    out = np.full(len(x), np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(x))
    if valid.size < n:
        return pd.Series(out, index=values.index)
    f = int(valid[0])
    if valid[n - 1] != f + n - 1:  # need n contiguous valid values
        return pd.Series(out, index=values.index)
    prev = float(np.mean(x[f : f + n]))
    out[f + n - 1] = prev
    for i in range(f + n, len(x)):
        xi = x[i] if np.isfinite(x[i]) else prev
        prev = (prev * (n - 1) + xi) / n
        out[i] = prev
    return pd.Series(out, index=values.index)


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev).abs(), (low - prev).abs()], axis=1
    ).max(axis=1)
    # first bar has no previous close: fall back to high-low (TA-Lib behaviour)
    tr.iloc[0] = float(high.iloc[0] - low.iloc[0])
    return tr


def atr_wilder(high, low, close, n: int = 14) -> pd.Series:
    """Wilder Average True Range (TA-Lib ``ATR``)."""
    high, low, close = _as_series(high), _as_series(low), _as_series(close)
    return _wilder_smooth(_true_range(high, low, close), n)


def adx_wilder(high, low, close, n: int = 14) -> pd.Series:
    """Wilder Average Directional Index (TA-Lib ``ADX``)."""
    high, low, close = _as_series(high), _as_series(low), _as_series(close)
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0).fillna(0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0).fillna(0.0)

    tr_s = _wilder_smooth(_true_range(high, low, close), n)
    plus_s = _wilder_smooth(plus_dm, n)
    minus_s = _wilder_smooth(minus_dm, n)

    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * plus_s / tr_s
        minus_di = 100.0 * minus_s / tr_s
        denom = plus_di + minus_di
        dx = 100.0 * (plus_di - minus_di).abs() / denom
    return _wilder_smooth(dx, n)


def efficiency_ratio(close, n: int = 20) -> pd.Series:
    """Kaufman efficiency ratio: |net move| / sum(|bar moves|) over ``n`` bars."""
    close = _as_series(close)
    change = (close - close.shift(n)).abs()
    volatility = close.diff().abs().rolling(n, min_periods=n).sum()
    return change / volatility


def variance_ratio(close, q: int = 12, window: int = 250) -> pd.Series:
    """Lo-MacKinlay variance ratio VR = Var(q-bar ret) / (q * Var(1-bar ret))."""
    close = _as_series(close)
    r1 = np.log(close).diff()
    rq = np.log(close / close.shift(q))
    num = rq.rolling(window, min_periods=window).var()
    den = q * r1.rolling(window, min_periods=window).var()
    return num / den


def ema(close, n: int) -> pd.Series:
    """EMA seeded with an SMA of the first ``n`` values (TA-Lib convention)."""
    close = _as_series(close)
    x = close.to_numpy(dtype=float)
    out = np.full(len(x), np.nan, dtype=float)
    if len(x) < n:
        return pd.Series(out, index=close.index)
    alpha = 2.0 / (n + 1.0)
    out[n - 1] = float(np.mean(x[:n]))
    for i in range(n, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return pd.Series(out, index=close.index)


def direction(close, ema_len: int = 50, slope_lb: int = 3) -> pd.Series:
    """Causal trend direction: +1/-1 when price and EMA slope agree, else carry."""
    close = _as_series(close)
    e = ema(close, ema_len)
    slope = e.diff(slope_lb)
    raw = np.where((close > e) & (slope > 0), 1, np.where((close < e) & (slope < 0), -1, 0))
    out = np.zeros(len(close), dtype=int)
    prev = 0
    for i in range(len(close)):
        if raw[i] != 0:
            prev = int(raw[i])
        out[i] = prev
    return pd.Series(out, index=close.index)


def gate_state(er, adx, vr, dir, on=(0.30, 20.0, 1.05), off=(0.22, 17.0, 0.98)) -> pd.Series:
    """Hysteretic trend/range gate.

    ``ON`` -> 1 when all ``on`` thresholds hold and direction is known;
    ``OFF`` -> 0 when any ``off`` threshold is breached; otherwise keep previous.
    State starts at 0. Single causal pass.
    """
    er = _as_series(er)
    adx = _as_series(adx)
    vr = _as_series(vr)
    dir = _as_series(dir)
    on0, on1, on2 = on
    off0, off1, off2 = off
    out = np.zeros(len(er), dtype=int)
    state = 0
    for i in range(len(er)):
        e, a, v, d = float(er.iloc[i]), float(adx.iloc[i]), float(vr.iloc[i]), float(dir.iloc[i])
        if d != 0 and e >= on0 and a >= on1 and v >= on2:
            state = 1
        elif e < off0 or a < off1 or v < off2:
            state = 0
        out[i] = state
    return pd.Series(out, index=er.index)


def compute_features(df60: pd.DataFrame) -> pd.DataFrame:
    """Return causal features [atr, adx, er, vr, ema50, dir, gate] for 60min bars."""
    high, low, close = df60["high"], df60["low"], df60["close"]
    out = pd.DataFrame(index=df60.index)
    out["atr"] = atr_wilder(high, low, close, n=14)
    out["adx"] = adx_wilder(high, low, close, n=14)
    out["er"] = efficiency_ratio(close, n=20)
    out["vr"] = variance_ratio(close, q=12, window=250)
    out["ema50"] = ema(close, 50)
    out["dir"] = direction(close, ema_len=50, slope_lb=3)
    out["gate"] = gate_state(out["er"], out["adx"], out["vr"], out["dir"])
    return out


# --------------------------------------------------------------------------- #
# M2 additions: structure pivots, pullback-resumption trigger, ATR percentile
# --------------------------------------------------------------------------- #
def pivot_low(low, n: int = 2):
    """Fractal pivot low with unique-minimum requirement.

    ``low[i]`` is a pivot iff it equals ``min(low[i-n .. i+n])`` and is the
    UNIQUE minimum in that window (ties → no pivot). Confirmation happens only
    after the ``n`` right-side bars close, so the returned boolean flags the
    CONFIRMATION bar ``i+n`` while ``pivot_low_price`` is aligned to the pivot
    bar ``i`` (price there, NaN elsewhere).

    Returns
    -------
    (pivot_low_flag, pivot_low_price) : (pd.Series[bool], pd.Series[float])
    """
    low = _as_series(low)
    arr = low.to_numpy(dtype=float)
    m = len(arr)
    flag = np.zeros(m, dtype=bool)
    price = np.full(m, np.nan, dtype=float)
    for i in range(n, m - n):
        window = arr[i - n : i + n + 1]
        mn = window.min()
        if arr[i] == mn and int((window == mn).sum()) == 1:
            flag[i + n] = True
            price[i] = arr[i]
    return pd.Series(flag, index=low.index), pd.Series(price, index=low.index)


def pullback_resumption_trigger(close, high, low, atr, depth_mult: float = 1.0,
                                lookback: int = 10) -> pd.Series:
    """Causal pullback-resumption trigger.

    ``swing_high = high.rolling(lookback).max()`` (inclusive of the current bar,
    known at close t); ``armed = (swing_high - close) >= depth_mult*atr``; the
    trigger fires on ``armed & (close > high.shift(1))``. All inputs are known
    at the close of ``t`` — no forward reference.
    """
    close = _as_series(close)
    high = _as_series(high)
    atr = _as_series(atr)
    swing_high = high.rolling(lookback, min_periods=lookback).max()
    armed = (swing_high - close) >= depth_mult * atr
    return (armed & (close > high.shift(1))).fillna(False)


def atr_percentile(atr, window: int = 500, min_periods: int = 100) -> pd.Series:
    """Causal rolling percentile rank (0-100) of ATR within the trailing window."""
    atr = _as_series(atr)
    return atr.rolling(window, min_periods=min_periods).rank(pct=True) * 100.0
