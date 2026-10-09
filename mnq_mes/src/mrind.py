"""Market-regime indicator — canonical Python reference implementation.

Canonical contract: ``docs/regime-indicator-spec.md`` (§2 params, §3 computations,
§4 signals, §6 API/parity). This module is the reference; the TradingView Pine
port and the Freqtrade strategy must reproduce its formulas exactly.

Parity notes (spec §6)
----------------------
* TA-Lib ``ATR``/``ADX``/``RSI`` (Wilder smoothing) == Pine ``ta.atr``/``ta.adx``/``ta.rsi``.
* TA-Lib ``BBANDS`` (SMA middle band) uses the population stdev (ddof=0) == Pine ``ta.stdev``.
* TA-Lib ``EMA`` smoothing 2/(n+1) == Pine ``ta.ema``.
* Donchian/BB/EMA windows are inclusive of the current bar (TA-Lib/Pine convention);
  Donchian is computed with pandas ``rolling`` == Pine ``ta.highest``/``ta.lowest``.
* The first ``warmup_bars()`` rows (spec §6: max of all lookbacks) are forced to
  ``state=0`` (RANGE) and produce no signals. Rows where any core indicator is
  still NaN (e.g. ADX needs ~2x its period) are also forced to ``state=0``.

Exit columns (spec §4, per-bar guidance, no position tracking)
--------------------------------------------------------------
``exit_long`` is True on a bar when a hypothetical long would be exited, evaluated
independently per bar:
  * in RANGE:      close >= bb_mid                      (range-long mean-reversion done)
  * in TREND_UP:   close < ema_slow                     (trend-long anchor broken)
  * in TREND_DOWN: always True                          (no long should be held)
``exit_short`` is the exact mirror.

Determinism: pure functions, no module-level mutable state, inputs are never mutated
(always operate on a copy).

Notes:
* ``open`` is accepted for freqtrade schema compatibility but is not consumed by any formula.
* ``bb_hi``/``bb_lo`` are display-only outputs (Pine visuals); signals/exits use only ``bb_mid``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import talib

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
TREND_UP = 1
TREND_DOWN = -1
RANGE = 0

STATE_LABELS = {TREND_UP: "TREND_UP", TREND_DOWN: "TREND_DOWN", RANGE: "RANGE"}

SIGNAL_TAGS = (
    "RANGE_LONG",
    "TREND_LONG",
    "BRKOUT_LONG",
    "RANGE_SHORT",
    "TREND_SHORT",
    "BRKOUT_SHORT",
)
LONG_TAGS = frozenset({"RANGE_LONG", "TREND_LONG", "BRKOUT_LONG"})
SHORT_TAGS = frozenset({"RANGE_SHORT", "TREND_SHORT", "BRKOUT_SHORT"})

# Columns added by compute_regime (spec §6)
REGIME_COLS = (
    "atr",
    "adx",
    "ema_fast",
    "ema_slow",
    "don_hi",
    "don_lo",
    "bb_mid",
    "bb_hi",
    "bb_lo",
    "rsi",
    "span_atr",
    "trending",
    "up",
    "down",
    "state",
    "state_label",
)

# Columns added by signal_events (spec §6)
SIGNAL_COLS = ("don_hi_prev", "don_lo_prev", "sig", "dir", "exit_long", "exit_short")

INPUT_COLS = ("open", "high", "low", "close")


# --------------------------------------------------------------------------- #
# Parameters (spec §2 — field names and defaults are part of the contract)
# --------------------------------------------------------------------------- #
@dataclass
class RegimeParams:
    """Indicator parameters. Field names/defaults match spec §2 exactly."""

    atr_len: int = 14          # ATR period (Wilder)
    adx_len: int = 14          # ADX period (Wilder)
    adx_threshold: float = 20.0  # ADX >= threshold -> trending candidate
    ema_fast_len: int = 9      # fast EMA
    ema_slow_len: int = 21     # slow EMA (direction anchor)
    donch_len: int = 20        # Donchian channel length (range bounds)
    bb_len: int = 20           # Bollinger length (exit reference)
    bb_mult: float = 2.0       # BB std multiplier
    rsi_len: int = 14          # RSI length (Wilder)
    chan_atr_min: float = 1.0  # min span_atr for a trending regime


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def warmup_bars(p: RegimeParams = RegimeParams()) -> int:
    """Number of leading bars treated as warmup (spec §6)."""
    return max(
        p.atr_len,
        p.adx_len,
        p.donch_len,
        p.bb_len,
        p.rsi_len,
        p.ema_fast_len,
        p.ema_slow_len,
    )


def _resolve_cols(df: pd.DataFrame) -> dict:
    """Map lowercase logical names to the actual input columns (case-tolerant)."""
    cols = {}
    for lc in INPUT_COLS:
        if lc in df.columns:
            cols[lc] = lc
        elif lc.capitalize() in df.columns:
            cols[lc] = lc.capitalize()
        else:
            raise ValueError(
                f"input must contain a '{lc}' or '{lc.capitalize()}' column; "
                f"got columns: {list(df.columns)}"
            )
    return cols


# --------------------------------------------------------------------------- #
# Regime computation (spec §3)
# --------------------------------------------------------------------------- #
def compute_regime(df: pd.DataFrame, p: RegimeParams = RegimeParams()) -> pd.DataFrame:
    """Add regime columns to a copy of ``df`` (needs open/high/low/close).

    Returns a copy with the original input columns preserved plus: atr, adx,
    ema_fast, ema_slow, don_hi, don_lo, bb_mid, bb_hi, bb_lo, rsi, span_atr,
    trending, up, down, state (1/-1/0), state_label.
    """
    out = df.copy()
    c = _resolve_cols(out)
    high = out[c["high"]].to_numpy(dtype=float)
    low = out[c["low"]].to_numpy(dtype=float)
    close = out[c["close"]].to_numpy(dtype=float)

    # ---- indicators (TA-Lib, Wilder smoothing / population stdev) ---- #
    atr = talib.ATR(high, low, close, timeperiod=p.atr_len)
    adx = talib.ADX(high, low, close, timeperiod=p.adx_len)
    ema_fast = talib.EMA(close, timeperiod=p.ema_fast_len)
    ema_slow = talib.EMA(close, timeperiod=p.ema_slow_len)
    don_hi = (
        out[c["high"]].rolling(p.donch_len, min_periods=p.donch_len).max().to_numpy(dtype=float)
    )
    don_lo = (
        out[c["low"]].rolling(p.donch_len, min_periods=p.donch_len).min().to_numpy(dtype=float)
    )
    bb_hi, bb_mid, bb_lo = talib.BBANDS(
        close,
        timeperiod=p.bb_len,
        nbdevup=p.bb_mult,
        nbdevdn=p.bb_mult,
        matype=0,  # SMA middle band -> population stdev (Pine parity)
    )
    rsi = talib.RSI(close, timeperiod=p.rsi_len)

    out["atr"] = atr
    out["adx"] = adx
    out["ema_fast"] = ema_fast
    out["ema_slow"] = ema_slow
    out["don_hi"] = don_hi
    out["don_lo"] = don_lo
    out["bb_mid"] = bb_mid
    out["bb_hi"] = bb_hi
    out["bb_lo"] = bb_lo
    out["rsi"] = rsi

    # ---- derived series (spec §3) ---- #
    with np.errstate(divide="ignore", invalid="ignore"):
        span_atr = (don_hi - don_lo) / np.where(atr == 0.0, np.nan, atr)
    out["span_atr"] = span_atr

    trending = (adx >= p.adx_threshold) & (span_atr >= p.chan_atr_min)
    up = close > ema_slow
    down = close < ema_slow
    out["trending"] = trending.astype(bool)
    out["up"] = up.astype(bool)
    out["down"] = down.astype(bool)

    state = np.where(trending & up, TREND_UP, np.where(trending & down, TREND_DOWN, RANGE))

    # spec §6: first warmup_bars() rows get state=0; rows with NaN core
    # indicators (e.g. ADX needs ~2x period) also get state=0.
    n = len(out)
    invalid = np.arange(n) < warmup_bars(p)
    nanmask = ~(
        np.isfinite(atr)
        & np.isfinite(adx)
        & np.isfinite(ema_fast)
        & np.isfinite(ema_slow)
        & np.isfinite(don_hi)
        & np.isfinite(don_lo)
        & np.isfinite(bb_mid)
        & np.isfinite(rsi)
        & np.isfinite(span_atr)
    )
    state = np.where(invalid | nanmask, RANGE, state)
    out["state"] = state.astype(int)
    out["state_label"] = np.where(
        state == TREND_UP, "TREND_UP", np.where(state == TREND_DOWN, "TREND_DOWN", "RANGE")
    )
    return out


# --------------------------------------------------------------------------- #
# Signals (spec §4)
# --------------------------------------------------------------------------- #
def signal_events(df: pd.DataFrame, p: RegimeParams = RegimeParams()) -> pd.DataFrame:
    """Add signal/exit columns to a copy of ``df``.

    If regime columns are already present (from ``compute_regime``) they are
    reused; otherwise ``compute_regime`` is called first. Adds: don_hi_prev,
    don_lo_prev, sig (tag or ""), dir (1/-1/0), exit_long, exit_short.

    Tag precedence (spec §4): RANGE_LONG < TREND_LONG < BRKOUT_LONG (breakout
    wins); symmetric for shorts. Breakout uses Pine ``ta.crossover`` semantics
    against the prior-bar channel: close[t] > don_hi_prev[t] and
    close[t-1] <= don_hi_prev[t-1] (== don_hi[t-2]); mirror for shorts.
    """
    if all(col in df.columns for col in REGIME_COLS):
        out = df.copy()
    else:
        out = compute_regime(df, p)

    c = _resolve_cols(out)
    n = len(out)
    state = out["state"].to_numpy(dtype=int)
    close = out[c["close"]].to_numpy(dtype=float)
    high = out[c["high"]].to_numpy(dtype=float)
    low = out[c["low"]].to_numpy(dtype=float)
    don_hi = out["don_hi"].to_numpy(dtype=float)
    don_lo = out["don_lo"].to_numpy(dtype=float)
    ema_fast = out["ema_fast"].to_numpy(dtype=float)
    ema_slow = out["ema_slow"].to_numpy(dtype=float)
    rsi = out["rsi"].to_numpy(dtype=float)

    # prior-bar channel + shift helpers (Pine-style history-referencing)
    don_hi_prev = np.full(n, np.nan)
    don_hi_prev[1:] = don_hi[:-1]
    don_lo_prev = np.full(n, np.nan)
    don_lo_prev[1:] = don_lo[:-1]
    don_hi_prev2 = np.full(n, np.nan)
    don_hi_prev2[2:] = don_hi[:-2]
    don_lo_prev2 = np.full(n, np.nan)
    don_lo_prev2[2:] = don_lo[:-2]
    close_prev = np.full(n, np.nan)
    close_prev[1:] = close[:-1]

    out["don_hi_prev"] = don_hi_prev
    out["don_lo_prev"] = don_lo_prev

    # ---- long signals (spec §4) ---- #
    range_long = (state == RANGE) & (low <= don_lo) & (rsi < 45.0)
    trend_long = (state == TREND_UP) & (low <= ema_fast) & (close > ema_slow)
    brkout_long = (
        (state == TREND_UP)
        & (close > don_hi_prev)
        & (close_prev <= don_hi_prev2)
    )

    # ---- short signals (spec §4) ---- #
    range_short = (state == RANGE) & (high >= don_hi) & (rsi > 55.0)
    trend_short = (state == TREND_DOWN) & (high >= ema_fast) & (close < ema_slow)
    brkout_short = (
        (state == TREND_DOWN)
        & (close < don_lo_prev)
        & (close_prev >= don_lo_prev2)
    )

    # tag precedence: np.select picks the FIRST true condition, so list
    # breakout first, then trend, then range (breakout wins).
    conditions = [brkout_long, trend_long, range_long, brkout_short, trend_short, range_short]
    tags = ["BRKOUT_LONG", "TREND_LONG", "RANGE_LONG", "BRKOUT_SHORT", "TREND_SHORT", "RANGE_SHORT"]
    sig = np.select(conditions, tags, default="")

    # spec §6: warmup rows produce no signals
    sig = np.where(np.arange(n) < warmup_bars(p), "", sig)
    sig = sig.astype(str)

    direction = np.where(
        np.isin(sig, list(LONG_TAGS)), 1, np.where(np.isin(sig, list(SHORT_TAGS)), -1, 0)
    ).astype(int)

    # ---- exits (spec §4; per-bar guidance, no position tracking) ---- #
    bb_mid = out["bb_mid"].to_numpy(dtype=float)
    exit_long = (
        ((state == RANGE) & (close >= bb_mid))
        | ((state == TREND_UP) & (close < ema_slow))
        | (state == TREND_DOWN)
    )
    exit_short = (
        ((state == RANGE) & (close <= bb_mid))
        | ((state == TREND_DOWN) & (close > ema_slow))
        | (state == TREND_UP)
    )
    warm = np.arange(n) < warmup_bars(p)
    exit_long = np.where(warm, False, exit_long)
    exit_short = np.where(warm, False, exit_short)

    out["sig"] = sig
    out["dir"] = direction
    out["exit_long"] = exit_long.astype(bool)
    out["exit_short"] = exit_short.astype(bool)
    return out
