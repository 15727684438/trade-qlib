"""Reusable strategy construction for the 15min decision layer (IS only).

Keeps the signal/veto wiring in one place so the diagnostics suite, the M2 grid
and the reporting script all build *identical* signals. Every helper is causal:
series at ``t`` depend only on bars ``<= t``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import atr_percentile, ema, pullback_resumption_trigger

TON = ("ext", "vr", "vol", "tod", "gate")


def strategy_frame(bars: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    """Per-bar context columns (causal) shared by the trigger/veto builders."""
    close = bars["close"]
    atr = feat["atr"]
    out = pd.DataFrame(index=bars.index)
    out["ext"] = (close - ema(close, 20)) / atr
    out["atr_pct"] = atr_percentile(atr, window=500, min_periods=100)
    out["slope3"] = feat["ema50"].diff(3)
    out["context_long"] = (close > feat["ema50"]) & (out["slope3"] > 0)
    return out


def primary_trigger(bars: pd.DataFrame, atr: pd.Series, trigger: str,
                    depth_mult: float = 1.0, lookback: int = 10) -> pd.Series:
    """Raw trigger (before context/vetoes). ``trigger`` names from D2."""
    if trigger in ("a_donchian20_breakout", "donchian20_breakout"):
        don_hi = bars["high"].rolling(20, min_periods=20).max().shift(1)
        return (bars["close"] > don_hi).fillna(False)
    return pullback_resumption_trigger(
        bars["close"], bars["high"], bars["low"], atr, depth_mult=depth_mult, lookback=lookback
    )


def veto_blocked(bars: pd.DataFrame, feat: pd.DataFrame, sf: pd.DataFrame,
                 components=TON, ext_mult: float = 2.5, vr_thr: float = 1.0,
                 atr_pct_cap: float = 90.0, tod_hours=(16, 17)) -> pd.Series:
    """Boolean mask: True where an entry is blocked by the selected vetoes."""
    blocked = pd.Series(False, index=bars.index)
    if "ext" in components:
        blocked = blocked | (sf["ext"] > ext_mult)
    if "vr" in components:
        blocked = blocked | (feat["vr"] >= vr_thr)
    if "vol" in components:
        blocked = blocked | (sf["atr_pct"] > atr_pct_cap)
    if "tod" in components:
        blocked = blocked | pd.Series(bars.index.hour.isin(list(tod_hours)), index=bars.index)
    if "gate" in components:
        blocked = blocked | (feat["gate"] != 1)
    return blocked.fillna(True)


def build_signal(bars: pd.DataFrame, feat: pd.DataFrame, atr: pd.Series,
                 trigger: str = "b_pullback_resumption", depth_mult: float = 1.0,
                 veto_components=(), ext_mult: float = 2.5, vr_thr: float = 1.0,
                 atr_pct_cap: float = 90.0, tod_hours=(16, 17)) -> pd.Series:
    """Full long-entry signal: context & trigger & ~vetoes."""
    sf = strategy_frame(bars, feat)
    trig = primary_trigger(bars, atr, trigger, depth_mult=depth_mult)
    sig = sf["context_long"] & trig
    if veto_components:
        blocked = veto_blocked(bars, feat, sf, components=veto_components,
                               ext_mult=ext_mult, vr_thr=vr_thr,
                               atr_pct_cap=atr_pct_cap, tod_hours=tod_hours)
        sig = sig & (~blocked)
    return sig.fillna(False)


def run_t1(bars_dec: pd.DataFrame, bars1: pd.DataFrame, feat: pd.DataFrame,
           signal: pd.Series, cost, pivot_n: int = 2, buffer_mult: float = 0.25,
           stop_mult: float = 1.5, freq_minutes: int = 15) -> dict:
    """Run the 1min-executed structure-low trailing stop on a decision signal."""
    from .engine import run_structure_stop

    return run_structure_stop(
        bars_dec, bars1, feat, signal, cost,
        stop_mult=stop_mult, buffer_mult=buffer_mult, pivot_n=pivot_n,
        freq_minutes=freq_minutes,
    )
