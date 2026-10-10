"""Strategy V2: EMA20/144 regime + structural-low trailing stop (15min decision /
1min execution). All causal (a value at bar t uses bars <= t only).

Reuses ``features.ema`` / ``features.pivot_low`` and mirrors the fill conventions
of ``engine.run_structure_stop`` (entry = first 1min open of t+1 + 1 tick, stop
fill = min(stop, open) - 2 ticks, regime exit = next 1min open - 1 tick,
commission $0.75/side, point_value $2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import ema, pivot_low


EMA_FAST, EMA_SLOW = 20, 144


def regime_series(bars: pd.DataFrame):
    """Return (regime, ema20, ema144); regime = EMA20>EMA144 AND close>EMA144."""
    close = bars["close"]
    e_fast = ema(close, EMA_FAST)
    e_slow = ema(close, EMA_SLOW)
    regime = (e_fast > e_slow) & (close > e_slow)
    return regime.fillna(False), e_fast, e_slow


def rth_mask(index: pd.DatetimeIndex) -> pd.Series:
    """RTH entry window 09:45-15:45 (naive US-Eastern bars)."""
    mins = index.hour * 60 + index.minute
    return pd.Series((mins >= 9 * 60 + 45) & (mins <= 15 * 60 + 45), index=index)


def build_signals_v2(bars, feat, sf, trigger, atrp_cap=90.0, er_min=None, rth=False):
    """Regime + trigger + base gates {VR<1, ATRpct<=cap, ext<=2.5} (+ optional gates)."""
    close, open_, low, high = bars["close"], bars["open"], bars["low"], bars["high"]
    regime, e_fast, e_slow = regime_series(bars)
    atr = feat["atr"]
    ext = (close - e_fast) / atr
    blocked = (feat["vr"] >= 1.0) | (sf["atr_pct"] > float(atrp_cap)) | (ext > 2.5)

    if trigger == "T1":
        trig = regime & (low.rolling(3, min_periods=3).min() <= e_fast) & (close > e_fast) & (close > open_)
    elif trigger == "T2":
        trig = regime & (close > high.rolling(20, min_periods=20).max().shift(1))
    elif trigger == "T3":
        trig = regime & (close > high.rolling(10, min_periods=10).max().shift(1))
    else:
        raise ValueError(f"unknown trigger {trigger}")

    sig = trig & (~blocked)
    if er_min is not None:
        sig = sig & (feat["er"] >= float(er_min))
    if rth:
        sig = sig & rth_mask(bars.index)
    return sig.fillna(False), e_fast, e_slow


def run_ema_struct(bars15, bars1, feat15, signal, cost, *,
                   pivot_n: int = 2, buf: float = 0.25, floor_mult: float = 2.0,
                   act_mult: float = 0.0, exit_ema=None, regime_confirm: int = 0,
                   breakeven_atr: float = None, breaker: dict = None,
                   freq_minutes: int = 15) -> dict:
    """Structural-low trailing stop with regime exit, activation, floor, breaker.

    Returns a dict compatible with ``metrics_row`` (normalised equity + trades).
    """
    idx15 = bars15.index
    c15 = bars15["close"].to_numpy(float)
    atr15 = feat15["atr"].reindex(idx15).to_numpy(float)
    sig = signal.reindex(idx15).fillna(False).to_numpy(bool)
    n15 = len(bars15)
    tick = float(cost.tick)
    pv = float(cost.point_value)
    comm = float(cost.commission_per_side)
    ema_exit = (exit_ema.reindex(idx15).to_numpy(float) if exit_ema is not None
                else feat15["ema50"].reindex(idx15).to_numpy(float))

    piv_flag, piv_price = pivot_low(bars15["low"], n=pivot_n)
    piv_flag = piv_flag.to_numpy(bool)
    piv_price = piv_price.to_numpy(float)

    idx1 = bars1.index
    o1 = bars1["open"].to_numpy(float)
    l1 = bars1["low"].to_numpy(float)
    c1 = bars1["close"].to_numpy(float)
    n1 = len(bars1)
    t1 = idx1.to_numpy()
    t15 = idx15.to_numpy()
    bs = np.searchsorted(t1, t15)
    be = np.searchsorted(t1, t15 + np.timedelta64(int(freq_minutes), "m"))
    need = 1 if not regime_confirm else int(regime_confirm)

    breaker_on = breaker is not None
    br_daily = br_weekly = br_dd = None
    br_resume = 0.5
    if breaker_on:
        br_daily = breaker.get("daily")
        br_weekly = breaker.get("weekly")
        br_dd = breaker.get("dd")
        br_resume = float(breaker.get("dd_resume_frac", 0.5))
        day_arr = idx15.normalize().to_numpy()
        week_arr = idx15.to_period("W").astype("int64").to_numpy()
    peak_pts = 0.0
    dd_active = day_halt = week_halt = False
    cur_day = cur_week = None
    day_ref = week_ref = 0.0
    brk_active = False

    realized = 0.0
    pos = 0
    entry_fill = 0.0
    entry_atr = np.nan
    entry_bar = 0
    entry_t1 = 0
    stop = np.nan
    pending = False
    pending_atr = np.nan
    pending_exit = False
    trades = []
    equity_usd = np.zeros(n15)
    n_breakouts = 0
    n_blocked = 0
    halted = set()

    def _close(xk, exit_time, exit_raw, reason):
        nonlocal realized, pos, entry_fill
        pnl_pts = float(exit_raw - entry_fill)
        trades.append({
            "entry_time": idx1[entry_t1], "exit_time": exit_time, "side": 1,
            "entry_px": float(entry_fill), "exit_px": float(exit_raw),
            "pnl_pts": pnl_pts, "pnl_usd_net": float(pnl_pts * pv - 2.0 * comm),
            "bars_held": int(xk - entry_bar), "atr_entry": float(entry_atr),
            "entry_bar": int(entry_bar), "exit_bar": int(xk), "exit_reason": reason,
        })
        realized += pnl_pts * pv - 2.0 * comm
        pos = 0

    for k in range(n15):
        # deferred regime-break exit at this bar's first 1min open
        if pos == 1 and pending_exit:
            if int(bs[k]) < n1:
                ne = int(bs[k])
                _close(k, idx1[ne], o1[ne] - tick, "regime")
            else:
                _close(n15 - 1, idx1[n1 - 1], c1[n1 - 1] - tick, "end")
            pending_exit = False

        # entry (pending set at the previous bar)
        if pos == 0 and pending:
            e1 = int(bs[k])
            if e1 < n1:
                entry_fill = o1[e1] + tick
                entry_atr = pending_atr
                stop = entry_fill - floor_mult * entry_atr
                entry_bar = k
                entry_t1 = e1
                pos = 1
            pending = False

        exited_this_bar = False
        # intrabar stop scan (1min lows)
        if pos == 1:
            s = int(bs[k]); e = int(be[k])
            if k == entry_bar:
                s = max(s, entry_t1)
            if e > s:
                seg = l1[s:e]
                hit = seg <= stop
                if hit.any():
                    j = s + int(np.argmax(hit))
                    raw = min(stop, o1[j]) if o1[j] < stop else stop
                    _close(k, idx1[j], raw - 2.0 * tick, "stop")
                    exited_this_bar = True

        # regime-break detection at this bar's close -> exit next bar
        if pos == 1 and not exited_this_bar and not pending_exit:
            if need == 1:
                broken = bool(np.isfinite(ema_exit[k]) and c15[k] < ema_exit[k])
            else:
                broken = k >= need - 1 and all(
                    np.isfinite(ema_exit[k - j]) and c15[k - j] < ema_exit[k - j]
                    for j in range(need))
            if broken:
                pending_exit = True

        # breakeven + activation + ratchet (only if still holding)
        if pos == 1 and not pending_exit:
            if breakeven_atr is not None and np.isfinite(entry_atr):
                if (c15[k] - entry_fill) >= breakeven_atr * entry_atr:
                    be_level = entry_fill + tick + 2.0 * comm / pv
                    if be_level > stop:
                        stop = be_level
            if act_mult <= 0.0 or (c15[k] - entry_fill) >= act_mult * entry_atr:
                if (k >= pivot_n and piv_flag[k]
                        and np.isfinite(piv_price[k - pivot_n]) and np.isfinite(atr15[k])):
                    cand = piv_price[k - pivot_n] - buf * atr15[k]
                    level = float(np.ceil(cand / tick - 1e-9) * tick)
                    if level > stop:
                        stop = level

        # mark-to-market
        equity_usd[k] = realized + ((c15[k] - entry_fill) * pv if pos == 1 else 0.0)

        # breaker (blocks next-bar entries only)
        if breaker_on:
            eq_pts = equity_usd[k] / pv
            if eq_pts > peak_pts:
                peak_pts = eq_pts
            dkey = day_arr[k]
            if dkey != cur_day:
                cur_day = dkey
                day_ref = (equity_usd[k - 1] / pv) if k > 0 else eq_pts
                day_halt = False
            wkey = week_arr[k]
            if wkey != cur_week:
                cur_week = wkey
                week_ref = (equity_usd[k - 1] / pv) if k > 0 else eq_pts
                week_halt = False
            if br_daily is not None and (eq_pts - day_ref) <= br_daily:
                day_halt = True
            if br_weekly is not None and (eq_pts - week_ref) <= br_weekly:
                week_halt = True
            if br_dd is not None:
                dd_pt = peak_pts - eq_pts
                if dd_pt > br_dd:
                    dd_active = True
                elif dd_pt < br_dd * br_resume:
                    dd_active = False
            brk_active = bool(day_halt or week_halt or dd_active)
            if brk_active:
                halted.add(dkey)

        # signal detection (next-bar entry)
        if pos == 0 and not pending and k + 1 < n15 and sig[k]:
            if breaker_on and brk_active:
                n_blocked += 1
            else:
                n_breakouts += 1
                pending = True
                pending_atr = atr15[k]

    if pos == 1:
        _close(n15 - 1, idx1[n1 - 1], c1[n1 - 1] - tick, "end")
        equity_usd[n15 - 1] = realized

    capital = float(c15[0]) * pv
    return {
        "equity": pd.Series(1.0 + equity_usd / capital, index=idx15),
        "trades": trades, "capital": capital, "cost": cost,
        "n_breakouts": int(n_breakouts), "n_breaker_blocked": int(n_blocked),
        "halted_days": int(len(halted)),
    }
