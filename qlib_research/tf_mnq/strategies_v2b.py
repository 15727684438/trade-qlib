"""Strategy V2B: EMA20/144 regime + Donchian breakout, LONG **and** SHORT, with an
optional 2-leg partial take-profit and a structural (pivot) trailing stop on the
runner. 5min decisions / 1min execution. All causal.

This module generalises ``strategies_v2.run_ema_struct``:
  * SHORT trades mirror the long structural-stop path (pivot_high, stop above
    entry, regime exit when close > EMA).
  * a single partial take-profit leg (``partial_R`` in R units, where
    R = floor_mult * ATR(entry)) closes one contract and the runner trails.

Fill conventions
----------------
``fills="cost"``  : canonical cost model, adverse slip = ``cost.slip_ticks * tick``
                    on entry, stop, regime, partial and end-of-data fills. This is
                    what makes the 1.5t vs 3.0t stress runs differ.
``fills="legacy"``: reproduces the exact fill conventions of
                    ``strategies_v2.run_ema_struct`` (entry +1 tick, stop -2 ticks,
                    regime/end -1 tick, no cost-dependent slip) so the long-only,
                    1-contract, no-partial path can be verified trade-for-trade
                    against the frozen A_T2 anchor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import _as_series, ema, pivot_low


EMA_FAST, EMA_SLOW = 20, 144


# --------------------------------------------------------------------------- #
# signal construction
# --------------------------------------------------------------------------- #
def regime_series_b(bars: pd.DataFrame):
    """Return (long_regime, short_regime, ema20, ema144)."""
    close = bars["close"]
    e_fast = ema(close, EMA_FAST)
    e_slow = ema(close, EMA_SLOW)
    long_reg = ((e_fast > e_slow) & (close > e_slow)).fillna(False)
    short_reg = ((e_fast < e_slow) & (close < e_slow)).fillna(False)
    return long_reg, short_reg, e_fast, e_slow


def rth_mask(index: pd.DatetimeIndex) -> pd.Series:
    """RTH entry window 09:45-15:45 (naive US-Eastern bars)."""
    mins = index.hour * 60 + index.minute
    return pd.Series((mins >= 9 * 60 + 45) & (mins <= 15 * 60 + 45), index=index)


def build_signals_v2b(bars, feat, sf, trigger, atrp_cap=90.0, rth=False,
                      allow_short=True):
    """Regime + trigger + gates {VR<1, ATRpct<=cap, extension<=2.5}.

    Returns ``(sig_long, sig_short)`` boolean Series. Extension is measured from
    EMA20 in the trade direction (mirrored for shorts).
    """
    close, open_, low, high = bars["close"], bars["open"], bars["low"], bars["high"]
    reg_l, reg_s, e_fast, e_slow = regime_series_b(bars)
    atr = feat["atr"]
    ext_l = (close - e_fast) / atr
    ext_s = (e_fast - close) / atr

    if trigger == "T1":
        trig_l = reg_l & (low.rolling(3, min_periods=3).min() <= e_fast) & \
            (close > e_fast) & (close > open_)
        trig_s = reg_s & (high.rolling(3, min_periods=3).max() >= e_fast) & \
            (close < e_fast) & (close < open_)
    elif trigger == "T2":
        trig_l = reg_l & (close > high.rolling(20, min_periods=20).max().shift(1))
        trig_s = reg_s & (close < low.rolling(20, min_periods=20).min().shift(1))
    elif trigger == "T3":
        trig_l = reg_l & (close > high.rolling(10, min_periods=10).max().shift(1))
        trig_s = reg_s & (close < low.rolling(10, min_periods=10).min().shift(1))
    else:
        raise ValueError(f"unknown trigger {trigger}")

    blocked_l = (feat["vr"] >= 1.0) | (sf["atr_pct"] > float(atrp_cap)) | (ext_l > 2.5)
    blocked_s = (feat["vr"] >= 1.0) | (sf["atr_pct"] > float(atrp_cap)) | (ext_s > 2.5)

    sig_l = (trig_l & (~blocked_l)).fillna(False)
    sig_s = (trig_s & (~blocked_s)).fillna(False)
    if not allow_short:
        sig_s = pd.Series(False, index=bars.index)
    if rth:
        m = rth_mask(bars.index)
        sig_l = sig_l & m
        sig_s = sig_s & m
    return sig_l.fillna(False), sig_s.fillna(False)


# --------------------------------------------------------------------------- #
# pivot high (mirror of features.pivot_low)
# --------------------------------------------------------------------------- #
def pivot_high(high, n: int = 2):
    """Fractal pivot high; confirmation flagged at bar ``i+n``, price at ``i``."""
    high = _as_series(high)
    arr = high.to_numpy(dtype=float)
    m = len(arr)
    flag = np.zeros(m, dtype=bool)
    price = np.full(m, np.nan, dtype=float)
    for i in range(n, m - n):
        window = arr[i - n : i + n + 1]
        mx = window.max()
        if arr[i] == mx and int((window == mx).sum()) == 1:
            flag[i + n] = True
            price[i] = arr[i]
    return pd.Series(flag, index=high.index), pd.Series(price, index=high.index)


# --------------------------------------------------------------------------- #
# engine
# --------------------------------------------------------------------------- #
def run_ema2_struct(bars15, bars1, feat15, sig_long, sig_short, cost, *,
                    size: int = 1, floor_mult: float = 2.0, pivot_n: int = 2,
                    buf: float = 0.25, partial_R: float = None,
                    exit_ema=None, regime_confirm: int = 0,
                    fills: str = "cost", freq_minutes: int = 5,
                    allow_short: bool = True) -> dict:
    """Long/short structural-stop engine with optional 2-leg partial exit.

    ``partial_R`` is expressed in R units (R = floor_mult * ATR(entry)); e.g.
    0.5 closes one contract at entry +/- 0.5R and trails the runner.
    """
    idx15 = bars15.index
    c15 = bars15["close"].to_numpy(float)
    atr15 = feat15["atr"].reindex(idx15).to_numpy(float)
    sl = sig_long.reindex(idx15).fillna(False).to_numpy(bool)
    ss = (sig_short.reindex(idx15).fillna(False).to_numpy(bool)
          if allow_short else np.zeros(len(idx15), bool))
    n15 = len(bars15)
    tick = float(cost.tick)
    pv = float(cost.point_value)
    comm = float(cost.commission_per_side)

    if fills == "legacy":
        e_slip = tick
        x_slip = tick
        stop_slip = 2.0 * tick
    else:
        e_slip = float(cost.slip_ticks) * tick
        x_slip = e_slip
        stop_slip = e_slip
    p_slip = max(x_slip, tick)  # partial exit: at least one tick adverse

    ema_exit = (exit_ema.reindex(idx15).to_numpy(float) if exit_ema is not None
                else feat15["ema50"].reindex(idx15).to_numpy(float))

    pivl_flag, pivl_price = pivot_low(bars15["low"], n=pivot_n)
    pivl_flag = pivl_flag.to_numpy(bool)
    pivl_price = pivl_price.to_numpy(float)
    pivh_flag, pivh_price = pivot_high(bars15["high"], n=pivot_n)
    pivh_flag = pivh_flag.to_numpy(bool)
    pivh_price = pivh_price.to_numpy(float)

    idx1 = bars1.index
    o1 = bars1["open"].to_numpy(float)
    h1 = bars1["high"].to_numpy(float)
    l1 = bars1["low"].to_numpy(float)
    c1 = bars1["close"].to_numpy(float)
    n1 = len(bars1)
    t1 = idx1.to_numpy()
    t15 = idx15.to_numpy()
    bs = np.searchsorted(t1, t15)
    be = np.searchsorted(t1, t15 + np.timedelta64(int(freq_minutes), "m"))
    need = 1 if not regime_confirm else int(regime_confirm)

    state = {
        "realized": 0.0, "pos": 0, "qty": 0, "entry_fill": 0.0, "entry_atr": np.nan,
        "entry_bar": 0, "entry_t1": 0, "stop": np.nan, "trade_pnl": 0.0,
        "partial_done": False,
    }
    pending = 0
    pending_atr = np.nan
    pending_exit = False
    trades = []
    equity_usd = np.zeros(n15)
    n_breakouts = 0
    n_partial = 0

    def finish(k, exit_raw, reason):
        st = state
        pnl = (exit_raw - st["entry_fill"]) * st["pos"] * st["qty"] * pv \
            - comm * (st["qty"] + size)
        st["trade_pnl"] += pnl
        st["realized"] += pnl
        trades.append({
            "entry_time": idx1[st["entry_t1"]], "exit_time": idx1[k],
            "side": int(st["pos"]),
            "entry_px": float(st["entry_fill"]), "exit_px": float(exit_raw),
            "pnl_pts": float(st["trade_pnl"] / (pv * size)),
            "pnl_usd_net": float(st["trade_pnl"]),
            "bars_held": int(k - st["entry_bar"]), "atr_entry": float(st["entry_atr"]),
            "entry_bar": int(st["entry_bar"]), "exit_bar": int(k),
            "exit_reason": reason, "size": int(size),
            "partial_done": bool(st["partial_done"]),
        })
        st["pos"] = 0
        st["qty"] = 0
        st["partial_done"] = False
        st["trade_pnl"] = 0.0
        st["stop"] = np.nan

    def do_partial(j, target):
        st = state
        raw = target - st["pos"] * p_slip
        pnl = (raw - st["entry_fill"]) * st["pos"] * pv - comm
        st["trade_pnl"] += pnl
        st["realized"] += pnl
        st["qty"] -= 1
        st["partial_done"] = True

    for k in range(n15):
        # 1. deferred regime-break exit (full remaining) at this bar's first open
        if state["pos"] != 0 and pending_exit:
            if int(bs[k]) < n1:
                ne = int(bs[k])
                finish(ne, o1[ne] - state["pos"] * x_slip, "regime")
            else:
                finish(n1 - 1, c1[n1 - 1] - state["pos"] * x_slip, "end")
            pending_exit = False

        # 2. entry (pending set at the previous bar's close)
        if state["pos"] == 0 and pending != 0:
            e1 = int(bs[k])
            if e1 < n1:
                state["pos"] = pending
                state["qty"] = int(size)
                state["entry_fill"] = o1[e1] + pending * e_slip
                state["entry_atr"] = pending_atr
                state["stop"] = state["entry_fill"] - pending * floor_mult * pending_atr
                state["entry_bar"] = k
                state["entry_t1"] = e1
                state["trade_pnl"] = 0.0
                state["partial_done"] = False
            pending = 0

        exited_this_bar = False
        # 3. intrabar scan (1min): stop first (conservative), then partial
        if state["pos"] != 0:
            s = int(bs[k]); e = int(be[k])
            if k == state["entry_bar"]:
                s = max(s, state["entry_t1"])
            if partial_R is not None and np.isfinite(state["entry_atr"]):
                target = state["entry_fill"] + state["pos"] * partial_R \
                    * floor_mult * state["entry_atr"]
            else:
                target = np.nan
            for j in range(s, e):
                if not np.isfinite(state["stop"]):
                    break
                if state["pos"] > 0:
                    if l1[j] <= state["stop"]:
                        raw = min(state["stop"], o1[j]) if o1[j] < state["stop"] \
                            else state["stop"]
                        finish(j, raw - state["pos"] * stop_slip, "stop")
                        exited_this_bar = True
                        break
                    if (not state["partial_done"] and np.isfinite(target)
                            and h1[j] >= target):
                        do_partial(j, target)
                        n_partial += 1
                else:
                    if h1[j] >= state["stop"]:
                        raw = max(state["stop"], o1[j]) if o1[j] > state["stop"] \
                            else state["stop"]
                        finish(j, raw - state["pos"] * stop_slip, "stop")
                        exited_this_bar = True
                        break
                    if (not state["partial_done"] and np.isfinite(target)
                            and l1[j] <= target):
                        do_partial(j, target)
                        n_partial += 1

        # 4. regime-break detection at this bar's close -> exit next bar
        if state["pos"] != 0 and not exited_this_bar and not pending_exit:
            if need == 1:
                broken = bool(np.isfinite(ema_exit[k])
                              and state["pos"] * (c15[k] - ema_exit[k]) < 0)
            else:
                broken = k >= need - 1 and all(
                    np.isfinite(ema_exit[k - j])
                    and state["pos"] * (c15[k - j] - ema_exit[k - j]) < 0
                    for j in range(need))
            if broken:
                pending_exit = True

        # 5. structural trailing stop on the runner (ratchet-only)
        if state["pos"] != 0 and not pending_exit:
            if state["pos"] > 0:
                if (k >= pivot_n and pivl_flag[k]
                        and np.isfinite(pivl_price[k - pivot_n])
                        and np.isfinite(atr15[k])):
                    cand = pivl_price[k - pivot_n] - buf * atr15[k]
                    level = float(np.ceil(cand / tick - 1e-9) * tick)
                    if level > state["stop"]:
                        state["stop"] = level
            else:
                if (k >= pivot_n and pivh_flag[k]
                        and np.isfinite(pivh_price[k - pivot_n])
                        and np.isfinite(atr15[k])):
                    cand = pivh_price[k - pivot_n] + buf * atr15[k]
                    level = float(np.floor(cand / tick + 1e-9) * tick)
                    if level < state["stop"]:
                        state["stop"] = level

        # 6. mark-to-market
        if state["pos"] != 0:
            equity_usd[k] = state["realized"] + (
                state["qty"] * state["pos"] * (c15[k] - state["entry_fill"]) * pv)
        else:
            equity_usd[k] = state["realized"]

        # 7. signal detection (next-bar entry)
        if state["pos"] == 0 and pending == 0 and k + 1 < n15:
            if sl[k]:
                pending = 1
                pending_atr = atr15[k]
                n_breakouts += 1
            elif ss[k]:
                pending = -1
                pending_atr = atr15[k]
                n_breakouts += 1

    if state["pos"] != 0:
        finish(n1 - 1, c1[n1 - 1] - state["pos"] * x_slip, "end")
        equity_usd[n15 - 1] = state["realized"]

    capital = float(c15[0]) * pv
    return {
        "equity": pd.Series(1.0 + equity_usd / capital, index=idx15),
        "trades": trades, "capital": capital, "cost": cost,
        "n_breakouts": int(n_breakouts), "n_breaker_blocked": 0,
        "halted_days": 0, "n_partial": int(n_partial),
    }
