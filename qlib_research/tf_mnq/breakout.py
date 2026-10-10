"""Generalised Donchian-breakout engine with exit shaping + fine-frame structure stop.

This is a superset of ``run_m1b_diagnostics.run_breakout`` (the B2/B2g base). With
all shaping options Off and ``entry_mode='market'`` it reproduces that engine's
trade ledger exactly (verified in ``scripts/run_m3_stages.py``).

Causality contract (unchanged): the target/exit state at bar ``k`` may depend only
on bars ``<= k``; a signal seen at the close of ``k`` is acted on at ``k+1`` (market)
or via a limit resting from ``k+1`` (retest). Stops are evaluated intrabar with
fills at ``min(stop, open)`` (long) / ``max(stop, open)`` (short).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import CostCfg


_DW_CACHE = {}


def _day_week_arrays(idx):
    """Cached (day, iso-week) numpy keys for the breaker (avoids recompute)."""
    key = (id(idx), len(idx), int(idx[0].value), int(idx[-1].value))
    hit = _DW_CACHE.get(key)
    if hit is None:
        hit = (idx.normalize().to_numpy(), idx.to_period("W").astype("int64").to_numpy())
        _DW_CACHE[key] = hit
    return hit


def run_breakout_ex(
    bars: pd.DataFrame,
    atr: pd.Series,
    cost: CostCfg,
    donch: int = 20,
    stop_mult: float = 3.0,
    allow_short: bool = False,
    veto_long: pd.Series = None,
    veto_short: pd.Series = None,
    breakeven_atr: float = None,
    time_stop_bars: int = None,
    time_stop_min_pnl: float = 0.0,
    partial_atr: float = None,
    partial_frac: float = 0.5,
    entry_mode: str = "market",
    retest_bars: int = 3,
    initial_stop_mult: float = None,
    ratchet_only: bool = False,
    breaker: dict = None,
) -> dict:
    idx = bars.index
    o = bars["open"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    lo = bars["low"].to_numpy(float)
    c = bars["close"].to_numpy(float)
    atrn = atr.reindex(idx).to_numpy(float)
    n = len(bars)
    pv = float(cost.point_value)
    slip = float(cost.slip_ticks) * float(cost.tick)
    comm = float(cost.commission_per_side)
    tick = float(cost.tick)

    don_hi = bars["high"].rolling(donch, min_periods=donch).max().shift(1).to_numpy(float)
    don_lo = bars["low"].rolling(donch, min_periods=donch).min().shift(1).to_numpy(float)
    vlong = (veto_long.reindex(idx).fillna(True).to_numpy(bool)
             if veto_long is not None else np.zeros(n, bool))
    vshort = (veto_short.reindex(idx).fillna(True).to_numpy(bool)
              if veto_short is not None else np.zeros(n, bool))

    realized = 0.0
    pos = 0
    pos_frac = 1.0
    entry_fill = entry_raw = 0.0
    entry_bar = 0
    entry_atr = np.nan
    signal_bar = 0
    water = 0.0
    pending = 0
    pending_signal_bar = 0
    limit_side = 0
    limit_level = np.nan
    limit_until = -1
    limit_signal_bar = 0
    partial_done = False
    partial_pnl_pts = 0.0
    n_breakouts = 0
    equity_usd = np.zeros(n)
    trades = []

    # --- risk-first additions (all causal; defaults reproduce legacy behaviour) ---
    stop = np.nan  # persistent stop level (long-only)
    breaker_on = breaker is not None
    br_daily = br_weekly = br_dd = None
    br_resume = 0.5
    peak_pts = 0.0
    brk_active = False
    dd_active = False
    day_halt = week_halt = False
    cur_day = cur_week = None
    day_ref_pts = week_ref_pts = 0.0
    n_breaker_blocked = 0
    halted_days = set()
    if breaker_on:
        br_daily = breaker.get("daily")
        br_weekly = breaker.get("weekly")
        br_dd = breaker.get("dd")
        br_resume = float(breaker.get("dd_resume_frac", 0.5))
        day_arr, week_arr = _day_week_arrays(idx)

    def _round_stop(level, side):
        if side > 0:
            return float(np.ceil(level / tick - 1e-9) * tick)
        return float(np.floor(level / tick + 1e-9) * tick)

    def _enter(k, side, fill, raw, sig_bar):
        nonlocal pos, pos_frac, entry_fill, entry_raw, entry_bar, entry_atr, water, stop
        nonlocal partial_done, partial_pnl_pts, realized, signal_bar
        pos = side
        pos_frac = 1.0
        entry_fill = fill
        entry_raw = raw
        entry_bar = k
        entry_atr = atrn[k]
        signal_bar = sig_bar
        water = h[k] if side > 0 else lo[k]
        if initial_stop_mult is not None and np.isfinite(entry_atr):
            stop = entry_fill - side * initial_stop_mult * entry_atr
        else:
            stop = -np.inf if side > 0 else np.inf
        partial_done = False
        partial_pnl_pts = 0.0
        realized -= comm  # one side of commission at entry (base behaviour)

    def _close(k, raw, reason):
        nonlocal pos, pos_frac, realized, trades, stop
        exit_fill = raw - pos * slip
        rem_pts = (exit_fill - entry_fill) * pos * pos_frac
        realized += rem_pts * pv - comm * pos_frac
        total_pts = partial_pnl_pts + rem_pts
        trades.append({
            "entry_time": idx[entry_bar], "exit_time": idx[k], "side": int(pos),
            "entry_px": float(entry_fill), "exit_px": float(exit_fill),
            "pnl_pts": float(total_pts), "pnl_usd_net": float(total_pts * pv - 2.0 * comm),
            "bars_held": int(k - entry_bar), "atr_entry": float(entry_atr),
            "entry_bar": int(entry_bar), "exit_bar": int(k), "signal_bar": int(signal_bar),
            "frac": 1.0, "partial_frac": float(1.0 - pos_frac),
            "exit_reason": reason,
        })
        pos = 0
        pos_frac = 1.0
        stop = np.nan

    def _do_partial(k, target):
        nonlocal pos_frac, partial_done, partial_pnl_pts, realized
        p = (target - entry_fill) * pos * partial_frac
        partial_pnl_pts += p
        realized += p * pv - comm * partial_frac
        pos_frac -= partial_frac
        partial_done = True
        if pos_frac <= 1e-12:
            # full size already realized at target
            _finalize_partial_only(k, target)
            return True
        return False

    def _finalize_partial_only(k, target):
        nonlocal pos, trades
        total_pts = partial_pnl_pts
        trades.append({
            "entry_time": idx[entry_bar], "exit_time": idx[k], "side": int(pos),
            "entry_px": float(entry_fill), "exit_px": float(target - pos * slip),
            "pnl_pts": float(total_pts), "pnl_usd_net": float(total_pts * pv - 2.0 * comm),
            "bars_held": int(k - entry_bar), "atr_entry": float(entry_atr),
            "entry_bar": int(entry_bar), "exit_bar": int(k), "signal_bar": int(signal_bar),
            "frac": 1.0, "partial_frac": 1.0, "exit_reason": "partial_full",
        })
        pos = 0

    for k in range(n):
        # ---------------- entry handling ----------------
        if pos == 0:
            if pending != 0:
                side = pending
                pending = 0
                _enter(k, side, o[k] + side * slip, o[k], pending_signal_bar)
            elif limit_side != 0 and k <= limit_until:
                if limit_side > 0 and lo[k] <= limit_level:
                    raw = min(limit_level, o[k])
                    _enter(k, 1, raw + slip, raw, limit_signal_bar)
                    limit_side = 0
                elif limit_side < 0 and h[k] >= limit_level:
                    raw = max(limit_level, o[k])
                    _enter(k, -1, raw - slip, raw, limit_signal_bar)
                    limit_side = 0
            elif limit_side != 0 and k > limit_until:
                limit_side = 0  # expired unfilled

        # ---------------- stop / partial / time ----------------
        if pos != 0 and k > 1 and np.isfinite(atrn[k - 1]):
            atr_prev = atrn[k - 1]
            trail = water - pos * stop_mult * atr_prev
            if ratchet_only and np.isfinite(stop):
                stop = max(stop, trail) if pos > 0 else min(stop, trail)
            else:
                stop = trail
            if initial_stop_mult is not None and np.isfinite(entry_atr):
                hard = entry_fill - pos * initial_stop_mult * entry_atr
                stop = max(stop, hard) if pos > 0 else min(stop, hard)
            if breakeven_atr is not None and np.isfinite(entry_atr):
                if pos * (water - entry_fill) >= breakeven_atr * entry_atr:
                    be = entry_fill + pos * (slip + 2.0 * comm / pv)
                    stop = max(stop, _round_stop(be, pos)) if pos > 0 else min(stop, _round_stop(be, pos))
            hit = (lo[k] <= stop) if pos > 0 else (h[k] >= stop)
            if hit:
                raw = min(stop, o[k]) if pos > 0 else max(stop, o[k])
                _close(k, raw, "stop")
            else:
                if pos > 0:
                    water = max(water, h[k])
                else:
                    water = min(water, lo[k])

        if pos != 0 and partial_atr is not None and not partial_done and np.isfinite(entry_atr):
            target = entry_fill + pos * partial_atr * entry_atr
            if (pos > 0 and h[k] >= target) or (pos < 0 and lo[k] <= target):
                if _do_partial(k, target):
                    pass

        if pos != 0 and time_stop_bars is not None and (k - entry_bar) >= time_stop_bars:
            if pos * (c[k] - entry_fill) < time_stop_min_pnl:
                _close(k, c[k], "time")

        # ---------------- mark-to-market ----------------
        unreal = pos_frac * pos * (c[k] - entry_fill) * pv if pos != 0 else 0.0
        equity_usd[k] = realized + unreal

        # ---------------- breaker (evaluated at decision close; blocks NEXT-bar entries) ---
        if breaker_on:
            eq_pts = equity_usd[k] / pv
            if eq_pts > peak_pts:
                peak_pts = eq_pts
            dkey = day_arr[k]
            if dkey != cur_day:
                cur_day = dkey
                day_ref_pts = (equity_usd[k - 1] / pv) if k > 0 else eq_pts
                day_halt = False
            wkey = week_arr[k]
            if wkey != cur_week:
                cur_week = wkey
                week_ref_pts = (equity_usd[k - 1] / pv) if k > 0 else eq_pts
                week_halt = False
            if br_daily is not None and (eq_pts - day_ref_pts) <= br_daily:
                day_halt = True
            if br_weekly is not None and (eq_pts - week_ref_pts) <= br_weekly:
                week_halt = True
            if br_dd is not None:
                dd_pt = peak_pts - eq_pts
                if dd_pt > br_dd:
                    dd_active = True
                elif dd_pt < br_dd * br_resume:
                    dd_active = False
            brk_active = bool(day_halt or week_halt or dd_active)
            if brk_active:
                halted_days.add(dkey)

        # ---------------- signal detection (next-bar action) ----------------
        if pos == 0 and pending == 0 and limit_side == 0 and k + 1 < n and np.isfinite(don_hi[k]):
            if c[k] > don_hi[k] and not vlong[k]:
                if brk_active:
                    n_breaker_blocked += 1
                else:
                    n_breakouts += 1
                    if entry_mode == "market":
                        pending = 1
                        pending_signal_bar = k
                    else:
                        limit_side = 1
                        limit_level = float(don_hi[k])
                        limit_until = k + retest_bars
                        limit_signal_bar = k
            elif allow_short and c[k] < don_lo[k] and not vshort[k]:
                if brk_active:
                    n_breaker_blocked += 1
                else:
                    n_breakouts += 1
                    if entry_mode == "market":
                        pending = -1
                        pending_signal_bar = k
                    else:
                        limit_side = -1
                        limit_level = float(don_lo[k])
                        limit_until = k + retest_bars
                        limit_signal_bar = k

    if pos != 0:
        _close(n - 1, c[n - 1], "end")
        equity_usd[n - 1] = realized

    capital = float(c[0]) * pv
    return {
        "equity": pd.Series(1.0 + equity_usd / capital, index=idx, name="equity"),
        "trades": trades,
        "capital": capital,
        "cost": cost,
        "n_breakouts": int(n_breakouts),
        "n_trades": len(trades),
        "n_breaker_blocked": int(n_breaker_blocked),
        "halted_days": int(len(halted_days)),
    }


def run_breakout_trail_1min(
    bars_tf: pd.DataFrame,
    bars1: pd.DataFrame,
    atr_tf: pd.Series,
    cost: CostCfg,
    donch: int = 20,
    stop_mult: float = 2.5,
    veto_long: pd.Series = None,
    allow_short: bool = False,
    freq_minutes: int = 15,
) -> dict:
    """Donchian breakout on ``bars_tf`` decisions, ATR trailing stop executed on
    1min bars. Entry = first 1min of the bar after the signal close + slip; the
    trail stop is refreshed at each decision-bar boundary using the last closed
    ATR and the running favourable extreme, and checked on the 1min lows in
    between. No lookahead: the 1min segment scanned for decision bar ``c`` starts
    only after the signal at ``t < c`` is known.
    """
    idx_tf = bars_tf.index
    t_tf = idx_tf.to_numpy()
    c_tf = bars_tf["close"].to_numpy(float)
    atr = atr_tf.reindex(idx_tf).to_numpy(float)
    n_tf = len(bars_tf)
    don_hi = bars_tf["high"].rolling(donch, min_periods=donch).max().shift(1).to_numpy(float)
    vlong = (veto_long.reindex(idx_tf).fillna(True).to_numpy(bool)
             if veto_long is not None else np.zeros(n_tf, bool))

    idx1 = bars1.index
    t1 = idx1.to_numpy()
    o1 = bars1["open"].to_numpy(float)
    h1 = bars1["high"].to_numpy(float)
    l1 = bars1["low"].to_numpy(float)
    c1 = bars1["close"].to_numpy(float)
    n1 = len(bars1)
    tick = float(cost.tick)
    pv = float(cost.point_value)
    comm = float(cost.commission_per_side)
    slip = float(cost.slip_ticks) * tick

    bar_start = np.searchsorted(t1, t_tf)
    bar_end = np.searchsorted(t1, t_tf + np.timedelta64(int(freq_minutes), "m"))

    trades = []
    n_breakouts = 0
    last_exit = None
    for t in range(n_tf - 1):
        if not np.isfinite(atr[t]) or atr[t] <= 0:
            continue
        if not (c_tf[t] > don_hi[t]) or vlong[t]:
            continue
        n_breakouts += 1
        next_t = idx_tf[t + 1]
        if last_exit is not None and next_t <= last_exit:
            continue
        e1 = int(bar_start[t + 1])
        if e1 >= n1:
            continue
        assert idx1[e1] >= next_t and idx1[e1] > idx_tf[t], "lookahead: fill <= decision"
        entry_fill = o1[e1] + slip
        entry_bar = t + 1
        stop = entry_fill - stop_mult * atr[t]
        water = float(np.max(h1[e1:max(e1 + 1, int(bar_end[entry_bar]))]))
        reason = None
        c = entry_bar
        while True:
            if c >= n_tf:
                exit_raw = c1[n1 - 1] - slip
                exit_time = idx1[n1 - 1]
                reason, xb = "end", n_tf - 1
                break
            s = int(bar_start[c]); e = int(bar_end[c])
            if c == entry_bar:
                s = max(s, e1)
            hit = -1
            if e > s:
                seg = l1[s:e]
                w = np.flatnonzero(seg <= stop)
                if w.size:
                    hit = s + int(w[0])
            if hit >= 0:
                raw = min(stop, o1[hit]) if o1[hit] < stop else stop
                exit_raw = raw - slip
                exit_time = idx1[hit]
                reason, xb = "stop", c
                break
            if e > s:
                water = max(water, float(h1[s:e].max()))
            if c < n_tf - 1 and np.isfinite(atr[c]):
                stop = water - stop_mult * atr[c]
            c += 1
        pnl_pts = exit_raw - entry_fill
        trades.append({
            "entry_time": idx1[e1], "exit_time": exit_time, "side": 1,
            "entry_px": float(entry_fill), "exit_px": float(exit_raw),
            "pnl_pts": float(pnl_pts), "pnl_usd_net": float(pnl_pts * pv - 2.0 * comm),
            "bars_held": int(xb - entry_bar + 1), "atr_entry": float(atr[t]),
            "entry_bar": int(entry_bar), "exit_bar": int(xb), "signal_bar": int(t),
            "exit_reason": reason, "frac": 1.0,
        })
        last_exit = exit_time

    equity_pts = np.full(n_tf, np.nan)
    realized = 0.0
    for tr in trades:
        eb, xb = tr["entry_bar"], tr["exit_bar"]
        for kk in range(eb, xb + 1):
            if kk < xb:
                equity_pts[kk] = realized + (c_tf[kk] - tr["entry_px"])
            else:
                equity_pts[kk] = realized + tr["pnl_pts"]
        realized += tr["pnl_pts"]
    equity_pts = pd.Series(equity_pts, index=idx_tf).ffill().fillna(0.0)
    capital = float(c_tf[0]) * pv
    return {
        "equity": pd.Series(1.0 + equity_pts.to_numpy() / capital, index=idx_tf, name="equity"),
        "trades": trades, "capital": capital, "cost": cost,
        "n_breakouts": int(n_breakouts), "n_trades": len(trades),
    }


def run_structure_stop_ff(
    bars_dec: pd.DataFrame,
    bars_piv: pd.DataFrame,
    bars1min: pd.DataFrame,
    feat: pd.DataFrame,
    signals: pd.Series,
    cost: CostCfg,
    stop_mult: float = 1.5,
    buffer_mult: float = 0.25,
    pivot_n: int = 2,
    freq_minutes: int = 15,
    pivot_freq_minutes: int = 5,
) -> dict:
    """Breakout signals on ``bars_dec``, trailing stop from ``bars_piv`` pivots,
    1min execution. Pivots are confirmed only after their right-side bars close;
    a pivot confirmed inside a decision bar is used from the next decision bar.
    """
    from .features import pivot_low

    idx_dec = bars_dec.index
    o_dec = bars_dec["open"].to_numpy(float)
    c_dec = bars_dec["close"].to_numpy(float)
    n_dec = len(bars_dec)
    atr = feat["atr"].reindex(idx_dec).to_numpy(float)
    sig = signals.reindex(idx_dec).fillna(False).to_numpy(bool)
    tick = float(cost.tick)
    pv = float(cost.point_value)
    comm = float(cost.commission_per_side)

    # latest confirmed pivot price as of each decision bar (causal: confirm time <= bar start)
    flag, price = pivot_low(bars_piv["low"], n=pivot_n)
    flag = flag.to_numpy(bool)
    price = price.to_numpy(float)
    tpiv = bars_piv.index.to_numpy()
    conf_t, conf_p = [], []
    for ci in np.flatnonzero(flag):
        p = price[ci - pivot_n]
        if np.isfinite(p):
            conf_t.append(tpiv[ci])
            conf_p.append(p)
    conf_t = np.array(conf_t)
    conf_p = np.array(conf_p)
    if conf_t.size:
        pos_idx = np.searchsorted(conf_t, idx_dec.to_numpy(), side="right") - 1
        latest_piv = np.where(pos_idx >= 0, conf_p[np.clip(pos_idx, 0, None)], np.nan)
    else:
        latest_piv = np.full(n_dec, np.nan)

    idx1 = bars1min.index
    o1 = bars1min["open"].to_numpy(float)
    c1 = bars1min["close"].to_numpy(float)
    l1 = bars1min["low"].to_numpy(float)
    n1 = len(bars1min)
    t1 = idx1.to_numpy()
    t_dec = idx_dec.to_numpy()
    bar_start = np.searchsorted(t1, t_dec)
    bar_end = np.searchsorted(t1, t_dec + np.timedelta64(int(freq_minutes), "m"))

    trades = []
    last_exit = None
    for t in range(n_dec - 1):
        if not sig[t] or not np.isfinite(atr[t]) or atr[t] <= 0:
            continue
        next_t = idx_dec[t + 1]
        if last_exit is not None and next_t <= last_exit:
            continue
        e1 = int(bar_start[t + 1])
        if e1 >= n1:
            continue
        assert idx1[e1] >= next_t and idx1[e1] > idx_dec[t], "lookahead: fill <= decision"
        entry_fill = o1[e1] + tick
        stop = entry_fill - stop_mult * atr[t]
        entry_hour = t + 1
        reason = None
        c = entry_hour
        while True:
            if c >= n_dec:
                raw = c1[n1 - 1] - tick
                xt = idx1[n1 - 1]
                reason, xb = "end", n_dec - 1
                break
            s = int(bar_start[c])
            e = int(bar_end[c])
            if c == entry_hour:
                s = max(s, e1)
            if e > s:
                seg = l1[s:e]
                hit = seg <= stop
                if hit.any():
                    j = s + int(np.argmax(hit))
                    raw = min(stop, o1[j]) if o1[j] < stop else stop
                    raw = raw - 2.0 * tick
                    xt = idx1[j]
                    reason, xb = "stop", c
                    break
            if np.isfinite(latest_piv[c]) and np.isfinite(atr[c]):
                cand = latest_piv[c] - buffer_mult * atr[c]
                level = np.ceil(cand / tick - 1e-9) * tick
                if level > stop:
                    stop = float(level)
            c += 1
        pnl_pts = raw - entry_fill
        trades.append({
            "entry_time": idx1[e1], "exit_time": xt, "side": 1,
            "entry_px": float(entry_fill), "exit_px": float(raw),
            "pnl_pts": float(pnl_pts), "pnl_usd_net": float(pnl_pts * pv - 2.0 * comm),
            "bars_held": int(xb - entry_hour + 1), "atr_entry": float(atr[t]),
            "entry_bar": int(entry_hour), "exit_bar": int(xb), "signal_bar": int(t),
            "exit_reason": reason,
        })
        last_exit = xt

    equity_usd = np.full(n_dec, np.nan)
    realized = 0.0
    for tr in trades:
        eh, xh = tr["entry_bar"], tr["exit_bar"]
        for kk in range(eh, xh + 1):
            if kk < xh:
                equity_usd[kk] = realized + (c_dec[kk] - tr["entry_px"]) * pv
            else:
                equity_usd[kk] = realized + tr["pnl_usd_net"]
        realized += tr["pnl_usd_net"]
    equity = pd.Series(equity_usd, index=idx_dec).ffill().fillna(0.0)
    capital = float(c_dec[0]) * pv
    equity = pd.Series(1.0 + equity.to_numpy() / capital, index=idx_dec, name="equity")
    return {"equity": equity, "trades": trades, "capital": capital, "cost": cost,
            "n_trades": len(trades)}
