"""No-lookahead event backtester over 60min decisions.

Contract
--------
* A target position is *known at each bar's CLOSE*.
* Fills happen at the *NEXT bar's OPEN* (execution index == decision index + 1).
* Long-only by default; a ``-1`` target is treated as flat unless
  ``allow_short=True``.
* Slippage is applied adversely on both entry and exit; commission is charged
  per side per contract.

MNQ contract: tick = 0.25 points ($0.50), point_value = $2.00/point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CostCfg:
    slip_ticks: float = 1.5
    commission_per_side: float = 0.75
    tick: float = 0.25
    point_value: float = 2.0


def _clean_ohlc(bars60: pd.DataFrame):
    df = bars60[["open", "high", "low", "close"]].astype(float)
    return df.index, (
        df["open"].to_numpy(),
        df["high"].to_numpy(),
        df["low"].to_numpy(),
        df["close"].to_numpy(),
    )


def decide_to_positions(desired: np.ndarray) -> np.ndarray:
    """Map decision-time targets to per-bar held positions (execution at t+1)."""
    desired = np.asarray(desired, dtype=int)
    n = len(desired)
    positions = np.zeros(n, dtype=int)
    if n > 1:
        positions[1:] = desired[:-1]
    # explicit no-lookahead guard: execution index = decision index + 1
    assert np.array_equal(positions[1:], desired[:-1]), "lookahead: fill != decision+1"
    return positions


def run_target_position(
    bars60: pd.DataFrame,
    target: pd.Series,
    cost: CostCfg,
    allow_short: bool = False,
    atr: pd.Series = None,
) -> dict:
    """Backtest a decision-time ``target`` series with next-open execution."""
    idx, (o, h, l, c) = _clean_ohlc(bars60)
    n = len(bars60)
    pv = float(cost.point_value)
    slip_pts = float(cost.slip_ticks) * float(cost.tick)

    desired = target.reindex(idx).fillna(0).to_numpy(dtype=float)
    desired = np.clip(np.nan_to_num(desired), -1, 1).astype(int)
    if not allow_short:
        desired = np.where(desired > 0, 1, 0)
    positions = decide_to_positions(desired)

    # per-bar mark-to-market (position transitions at the open)
    pnl_usd = np.zeros(n)
    cost_usd = np.zeros(n)
    for k in range(1, n):
        pts = positions[k - 1] * (o[k] - c[k - 1]) + positions[k] * (c[k] - o[k])
        pnl_usd[k] = pts * pv
        dpos = abs(int(positions[k]) - int(positions[k - 1]))
        cost_usd[k] = dpos * (cost.commission_per_side + slip_pts * pv)
    if n and positions[-1] != 0:  # flatten at last close
        cost_usd[-1] += abs(int(positions[-1])) * (cost.commission_per_side + slip_pts * pv)

    equity_usd = np.cumsum(pnl_usd - cost_usd)
    capital = float(c[0]) * pv
    equity = pd.Series(1.0 + equity_usd / capital, index=idx, name="equity")

    # trade ledger
    trades = []
    cur = None
    for k in range(1, n):
        if positions[k] == positions[k - 1]:
            continue
        if positions[k - 1] != 0 and cur is not None:
            trades.append(_make_trade(cur, k, idx[k], o[k], positions[k - 1], cost, slip_pts, pv, atr))
            cur = None
        if positions[k] != 0:
            cur = {
                "entry_time": idx[k],
                "entry_bar": k,
                "entry_raw": float(o[k]),
                "side": int(positions[k]),
            }
    if cur is not None:
        trades.append(_make_trade(cur, n - 1, idx[n - 1], c[n - 1], positions[-1], cost, slip_pts, pv, atr))

    return {
        "equity": equity,
        "trades": trades,
        "positions": pd.Series(positions, index=idx, name="position"),
        "capital": capital,
        "cost": cost,
    }


def _make_trade(cur, exit_bar, exit_time, exit_raw, side, cost, slip_pts, pv, atr):
    entry_raw = cur["entry_raw"]
    gross_pts = float(side) * (exit_raw - entry_raw)
    pnl_pts = gross_pts - 2.0 * slip_pts
    pnl_usd_net = pnl_pts * pv - 2.0 * cost.commission_per_side
    atr_entry = float("nan")
    if atr is not None:
        try:
            atr_entry = float(atr.iloc[cur["entry_bar"]])
        except Exception:
            atr_entry = float("nan")
    return {
        "entry_time": cur["entry_time"],
        "exit_time": exit_time,
        "side": int(side),
        "entry_px": entry_raw + float(side) * slip_pts,
        "exit_px": exit_raw - float(side) * slip_pts,
        "pnl_pts": float(pnl_pts),
        "pnl_usd_net": float(pnl_usd_net),
        "bars_held": int(exit_bar - cur["entry_bar"]),
        "atr_entry": atr_entry,
    }


def buy_and_hold(bars60: pd.DataFrame, cost: CostCfg, atr: pd.Series = None) -> dict:
    """B0: long one contract from the first bar's close to the last bar's close."""
    idx, (o, h, l, c) = _clean_ohlc(bars60)
    n = len(bars60)
    pv = float(cost.point_value)
    slip_pts = float(cost.slip_ticks) * float(cost.tick)

    first, last = float(c[0]), float(c[-1])
    # start at equity 1.0; charge both legs of the round-trip cost at the final bar
    equity = pd.Series(1.0 + (c / first - 1.0), index=idx, name="equity")
    equity.iloc[-1] -= (2.0 * cost.commission_per_side + 2.0 * slip_pts * pv) / (first * pv)

    atr_entry = float("nan")
    if atr is not None and len(atr):
        atr_entry = float(atr.iloc[0])
    trade = {
        "entry_time": idx[0],
        "exit_time": idx[-1],
        "side": 1,
        "entry_px": first + slip_pts,
        "exit_px": last - slip_pts,
        "pnl_pts": (last - first) - 2.0 * slip_pts,
        "pnl_usd_net": (last - first) * pv - 2.0 * cost.commission_per_side - 2.0 * slip_pts * pv,
        "bars_held": n - 1,
        "atr_entry": atr_entry,
    }
    return {
        "equity": equity,
        "trades": [trade],
        "positions": pd.Series(np.ones(n, dtype=int), index=idx, name="position"),
        "capital": first * pv,
        "cost": cost,
    }


def metrics(equity, trades, bar_index, cost: CostCfg, periods_per_year: float) -> dict:
    """Performance metrics. ``equity`` is a normalised per-bar curve (~1.0 base)."""
    eq = np.asarray(equity, dtype=float)
    n = len(eq)
    start, end = float(eq[0]), float(eq[-1])
    total_return = end / start - 1.0 if start != 0 else float("nan")

    if end > 0 and n > 0 and periods_per_year > 0:
        cagr = (end / start) ** (periods_per_year / n) - 1.0
    else:
        cagr = float("nan")

    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    max_dd = float(-np.min(dd)) if n else float("nan")
    mar = float(cagr / max_dd) if (np.isfinite(cagr) and max_dd > 0) else float("nan")

    pnl = np.array([t["pnl_usd_net"] for t in trades], dtype=float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_win = float(wins.sum())
    gross_loss = float(losses.sum())
    profit_factor = float(gross_win / abs(gross_loss)) if gross_loss != 0 else float("inf")
    n_trades = int(len(pnl))
    hit_rate = float(len(wins) / n_trades) if n_trades else float("nan")
    avg_win = float(wins.mean()) if wins.size else float("nan")
    avg_loss = float(losses.mean()) if losses.size else float("nan")

    bars_held = np.array([t.get("bars_held", 0) for t in trades], dtype=float)
    time_in_market = float(bars_held.sum() / n) if n else float("nan")

    # expectancy in R: R = ATR at entry when available, else average losing pts
    pts = np.array([t["pnl_pts"] for t in trades], dtype=float)
    atr_e = np.array([t.get("atr_entry", np.nan) for t in trades], dtype=float)
    loss_pts_all = np.array([t["pnl_pts"] for t in trades if t["pnl_pts"] < 0], dtype=float)
    avg_loss_pts = float(np.abs(loss_pts_all).mean()) if loss_pts_all.size else np.nan
    if n_trades:
        r = np.where(np.isfinite(atr_e) & (atr_e > 0), atr_e, avg_loss_pts)
        with np.errstate(divide="ignore", invalid="ignore"):
            valid = np.isfinite(r) & (r > 0)
            expectancy_r = float(np.mean(pts[valid] / r[valid])) if valid.any() else float("nan")
    else:
        expectancy_r = float("nan")

    return {
        "CAGR": cagr,
        "MaxDD": max_dd,
        "MAR": mar,
        "profit_factor": profit_factor,
        "hit_rate": hit_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "n_trades": n_trades,
        "time_in_market": time_in_market,
        "expectancy_R": expectancy_r,
        "total_return": total_return,
    }


# --------------------------------------------------------------------------- #
# M2: structure-low trailing stop executed on 1min bars
# --------------------------------------------------------------------------- #
def run_structure_stop(
    bars60: pd.DataFrame,
    bars1min: pd.DataFrame,
    feat: pd.DataFrame,
    signals: pd.Series,
    cost: CostCfg,
    stop_mult: float = 1.5,
    buffer_mult: float = 0.25,
    pivot_n: int = 2,
    allow_short: bool = False,
    freq_minutes: int = 60,
) -> dict:
    """T1' backtest: decision-bar pullback-resumption, 1min execution.

    The decision frame ``bars60`` may be any intraday freq; ``freq_minutes``
    gives its bar length in minutes (60 for the legacy 60min run, 15 for the
    15min decision run). All decision inputs are known at the close of ``t``.

    - Decision at the CLOSE of decision bar ``t``; entry fill = open of the
      FIRST 1min bar of bar ``t+1`` plus one adverse tick.
    - Initial stop = fill - stop_mult*ATR14(t) (long).
    - Ratchet on confirmed pivot low ``c``: candidate = pivot_price -
      buffer_mult*ATR14(c); new stop = max(old, ceil(candidate/tick)*tick);
      never loosens.
    - Intrabar stop scan on 1min lows; fill = min(stop, bar_open) - 2 ticks.
    - Context exit: decision close < EMA50 at bar c -> first 1min open of bar
      c+1 minus one tick.
    - Gaps handled naturally via the first 1min bar after the gap.
    """
    if allow_short:
        raise NotImplementedError("structure-stop short side not implemented (long-only system)")

    from .features import pivot_low

    idx60 = bars60.index
    o60 = bars60["open"].to_numpy(float)
    c60 = bars60["close"].to_numpy(float)
    n60 = len(bars60)
    atr = feat["atr"].reindex(idx60).to_numpy(float)
    ema50 = feat["ema50"].reindex(idx60).to_numpy(float)
    sig = signals.reindex(idx60).fillna(False).to_numpy(bool)

    tick = float(cost.tick)
    pv = float(cost.point_value)
    comm = float(cost.commission_per_side)

    piv_flag, piv_price = pivot_low(bars60["low"], n=pivot_n)
    piv_flag = piv_flag.to_numpy(bool)
    piv_price = piv_price.to_numpy(float)

    idx1 = bars1min.index
    o1 = bars1min["open"].to_numpy(float)
    c1 = bars1min["close"].to_numpy(float)
    l1 = bars1min["low"].to_numpy(float)
    n1 = len(bars1min)
    t1 = idx1.to_numpy()
    t60 = idx60.to_numpy()
    hour_start = np.searchsorted(t1, t60)
    hour_end = np.searchsorted(t1, t60 + np.timedelta64(int(freq_minutes), "m"))

    trades = []
    last_exit_time = None
    for t in range(n60 - 1):
        if not sig[t] or not np.isfinite(atr[t]) or atr[t] <= 0:
            continue
        next_hour_time = idx60[t + 1]
        if last_exit_time is not None and next_hour_time <= last_exit_time:
            continue  # no overlapping position
        e1 = int(hour_start[t + 1])
        if e1 >= n1:
            continue
        assert idx1[e1] >= next_hour_time and idx1[e1] > idx60[t], "lookahead: fill <= decision"

        entry_fill = o1[e1] + tick
        stop = entry_fill - stop_mult * atr[t]
        entry_hour = t + 1
        entry_time = idx1[e1]

        reason = None
        exit_time = None
        exit_raw = None
        exit_bar60 = None
        c = entry_hour
        while True:
            if c >= n60:
                exit_raw = c1[n1 - 1] - tick
                exit_time = idx1[n1 - 1]
                reason, exit_bar60 = "end", n60 - 1
                break
            s = int(hour_start[c])
            e = int(hour_end[c])
            if c == entry_hour:
                s = max(s, e1)
            if e > s:
                seg = l1[s:e]
                hit = seg <= stop
                if hit.any():
                    j = s + int(np.argmax(hit))
                    raw = min(stop, o1[j]) if o1[j] < stop else stop
                    exit_raw = raw - 2.0 * tick
                    exit_time = idx1[j]
                    reason, exit_bar60 = "stop", c
                    break
            if piv_flag[c] and np.isfinite(piv_price[c - pivot_n]) and np.isfinite(atr[c]):
                cand = piv_price[c - pivot_n] - buffer_mult * atr[c]
                level = np.ceil(cand / tick - 1e-9) * tick
                if level > stop:
                    stop = float(level)
            if np.isfinite(ema50[c]) and c60[c] < ema50[c]:
                if c + 1 < n60 and hour_start[c + 1] < n1:
                    ne = int(hour_start[c + 1])
                    exit_raw = o1[ne] - tick
                    exit_time = idx1[ne]
                    reason, exit_bar60 = "context", c + 1
                    break
                exit_raw = c1[n1 - 1] - tick
                exit_time = idx1[n1 - 1]
                reason, exit_bar60 = "end", n60 - 1
                break
            c += 1

        pnl_pts = exit_raw - entry_fill
        pnl_usd = pnl_pts * pv - 2.0 * comm
        trades.append({
            "entry_time": entry_time, "exit_time": exit_time, "side": 1,
            "entry_px": float(entry_fill), "exit_px": float(exit_raw),
            "pnl_pts": float(pnl_pts), "pnl_usd_net": float(pnl_usd),
            "bars_held": int(exit_bar60 - entry_hour + 1),
            "atr_entry": float(atr[t]) if np.isfinite(atr[t]) else np.nan,
            "entry_bar60": int(entry_hour), "exit_bar60": int(exit_bar60),
            "entry_fill": float(entry_fill), "exit_reason": reason,
            "stop_final": float(stop),
        })
        last_exit_time = exit_time

    # per-60min mark-to-market equity (position marked at each held 60min close)
    equity_usd = np.full(n60, np.nan)
    realized = 0.0
    for tr in trades:
        eh, xh = tr["entry_bar60"], tr["exit_bar60"]
        for k in range(eh, xh + 1):
            if k < xh:
                equity_usd[k] = realized + (c60[k] - tr["entry_fill"]) * pv
            else:
                equity_usd[k] = realized + tr["pnl_usd_net"]
        realized += tr["pnl_usd_net"]
    equity = pd.Series(equity_usd, index=idx60).ffill().fillna(0.0)
    capital = float(c60[0]) * pv
    equity = pd.Series(1.0 + equity.to_numpy() / capital, index=idx60, name="equity")

    return {
        "equity": equity,
        "trades": trades,
        "capital": capital,
        "cost": cost,
    }
