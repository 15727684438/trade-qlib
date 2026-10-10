"""Momentum "impulse push" analytics + second-push prototype (IS only).

Task A: a CAUSAL zigzag swing segmenter (reversal >= threshold from the last
extreme) produces alternating legs; consecutive 5-leg windows form impulse
sequences with pushes P1 (leg0), P2 (leg2), P3 (leg4). We study the empirical
distribution of r2 = P2/P1 (the user's "2nd push length == 1st push length"
question) and r3 = P3/P2 (the "avoid the 3rd push" premise).

Task B: a prototype that buys/sells the break of the push1 extreme (start of
push2) after a correction, targets f*P1, stops beyond the correction extreme
with a 2.0*ATR floor, and takes at most one trade per impulse sequence (never
enters the 3rd push).

Causality: the zigzag confirms a pivot only once price has reversed by the
threshold from that extreme, so every reported value at bar t uses bars <= t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import CostCfg


# --------------------------------------------------------------------------- #
# causal zigzag
# --------------------------------------------------------------------------- #
def causal_zigzag(high, low, atr=None, mode: str = "atr", param: float = 1.5):
    """Confirm alternating swing points causally.

    ``mode="atr"``  : reversal threshold = ``param * ATR[t]`` (ATR at the
                      confirming bar, known at t).
    ``mode="pct"``  : reversal threshold = ``param * ext_p`` (percent of the
                      running extreme price).

    Returns a list of pivots ``(price, bar_index, conf_bar, dir)`` with
    ``dir=+1`` a peak and ``dir=-1`` a trough. A pivot is appended at the bar
    where the reversal threshold is first satisfied.
    """
    hi = np.asarray(high, float)
    lo = np.asarray(low, float)
    at = np.asarray(atr, float) if atr is not None else None
    n = len(hi)
    piv = []
    dir_ = 0
    ext_p = np.nan
    ext_i = -1
    for t in range(n):
        if mode == "atr":
            if at is None or not np.isfinite(at[t]) or at[t] <= 0:
                continue
            thr = float(param) * at[t]
        else:
            if dir_ == 0 or not np.isfinite(ext_p) or ext_p <= 0:
                thr = np.nan
            else:
                thr = float(param) * ext_p  # param already a fraction (e.g. 0.005)
            if not np.isfinite(thr):
                if dir_ == 0:
                    ext_p, ext_i, dir_ = hi[t], t, 1
                continue
        if dir_ == 0:
            ext_p, ext_i, dir_ = hi[t], t, 1
            continue
        if dir_ == 1:
            if hi[t] >= ext_p:
                ext_p, ext_i = hi[t], t
            if lo[t] <= ext_p - thr:
                piv.append((float(ext_p), int(ext_i), int(t), 1))
                dir_ = -1
                ext_p, ext_i = lo[t], t
        else:
            if lo[t] <= ext_p:
                ext_p, ext_i = lo[t], t
            if hi[t] >= ext_p + thr:
                piv.append((float(ext_p), int(ext_i), int(t), -1))
                dir_ = 1
                ext_p, ext_i = hi[t], t
    return piv


def legs_from_pivots(pivots):
    """Alternating legs between consecutive pivots."""
    legs = []
    for i in range(len(pivots) - 1):
        p0, p1 = pivots[i], pivots[i + 1]
        d = 1 if p1[0] > p0[0] else -1
        legs.append({
            "dir": d, "length": float(abs(p1[0] - p0[0])),
            "start_i": int(p0[1]), "end_i": int(p1[1]), "conf_i": int(p1[2]),
            "start_p": float(p0[0]), "end_p": float(p1[0]),
        })
    return legs


def collect_sequences(legs, atrp=None):
    """All consecutive 5-leg windows -> push triples (P1,P2,P3)."""
    seqs = []
    for i in range(len(legs) - 4):
        L = legs[i:i + 5]
        d = L[0]["dir"]
        P1, P2, P3 = L[0]["length"], L[2]["length"], L[4]["length"]
        if P1 <= 0 or P2 <= 0:
            continue
        si = L[0]["start_i"]
        seqs.append({
            "start_i": si, "dir": d, "P1": P1, "P2": P2, "P3": P3,
            "r2": P2 / P1, "r3": P3 / P2,
            "atrp": (float(atrp[si]) if atrp is not None and si < len(atrp)
                     else np.nan),
        })
    return seqs


_TOLS = (0.05, 0.10, 0.15, 0.20, 0.25)


def summarize(seqs):
    """Distribution of r2 / r3 (+ equality probabilities)."""
    out = {"n_seq": len(seqs)}
    r2 = np.array([s["r2"] for s in seqs if np.isfinite(s["r2"])], float)
    r3 = np.array([s["r3"] for s in seqs if np.isfinite(s["r3"])], float)
    if r2.size:
        out.update({
            "r2_median": float(np.median(r2)), "r2_q1": float(np.percentile(r2, 25)),
            "r2_q3": float(np.percentile(r2, 75)), "r2_mean": float(r2.mean()),
            "p_r2_ge1": float((r2 >= 1).mean()), "p_r2_le1": float((r2 <= 1).mean()),
        })
        for tol in _TOLS:
            out[f"p_eq_{int(tol*100):03d}"] = float((np.abs(r2 - 1) <= tol).mean())
    if r3.size:
        out.update({
            "r3_median": float(np.median(r3)), "r3_q1": float(np.percentile(r3, 25)),
            "r3_q3": float(np.percentile(r3, 75)), "r3_mean": float(r3.mean()),
            "p_r3_ge1": float((r3 >= 1).mean()), "p_r3_le1": float((r3 <= 1).mean()),
            "p_r3_eq_015": float((np.abs(r3 - 1) <= 0.15).mean()),
        })
    return out


# --------------------------------------------------------------------------- #
# Task B — second-push prototype (15min decisions / 1min fills)
# --------------------------------------------------------------------------- #
def run_second_push(bars15, bars1, feat15, pivots, cost: CostCfg, *,
                    f: float = 1.0, stop_atr_floor: float = 2.0,
                    freq_minutes: int = 15, slip: float = None) -> dict:
    """Enter the break of the push1 extreme (start of push2); target f*P1; stop
    beyond the correction extreme with a >= ``stop_atr_floor``*ATR risk floor.
    At most one trade per impulse sequence (never enters the 3rd push)."""
    idx15 = bars15.index
    atr15 = feat15["atr"].reindex(idx15).to_numpy(float)
    n15 = len(bars15)
    tick = float(cost.tick)
    pv = float(cost.point_value)
    comm = float(cost.commission_per_side)
    slip = float(cost.slip_ticks) * tick if slip is None else float(slip)

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

    legs = legs_from_pivots(pivots)
    nleg = len(legs)
    trades = []
    reasons = {"target": 0, "stop": 0, "end": 0}
    last_exit = None
    i = 0
    while i <= nleg - 5:
        blk = legs[i:i + 5]
        push = blk[0]
        corr = blk[1]
        d = push["dir"]
        P1 = push["length"]
        B_price = push["end_p"]
        C_price = corr["end_p"]
        B_conf = push["conf_i"]
        deadline = blk[2]["conf_i"]        # entry must be during push2
        if B_conf + 1 > deadline:
            i += 1
            continue

        # ---- entry: first 1min bar (from B_conf+1) breaking B in direction d
        entry = None
        for k in range(B_conf + 1, min(deadline, n15 - 1) + 1):
            s, e = int(bs[k]), int(be[k])
            if e <= s:
                continue
            if d > 0:
                hit = np.flatnonzero(h1[s:e] > B_price)
            else:
                hit = np.flatnonzero(l1[s:e] < B_price)
            if hit.size:
                j = s + int(hit[0])
                raw = max(B_price, o1[j]) if d > 0 else min(B_price, o1[j])
                entry = (k, j, raw + d * slip)
                break
        if entry is None:
            i += 1
            continue
        k_e, j_e, entry_fill = entry
        entry_time = idx1[j_e]
        if last_exit is not None and entry_time <= last_exit:
            i += 1
            continue

        atr_e = atr15[k_e]
        if d > 0:
            stop = min(C_price - tick, entry_fill - stop_atr_floor * atr_e)
            target = entry_fill + f * P1
        else:
            stop = max(C_price + tick, entry_fill + stop_atr_floor * atr_e)
            target = entry_fill - f * P1

        # ---- exit scan (stop first, conservative)
        reason = None
        exit_time = exit_raw = None
        exit_bar_k = n15 - 1
        for k in range(k_e, n15):
            s, e = int(bs[k]), int(be[k])
            if k == k_e:
                s = max(s, j_e + 1)
            if e <= s:
                continue
            if d > 0:
                sh = np.flatnonzero(l1[s:e] <= stop)
                th = np.flatnonzero(h1[s:e] >= target)
            else:
                sh = np.flatnonzero(h1[s:e] >= stop)
                th = np.flatnonzero(l1[s:e] <= target)
            js = (s + int(sh[0])) if sh.size else None
            jt = (s + int(th[0])) if th.size else None
            if js is not None and (jt is None or js <= jt):
                raw = min(stop, o1[js]) if d > 0 else max(stop, o1[js])
                exit_raw = raw - d * slip
                exit_time = idx1[js]
                reason = "stop"
                exit_bar_k = k
                break
            if jt is not None:
                exit_raw = target - d * slip
                exit_time = idx1[jt]
                reason = "target"
                exit_bar_k = k
                break
        if reason is None:
            exit_raw = c1[n1 - 1] - d * slip
            exit_time = idx1[n1 - 1]
            reason = "end"
            exit_bar_k = n15 - 1

        pnl_pts = d * (exit_raw - entry_fill)
        pnl_usd = pnl_pts * pv - 2.0 * comm
        trades.append({
            "entry_time": entry_time, "exit_time": exit_time, "side": int(d),
            "entry_px": float(entry_fill), "exit_px": float(exit_raw),
            "stop_px": float(stop), "target_px": float(target),
            "P1": float(P1), "entry_bar": int(k_e), "exit_bar": int(exit_bar_k),
            "pnl_pts": float(pnl_pts), "pnl_usd_net": float(pnl_usd),
            "bars_held": int(exit_bar_k - k_e),
            "atr_entry": float(atr_e), "exit_reason": reason,
        })
        reasons[reason] += 1
        last_exit = exit_time
        i += 5  # skip the rest of this impulse sequence (never trade the 3rd push)

    # equity @1 contract (mark-to-market at 15min closes on held positions)
    c15 = bars15["close"].to_numpy(float)
    equity_usd = np.zeros(n15)
    realized = 0.0
    for tr in trades:
        kb, kx = tr["entry_bar"], tr["exit_bar"]
        for k in range(kb, kx + 1):
            if k < kx:
                equity_usd[k] = realized + tr["side"] * (c15[k] - tr["entry_px"]) * pv
            else:
                equity_usd[k] = realized + tr["pnl_usd_net"]
        realized += tr["pnl_usd_net"]
    capital = float(c15[0]) * pv
    return {
        "equity": pd.Series(1.0 + equity_usd / capital, index=idx15),
        "trades": trades, "capital": capital, "cost": cost,
        "reasons": reasons, "n_seq_legs": nleg,
    }
