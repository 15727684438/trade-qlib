"""Measured-Move (MM) study on approximately-equal-length TREND candle
combinations of >= 3 consecutive same-direction candles.

Study question: after a run of >=3 consecutive same-direction candles whose
BODIES are approximately equal, does price reach a target of DOUBLE the
combination's body distance (the "Measured Move")?

Definitions implemented here
----------------------------
* Combination = a maximal same-direction run of candles (all bullish or all
  bearish) that is then greedily segmented so every body stays within ``tol``
  of the running mean body of the current segment. Only segments of length
  ``>= min_len`` (default 3) are kept. This yields non-overlapping segments of
  length 3, 4, 5+ inside each directional run.
* Body = |close - open|.
* D      = dir * (close[last] - open[first])   (body distance, gap inclusive).
* D_sum  = sum of the individual candle bodies.
* MM target = close[last] + dir*D  (equivalently first_open + 2*D for a gap-free
  run); a D_sum variant target = close[last] + dir*D_sum is also reported.

Causality
---------
A combination is only known at the CLOSE of its last candle. Every forward
measurement therefore starts on the NEXT bar (index ``end+1``): the reference
price is ``open[end+1]`` and the forward window is ``[end+1, end+N]``. No
future information is used to identify a combination.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# combination extraction
# --------------------------------------------------------------------------- #
def extract_combos(open_, close, tol: float, min_len: int = 3) -> pd.DataFrame:
    """Causal extraction of equal-body trend combinations.

    Parameters
    ----------
    open_, close : array-likes of the executed (IS-sliced) bars.
    tol : float
        Each candle body must satisfy ``|body - mean_body| <= tol * mean_body``
        where ``mean_body`` is the running mean of the current segment.
    min_len : int
        Minimum number of candles in a reported combination.

    Returns
    -------
    DataFrame with columns ``start, end, dir, n, D, D_sum, mean_body`` (indices
    are integer bar positions).
    """
    o = np.asarray(open_, dtype=float)
    c = np.asarray(close, dtype=float)
    n = len(o)
    cols = ["start", "end", "dir", "n", "D", "D_sum", "mean_body"]
    if n < min_len:
        return pd.DataFrame({k: [] for k in cols})

    body = np.abs(c - o)
    sgn = np.sign(c - o).astype(np.int8)
    # run boundaries: change of sign (0 = doji breaks a run)
    bnd = np.flatnonzero(sgn[1:] != sgn[:-1]) + 1
    starts = np.concatenate(([0], bnd))
    ends = np.concatenate((bnd, [n]))

    rows = []
    for s, e in zip(starts.tolist(), ends.tolist()):
        d = int(sgn[s])
        if d == 0 or (e - s) < min_len:
            continue
        bs = body[s:e]
        seg_s = s
        cur_sum = float(bs[0])
        cur_n = 1
        for k in range(1, bs.shape[0]):
            m = cur_sum / cur_n
            b = float(bs[k])
            if abs(b - m) <= tol * m:
                cur_sum += b
                cur_n += 1
            else:
                if cur_n >= min_len:
                    rows.append((seg_s, seg_s + cur_n - 1, cur_sum, cur_n))
                seg_s = s + k
                cur_sum = b
                cur_n = 1
        if cur_n >= min_len:
            rows.append((seg_s, seg_s + cur_n - 1, cur_sum, cur_n))
    if not rows:
        return pd.DataFrame({k: [] for k in cols})

    arr = np.asarray(rows, dtype=float)  # start, end, sum_body, n
    st = arr[:, 0].astype(np.int64)
    en = arr[:, 1].astype(np.int64)
    nn = arr[:, 3].astype(np.int64)
    dirs = sgn[st].astype(np.int64)
    D = dirs * (c[en] - o[st])
    out = pd.DataFrame({
        "start": st, "end": en, "dir": dirs, "n": nn,
        "D": D, "D_sum": arr[:, 2], "mean_body": arr[:, 2] / nn,
    })
    return out


# --------------------------------------------------------------------------- #
# forward measurement (vectorised over events)
# --------------------------------------------------------------------------- #
def measure_forward(o, h, l, c, starts, ends, dirs, D, D_sum,
                    Ns=(5, 10, 20, 50), n_excursion: int = 20) -> dict:
    """Measure the forward path of each combination starting on the NEXT bar.

    Returns a dict of per-event arrays (same order as the inputs):
      hit{N}_body / hit{N}_sum : bool, target reached within N bars
      t_body / t_sum           : bars to first target hit (NaN if none <= max N)
      adverse                  : bool, start level touched within n_excursion
      adverse_before           : bool, start level touched before the body target
                                 (same-bar tie -> adverse, conservative)
      mfe / mae                : max favourable / adverse excursion in pts
                                 within ``n_excursion`` bars relative to ref
    """
    o = np.asarray(o, float); h = np.asarray(h, float)
    l = np.asarray(l, float); c = np.asarray(c, float)
    starts = np.asarray(starts, np.int64)
    ends = np.asarray(ends, np.int64)
    dirs = np.asarray(dirs, np.int64)
    D = np.asarray(D, float); D_sum = np.asarray(D_sum, float)
    n = len(c)
    ne = len(ends)
    long = dirs > 0
    maxN = int(max(Ns))

    ok = (ends + 1) < n
    safe = np.minimum(ends + 1, n - 1)
    ref = np.where(ok, o[safe], np.nan)
    start_lvl = o[starts]
    tgt_body = c[ends] + dirs * D
    tgt_sum = c[ends] + dirs * D_sum

    reached_b = np.zeros(ne, bool); tb = np.full(ne, np.nan)
    reached_s = np.zeros(ne, bool); ts = np.full(ne, np.nan)
    tadv = np.full(ne, np.nan)
    mfe = np.zeros(ne); mae = np.zeros(ne)
    snap_b = {}; snap_s = {}

    for k in range(1, maxN + 1):
        j = ends + k
        valid = j < n
        jj = np.where(valid, j, 0)
        hi = np.where(valid, h[jj], -np.inf)
        lo = np.where(valid, l[jj], np.inf)
        fav_b = np.where(long, hi >= tgt_body, lo <= tgt_body)
        fav_s = np.where(long, hi >= tgt_sum, lo <= tgt_sum)
        adv = np.where(long, lo <= start_lvl, hi >= start_lvl)

        new_b = fav_b & ~reached_b
        tb = np.where(new_b & np.isnan(tb), float(k), tb)
        reached_b |= fav_b
        new_s = fav_s & ~reached_s
        ts = np.where(new_s & np.isnan(ts), float(k), ts)
        reached_s |= fav_s
        tadv = np.where(adv & np.isnan(tadv), float(k), tadv)

        if k <= n_excursion:
            fin = np.isfinite(ref)
            fav_exc = np.where(long, hi - ref, ref - lo)
            adv_exc = np.where(long, ref - lo, hi - ref)
            mfe = np.maximum(mfe, np.where(fin & np.isfinite(fav_exc), fav_exc, 0.0))
            mae = np.maximum(mae, np.where(fin & np.isfinite(adv_exc), adv_exc, 0.0))

        if k in Ns:
            snap_b[k] = reached_b.copy()
            snap_s[k] = reached_s.copy()

    mfe = np.where(ok, np.maximum(mfe, 0.0), np.nan)
    mae = np.where(ok, np.maximum(mae, 0.0), np.nan)
    tb = np.where(ok, tb, np.nan)
    ts = np.where(ok, ts, np.nan)
    tadv = np.where(ok, tadv, np.nan)

    # adverse before target within n_excursion (tie -> adverse)
    tb_exc = np.where(np.isfinite(tb) & (tb <= n_excursion), tb, np.inf)
    adverse_exc = np.isfinite(tadv) & (tadv <= n_excursion)
    adverse_before = adverse_exc & (tadv <= tb_exc)

    out = {
        "ref": ref, "start_lvl": start_lvl,
        "target_body": tgt_body, "target_sum": tgt_sum,
        "t_body": tb, "t_sum": ts, "t_adv": tadv,
        "adverse": adverse_exc, "adverse_before": adverse_before,
        "mfe": mfe, "mae": mae, "ok": ok,
    }
    for N in Ns:
        out[f"hit{N}_body"] = snap_b.get(N, np.zeros(ne, bool))
        out[f"hit{N}_sum"] = snap_s.get(N, np.zeros(ne, bool))
    return out


# --------------------------------------------------------------------------- #
# base rate (random-bar forward moves)
# --------------------------------------------------------------------------- #
def forward_move_samples(h, l, o, anchors, N):
    """Sorted forward up/down moves over the next ``N`` bars for ``anchors``.

    ``anchors`` hold bar indices ``t`` for which bars ``t+1..t+N`` exist. The
    forward reference is ``open[t+1]`` (same convention as the events).
    """
    anchors = np.asarray(anchors, np.int64)
    fmax = np.full(anchors.shape[0], -np.inf)
    fmin = np.full(anchors.shape[0], np.inf)
    for k in range(1, int(N) + 1):
        np.maximum(fmax, h[anchors + k], out=fmax)
        np.minimum(fmin, l[anchors + k], out=fmin)
    o1 = o[anchors + 1]
    return np.sort(fmax - o1), np.sort(o1 - fmin)


def base_prob(sorted_moves, required):
    """P(random forward move >= required) using a sorted sample array."""
    m = sorted_moves.shape[0]
    if m == 0:
        return np.full(np.asarray(required, float).shape, np.nan)
    req = np.asarray(required, float)
    idx = np.searchsorted(sorted_moves, req, side="left")
    return (m - idx) / float(m)


# --------------------------------------------------------------------------- #
# stratifiers
# --------------------------------------------------------------------------- #
def tercile_labels(values):
    """Return (labels, bounds) where labels in {low, mid, high} by terciles."""
    v = np.asarray(values, float)
    fin = np.isfinite(v)
    if fin.sum() < 3:
        return np.array(["na"] * len(v)), (np.nan, np.nan)
    q1, q2 = np.quantile(v[fin], [1 / 3, 2 / 3])
    lab = np.full(len(v), "na", dtype=object)
    lab[fin & (v <= q1)] = "low"
    lab[fin & (v > q1) & (v <= q2)] = "mid"
    lab[fin & (v > q2)] = "high"
    return lab, (float(q1), float(q2))


def session_labels(index):
    """Asia / Europe / US / Post session by hour of the (ET, tz-naive) bar."""
    hh = index.hour
    lab = np.full(len(index), "na", dtype=object)
    lab[((hh >= 18) | (hh < 2))] = "Asia"
    lab[(hh >= 2) & (hh < 9)] = "Europe"
    lab[(hh >= 9) & (hh < 16)] = "US"
    lab[(hh >= 16) & (hh < 18)] = "Post"
    return lab


def n_candle_bucket(n):
    n = np.asarray(n)
    out = np.full(len(n), "5+", dtype=object)
    out[n == 3] = "3"
    out[n == 4] = "4"
    return out
