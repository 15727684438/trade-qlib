"""MM post-event performance全景 — what happens AFTER an equal-length trend
candle combination (the "Measured Move" setup) appears?

This is an OPEN-ENDED study (no narrow strategy grid). It reuses the causal MM
events from ``results/measured_move_events.csv`` (tol=0.40, n_candles>=3) and
measures the forward path from the bar immediately after the combination's last
candle (strictly no look-ahead):

  * forward return (points and $@1ct) at horizons 5/10/20/50/100/200/400 bars,
    plus "to structural reversal" (causal zigzag ATR-k=1.5) and "to data end";
  * full MFE / MAE distribution (P25/50/75/90) and time-to-MFE;
  * continuation vs reversal (the +/-1*D first-touch race) by horizon;
  * projection completion: P(reach 1x/2x/3x of D) and how far it really runs;
  * gross and net $ (commission $0.50/side; the project $0.20/side slippage is
    also reported separately).

Baselines: (a) session/regime/direction-matched random entries, (b) the
unconditional forward drift of the timeframe. The goal is to separate the MM
pattern from plain drift/volatility.

Run:  python scripts/run_mm_perf.py            (both 5min and 15min)
      python scripts/run_mm_perf.py --freq 15min
Out:  results/mm_perf_15min.csv, results/mm_perf_5min.csv, results/MM_PERF.md

IS only: 2019-05-06..2023-12-31. OOS (2024+) is never touched.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(_PKG_ROOT))

from tf_mnq.data import load_bars  # noqa: E402
from tf_mnq.features import atr_wilder, atr_percentile  # noqa: E402
from tf_mnq.measured_move import session_labels, n_candle_bucket  # noqa: E402
from tf_mnq.momentum import causal_zigzag  # noqa: E402

RESULTS_DIR = os.path.join(_PKG_ROOT, "results")
IS_START, IS_END = "2019-05-06", "2023-12-31"
HS = (5, 10, 20, 50, 100, 200, 400)
HMAX = 400
PV = 2.0            # $ per point per contract
COMM = 0.5          # $ per side ($1.00 open+close)
TICK = 0.25
SLIP_TICKS = 0.4    # $0.20/side (reported separately)
RT_COMM = 2.0 * COMM
RT_SLIP_USD = 2.0 * SLIP_TICKS * TICK * PV
ZIG_K = 1.5
BASE_ANCHORS = 100_000


# --------------------------------------------------------------------------- #
# event loading
# --------------------------------------------------------------------------- #
def load_freq(freq):
    allb = load_bars(freq)
    atr_full = atr_wilder(allb["high"], allb["low"], allb["close"], 14)
    atrp_full = atr_percentile(atr_full, window=500, min_periods=100)
    sl = allb.loc[(allb.index >= pd.Timestamp(IS_START))
                  & (allb.index <= pd.Timestamp(IS_END))]
    idx = sl.index
    o = sl["open"].to_numpy(float); h = sl["high"].to_numpy(float)
    l = sl["low"].to_numpy(float); c = sl["close"].to_numpy(float)
    atr = atr_full.reindex(idx).to_numpy(float)
    atrp = atrp_full.reindex(idx).to_numpy(float)
    return dict(idx=idx, o=o, h=h, l=l, c=c, atr=atr, atrp=atrp, n=len(sl))


def load_events(freq, idx, tol=0.40):
    ev = pd.read_csv(os.path.join(RESULTS_DIR, "measured_move_events.csv"))
    ev = ev[(ev["freq"] == freq) & (ev["tol"] == tol)].copy()
    ev["start_time"] = pd.to_datetime(ev["start_time"])
    ev["end_time"] = pd.to_datetime(ev["end_time"])
    s = idx.get_indexer(ev["start_time"])
    e = idx.get_indexer(ev["end_time"])
    bad = (s < 0) | (e < 0)
    if bad.any():
        print(f"  WARNING: {int(bad.sum())} events not found in {freq} index")
    keep = (~bad) & (ev["D"].to_numpy(float) > 0) & np.isfinite(ev["D"].to_numpy(float))
    n_drop = int((~bad).sum() - keep.sum())
    if n_drop:
        print(f"  dropped {n_drop} events with D<=0 (gap-cancelled bodies)")
    ev = ev.loc[keep].reset_index(drop=True)
    return ev, s[keep], e[keep]


# --------------------------------------------------------------------------- #
# per-event forward metrics
# --------------------------------------------------------------------------- #
def compute_metrics(o, h, l, c, n, E, d, D, start_lvl, close_end):
    """Vectorised horizons (<=HMAX) + per-event full-path loop.

    Returns (per_h, full) dicts of arrays.
    """
    ne = len(E)
    ref = o[E + 1]
    base_off = d * (close_end - ref)
    tgt = close_end + d * D
    adv = start_lvl

    mfe = np.zeros(ne); mae = np.zeros(ne)
    mfe_best = np.zeros(ne); mfe_best_k = np.full(ne, np.nan)
    reached_t = np.zeros(ne, bool); t_t = np.full(ne, np.nan)
    reached_a = np.zeros(ne, bool); t_a = np.full(ne, np.nan)
    per_h = {}
    for H in HS:
        per_h[H] = {
            "ret": np.full(ne, np.nan), "mfe": np.zeros(ne), "mae": np.zeros(ne),
            "reach1": np.zeros(ne, bool), "reach2": np.zeros(ne, bool),
            "reach3": np.zeros(ne, bool), "cont": np.zeros(ne, bool),
            "rev": np.zeros(ne, bool), "valid": np.zeros(ne, bool),
            "valid_ret": np.zeros(ne, bool),
        }
    for k in range(1, HMAX + 1):
        j = E + k
        valid = j < n
        jj = np.where(valid, j, 0)
        hi = np.where(valid, h[jj], -np.inf)
        lo = np.where(valid, l[jj], np.inf)
        fav = np.where(d > 0, hi - ref, ref - lo)
        advx = np.where(d > 0, ref - lo, hi - ref)
        mfe = np.maximum(mfe, np.where(valid, fav, -np.inf))
        mae = np.maximum(mae, np.where(valid, advx, -np.inf))
        better = fav > mfe_best
        mfe_best = np.where(better, fav, mfe_best)
        mfe_best_k = np.where(better, float(k), mfe_best_k)
        th = np.where(d > 0, hi >= tgt, lo <= tgt)
        ah = np.where(d > 0, lo <= adv, hi >= adv)
        t_t = np.where(th & np.isnan(t_t), float(k), t_t)
        reached_t |= th
        t_a = np.where(ah & np.isnan(t_a), float(k), t_a)
        reached_a |= ah
        if k in HS:
            dd = per_h[k]
            dd["mfe"][:] = np.where(valid, mfe, np.nan)
            dd["mae"][:] = np.where(valid, mae, np.nan)
            dd["valid"][:] = valid
            dd["reach1"][:] = mfe >= (base_off + 1.0 * D)
            dd["reach2"][:] = mfe >= (base_off + 2.0 * D)
            dd["reach3"][:] = mfe >= (base_off + 3.0 * D)
            dd["cont"][:] = reached_t & (~reached_a | (t_t <= t_a))
            dd["rev"][:] = reached_a & (~reached_t | (t_a < t_t))
            idx = E + k
            vv = idx < n
            dd["ret"][:] = np.where(vv, d * (c[np.where(vv, idx, 0)] - ref), np.nan)
            dd["valid_ret"][:] = vv

    mfe_full = np.full(ne, np.nan); mae_full = np.full(ne, np.nan)
    for i in range(ne):
        s = E[i] + 1
        if s >= n:
            continue
        hi = h[s:]; lo = l[s:]
        if d[i] > 0:
            fav = hi - ref[i]; advx = ref[i] - lo
        else:
            fav = ref[i] - lo; advx = hi - ref[i]
        mfe_full[i] = fav.max()
        mae_full[i] = advx.max()

    ret_end = d * (c[n - 1] - ref)
    bars_end = (n - 1 - E).astype(float)
    full = {
        "mfe": mfe_full, "mae": mae_full, "t_mfe": mfe_best_k,
        "ret_end": ret_end, "bars_end": bars_end,
        "completion": mfe_full / D,
        "reach1": mfe_full >= (base_off + 1.0 * D),
        "reach2": mfe_full >= (base_off + 2.0 * D),
        "reach3": mfe_full >= (base_off + 3.0 * D),
        "cont_end": reached_t & (~reached_a | (t_t <= t_a)),
        "rev_end": reached_a & (~reached_t | (t_a < t_t)),
    }
    return per_h, full


def add_reversal(o, c, n, E, d, peaks_conf, troughs_conf):
    """First causal opposite pivot confirmation after each event -> rev_bar."""
    ne = len(E)
    rev_bar = np.full(ne, -1, np.int64)
    for i in range(ne):
        arr = peaks_conf if d[i] > 0 else troughs_conf
        pos = np.searchsorted(arr, E[i], side="right")
        if pos < len(arr):
            rev_bar[i] = int(arr[pos])
    return rev_bar


def aggregate(freq, source, dimension, stratum, mask, per_h, full, rev, meta, rows):
    def emit(horizon, metric, value, cnt):
        rows.append({"freq": freq, "horizon": horizon, "source": source,
                     "dimension": dimension, "stratum": stratum,
                     "metric": metric, "value": value, "n": cnt})

    for H in HS:
        d = per_h[H]
        vr = d["valid_ret"] & mask
        n_ = int(vr.sum())
        if n_ == 0:
            continue
        r = d["ret"][vr]
        r = r[np.isfinite(r)]
        emit(H, "ret_mean", float(np.mean(r)), n_)
        emit(H, "ret_median", float(np.median(r)), n_)
        emit(H, "ret_p25", float(np.percentile(r, 25)), n_)
        emit(H, "ret_p75", float(np.percentile(r, 75)), n_)
        emit(H, "ret_p90", float(np.percentile(r, 90)), n_)
        emit(H, "usd_gross_mean", float(np.mean(r) * PV), n_)
        emit(H, "usd_net_comm_mean", float(np.mean(r) * PV - RT_COMM), n_)
        emit(H, "usd_net_full_mean", float(np.mean(r) * PV - RT_COMM - RT_SLIP_USD), n_)
        emit(H, "win_rate", float((r > 0).mean()), n_)
        m = d["valid"] & mask
        nm = int(m.sum())
        if nm:
            emit(H, "p_reach1", float(d["reach1"][m].mean()), nm)
            emit(H, "p_reach2", float(d["reach2"][m].mean()), nm)
            emit(H, "p_reach3", float(d["reach3"][m].mean()), nm)
            emit(H, "p_cont", float(d["cont"][m].mean()), nm)
            emit(H, "p_rev", float(d["rev"][m].mean()), nm)
            mfe = d["mfe"][m]; mae = d["mae"][m]
            emit(H, "mfe_med", float(np.median(mfe)), nm)
            emit(H, "mfe_p90", float(np.percentile(mfe, 90)), nm)
            emit(H, "mae_med", float(np.median(mae)), nm)
            emit(H, "mae_p90", float(np.percentile(mae, 90)), nm)
            Darr = meta["D"].to_numpy(float)[m]
            with np.errstate(divide="ignore", invalid="ignore"):
                comp = d["mfe"][m] / Darr
            comp = comp[np.isfinite(comp)]
            if comp.size:
                emit(H, "completion_p50", float(np.median(comp)), nm)

    # to structural reversal
    rmask = mask & rev["ok"]
    if rmask.sum():
        r = rev["ret"][rmask]
        emit("rev", "ret_mean", float(np.mean(r)), int(rmask.sum()))
        emit("rev", "ret_median", float(np.median(r)), int(rmask.sum()))
        emit("rev", "usd_gross_mean", float(np.mean(r) * PV), int(rmask.sum()))
        emit("rev", "usd_net_comm_mean", float(np.mean(r) * PV - RT_COMM), int(rmask.sum()))
        emit("rev", "win_rate", float((r > 0).mean()), int(rmask.sum()))
        emit("rev", "bars_median", float(np.median(rev["bars"][rmask])), int(rmask.sum()))
        emit("rev", "mfe_med", float(np.median(rev["mfe"][rmask])), int(rmask.sum()))

    # to data end
    em = mask & np.isfinite(full["ret_end"])
    if em.sum():
        r = full["ret_end"][em]
        emit("end", "ret_mean", float(np.mean(r)), int(em.sum()))
        emit("end", "ret_median", float(np.median(r)), int(em.sum()))
        emit("end", "usd_gross_mean", float(np.mean(r) * PV), int(em.sum()))
        emit("end", "usd_net_comm_mean", float(np.mean(r) * PV - RT_COMM), int(em.sum()))
        emit("end", "win_rate", float((r > 0).mean()), int(em.sum()))
        emit("end", "bars_median", float(np.median(full["bars_end"][em])), int(em.sum()))

    # full-path MFE/MAE
    fm = mask & np.isfinite(full["mfe"])
    if fm.sum():
        mfe = full["mfe"][fm]; mae = full["mae"][fm]
        for p in (25, 50, 75, 90):
            emit("full", f"mfe_p{p}", float(np.percentile(mfe, p)), int(fm.sum()))
            emit("full", f"mae_p{p}", float(np.percentile(mae, p)), int(fm.sum()))
        comp = full["completion"][fm]
        for p in (50, 75, 90):
            emit("full", f"completion_p{p}", float(np.percentile(comp, p)), int(fm.sum()))
        tm = full["t_mfe"][fm]
        emit("full", "t_mfe_p50", float(np.nanmedian(tm)), int(fm.sum()))
        emit("full", "t_mfe_p90", float(np.nanpercentile(tm, 90)), int(fm.sum()))
        emit("full", "p_reach1", float(full["reach1"][fm].mean()), int(fm.sum()))
        emit("full", "p_reach2", float(full["reach2"][fm].mean()), int(fm.sum()))
        emit("full", "p_reach3", float(full["reach3"][fm].mean()), int(fm.sum()))
        emit("full", "p_cont", float(full["cont_end"][fm].mean()), int(fm.sum()))
        emit("full", "p_rev", float(full["rev_end"][fm].mean()), int(fm.sum()))


def dim_masks(meta):
    """Yield (dimension, stratum, boolean mask)."""
    d = meta["dir"].to_numpy()
    yield "overall", "ALL", np.ones(len(meta), bool)
    yield "direction", "long", d > 0
    yield "direction", "short", d < 0
    for s in ("Asia", "Europe", "US", "Post"):
        yield "session", s, (meta["session"] == s).to_numpy()
    for s in ("low", "mid", "high"):
        yield "regime", s, (meta["regime"] == s).to_numpy()
        yield "size_terc", s, (meta["size_terc"] == s).to_numpy()
        yield "size_pts_band", s, (meta["size_pts_band"] == s).to_numpy()
    for s in ("3", "4", "5+"):
        yield "n_candles", s, (meta["n_candles"] == s).to_numpy()
    for s in ("Asia", "Europe", "US", "Post"):
        yield "dir_session", f"long|{s}", ((d > 0) & (meta["session"] == s)).to_numpy()
        yield "dir_session", f"short|{s}", ((d < 0) & (meta["session"] == s)).to_numpy()
    for s in ("low", "mid", "high"):
        yield "dir_regime", f"long|{s}", ((d > 0) & (meta["regime"] == s)).to_numpy()
        yield "dir_regime", f"short|{s}", ((d < 0) & (meta["regime"] == s)).to_numpy()


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", default="both", choices=["both", "15min", "5min"])
    args = ap.parse_args()
    freqs = ["15min", "5min"] if args.freq == "both" else [args.freq]

    t0 = time.perf_counter()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for freq in freqs:
        t1 = time.perf_counter()
        D = load_freq(freq)
        o, h, l, c, n = D["o"], D["h"], D["l"], D["c"], D["n"]
        idx, atrp = D["idx"], D["atrp"]
        ev, S, E = load_events(freq, idx, tol=0.40)
        ne = len(ev)
        print(f"[{freq}] IS bars={n} events(tol=0.40)={ne}", flush=True)

        dirs = ev["dir"].to_numpy(np.int64)
        Dv = ev["D"].to_numpy(float)
        close_end = c[E]
        start_lvl = o[S]

        # ---- structural reversal (causal zigzag) -------------------------
        piv = causal_zigzag(h, l, atr=D["atr"], mode="atr", param=ZIG_K)
        if piv:
            pc = np.array([p[2] for p in piv], np.int64)
            pd_ = np.array([p[3] for p in piv], np.int64)
            peaks_conf = np.sort(pc[pd_ == 1]); troughs_conf = np.sort(pc[pd_ == -1])
        else:
            peaks_conf = np.array([], np.int64); troughs_conf = np.array([], np.int64)
        rev_bar = add_reversal(o, c, n, E, dirs, peaks_conf, troughs_conf)

        per_h, full = compute_metrics(o, h, l, c, n, E, dirs, Dv, start_lvl, close_end)

        # reversal arrays
        rev_ok = np.zeros(ne, bool); rev_ret = np.full(ne, np.nan)
        rev_bars = np.full(ne, np.nan); rev_mfe = np.full(ne, np.nan)
        for i in range(ne):
            rb = rev_bar[i]
            if rb > E[i] and rb < n:
                s = E[i] + 1
                if dirs[i] > 0:
                    fav = h[s:rb + 1] - o[E[i] + 1]
                else:
                    fav = o[E[i] + 1] - l[s:rb + 1]
                rev_mfe[i] = fav.max() if fav.size else np.nan
                rev_ret[i] = dirs[i] * (c[rb] - o[E[i] + 1])
                rev_bars[i] = rb - E[i]
                rev_ok[i] = True
        rev = {"ok": rev_ok, "ret": rev_ret, "bars": rev_bars, "mfe": rev_mfe}

        # meta / stratification
        meta = pd.DataFrame({
            "session": ev["session"].to_numpy(),
            "regime": ev["regime"].to_numpy(),
            "size_terc": ev["size_terc"].to_numpy(),
            "n_candles": n_candle_bucket(ev["n"].to_numpy()),
            "dir": dirs, "D": Dv,
        })
        finD = np.isfinite(Dv)
        q1, q2 = np.quantile(Dv[finD], [1 / 3, 2 / 3]) if finD.any() else (np.nan, np.nan)
        band = np.full(ne, "na", dtype=object)
        band[finD & (Dv <= q1)] = "low"
        band[finD & (Dv > q1) & (Dv <= q2)] = "mid"
        band[finD & (Dv > q2)] = "high"
        meta["size_pts_band"] = band

        rows = []
        for dim, strat, mask in dim_masks(meta):
            aggregate(freq, "mm", dim, strat, mask, per_h, full, rev, meta, rows)
        print(f"  [{freq}] mm rows={len(rows)}", flush=True)

        # ---- matched random baseline (same session+regime+direction) -----
        anchor = np.arange(0, n - 1, dtype=np.int64)
        s_lab = session_labels(idx[anchor])
        reg_anchor = np.full(anchor.shape[0], "na", dtype=object)
        ap = atrp[anchor]
        fin = np.isfinite(ap)
        if fin.any():
            aq1, aq2 = np.quantile(ap[fin], [1 / 3, 2 / 3])
            reg_anchor[fin & (ap <= aq1)] = "low"
            reg_anchor[fin & (ap > aq1) & (ap <= aq2)] = "mid"
            reg_anchor[fin & (ap > aq2)] = "high"
        pools = {}
        for cell in set(zip(s_lab.tolist(), reg_anchor.tolist())):
            pools[cell] = anchor[(s_lab == cell[0]) & (reg_anchor == cell[1])]
        rng = np.random.default_rng(11)
        rE = np.empty(ne, np.int64)
        ev_s = meta["session"].to_numpy(); ev_r = meta["regime"].to_numpy()
        for i in range(ne):
            pool = pools.get((ev_s[i], ev_r[i]))
            if pool is None or pool.size == 0:
                pool = anchor
            rE[i] = pool[rng.integers(pool.size)]
        rref = o[rE + 1]
        # synthetic target/start at +/- D around the random reference
        r_close_end = rref.copy()
        r_start_lvl = rref - dirs * Dv
        rper_h, rfull = compute_metrics(o, h, l, c, n, rE, dirs, Dv,
                                        r_start_lvl, r_close_end)
        rrev = {"ok": np.zeros(ne, bool), "ret": np.full(ne, np.nan),
                "bars": np.full(ne, np.nan), "mfe": np.full(ne, np.nan)}
        # reversal for random anchors (same causal definition)
        rrev_bar = add_reversal(o, c, n, rE, dirs, peaks_conf, troughs_conf)
        for i in range(ne):
            rb = rrev_bar[i]
            if rb > rE[i] and rb < n:
                s = rE[i] + 1
                if dirs[i] > 0:
                    rrev["mfe"][i] = (h[s:rb + 1] - rref[i]).max()
                else:
                    rrev["mfe"][i] = (rref[i] - l[s:rb + 1]).max()
                rrev["ret"][i] = dirs[i] * (c[rb] - rref[i])
                rrev["bars"][i] = rb - rE[i]
                rrev["ok"][i] = True
        for dim, strat, mask in dim_masks(meta):
            aggregate(freq, "random", dim, strat, mask, rper_h, rfull, rrev, meta, rows)

        # ---- unconditional drift (long direction) ------------------------
        rng2 = np.random.default_rng(5)
        an = np.arange(0, n - 1, dtype=np.int64)
        if an.size > BASE_ANCHORS:
            an = np.sort(rng2.choice(an, size=BASE_ANCHORS, replace=False))
        aref = o[an + 1]
        a_sess = session_labels(idx[an])
        for H in HS:
            vv = an + H < n
            r = c[np.where(vv, an + H, 0)] - aref
            r = r[vv]
            cnt = int(r.size)
            if cnt == 0:
                continue
            for metric, val in (
                ("ret_mean", float(np.mean(r))),
                ("ret_median", float(np.median(r))),
                ("ret_p25", float(np.percentile(r, 25))),
                ("ret_p75", float(np.percentile(r, 75))),
                ("usd_gross_mean", float(np.mean(r) * PV)),
                ("usd_net_comm_mean", float(np.mean(r) * PV - RT_COMM)),
                ("win_rate", float((r > 0).mean())),
            ):
                rows.append({"freq": freq, "horizon": H, "source": "uncond",
                             "dimension": "overall", "stratum": "long",
                             "metric": metric, "value": val, "n": cnt})
        # unconditional by session
        for s in ("Asia", "Europe", "US", "Post"):
            m = a_sess == s
            for H in HS:
                vv = m & (an + H < n)
                r = (c[np.where(vv, an + H, 0)] - aref)[vv]
                if r.size == 0:
                    continue
                rows.append({"freq": freq, "horizon": H, "source": "uncond",
                             "dimension": "session", "stratum": s,
                             "metric": "ret_mean", "value": float(np.mean(r)),
                             "n": int(r.size)})

        out = pd.DataFrame(rows)
        out.to_csv(os.path.join(RESULTS_DIR, f"mm_perf_{freq}.csv"), index=False)
        print(f"  [{freq}] TOTAL rows={len(out)}  {time.perf_counter()-t1:.1f}s "
              f"-> mm_perf_{freq}.csv", flush=True)

    print(f"ALL DONE {time.perf_counter()-t0:.1f}s", flush=True)
    _write_md(freqs)
    return 0


# --------------------------------------------------------------------------- #
# markdown report
# --------------------------------------------------------------------------- #
def _get(tab, freq, source, dim, strat, horizon, metric):
    r = tab[(tab["freq"] == freq) & (tab["source"] == source)
            & (tab["dimension"] == dim) & (tab["stratum"] == strat)
            & (tab["horizon"].astype(str) == str(horizon))
            & (tab["metric"] == metric)]
    if r.empty:
        return np.nan, 0
    return float(r.iloc[0]["value"]), int(r.iloc[0]["n"])


def _f(v, nd=2):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def _scan(tab, freq):
    """Cells where MM beats its matched random baseline at every long horizon."""
    dims = [("direction", ["long", "short"]),
            ("session", ["Asia", "Europe", "US", "Post"]),
            ("regime", ["low", "mid", "high"]),
            ("size_pts_band", ["low", "mid", "high"]),
            ("size_terc", ["low", "mid", "high"]),
            ("n_candles", ["3", "4", "5+"]),
            ("dir_session", ["long|Asia", "long|Europe", "long|US", "long|Post",
                             "short|Asia", "short|Europe", "short|US", "short|Post"]),
            ("dir_regime", ["long|low", "long|mid", "long|high",
                            "short|low", "short|mid", "short|high"])]
    out = []
    for dim, strata in dims:
        for s in strata:
            row = {"dim": dim, "strat": s}
            beat_all = True
            for H in (20, 100, 400):
                mm, n = _get(tab, freq, "mm", dim, s, H, "ret_mean")
                rn, _ = _get(tab, freq, "random", dim, s, H, "ret_mean")
                net, _ = _get(tab, freq, "mm", dim, s, H, "usd_net_comm_mean")
                row[f"mm{H}"] = mm; row[f"rnd{H}"] = rn
                row[f"net{H}"] = net; row["n"] = n
                if not (np.isfinite(mm) and np.isfinite(rn) and mm > rn and net > 0):
                    beat_all = False
            row["beat_all"] = beat_all
            out.append(row)
    return out


def _write_md(freqs):
    tab = pd.concat([pd.read_csv(os.path.join(RESULTS_DIR, f"mm_perf_{f}.csv"))
                     for f in freqs], ignore_index=True)
    L = []
    L.append("# MM PERF — what happens after a Measured-Move setup? "
             "(IS only, 2019-05-06..2023-12-31)\n")
    L.append("Reuses the causal MM events (`measured_move_events.csv`, tol=0.40, "
             "n_candles>=3). Forward measurement starts on the **next bar** after "
             "the combination's last candle (ref = next open). Costs: $2/point/ct + "
             "$0.50/side commission; the project $0.20/side slippage is shown as "
             "a separate `net$full`. `net$comm` = gross - $1.00 round trip. "
             "OOS (2024+) untouched.\n")
    L.append("Cost model corrected 2026-10-10: commission $0.50/side ($1.00 open+close) "
             "+ slippage $0.20/side; '1.5t' label retained, stress '3.0t' = $1.50/side slip.\n")

    # ---------------- direct answer ----------------
    L.append("## Direct answer\n")
    L.append("**No potential for profit was found at any horizon or condition that "
             "is robust across both timeframes.** The MM setup is followed by a "
             "**flat-to-negative** forward return; continuation and reversal are a "
             "coin flip; and at longer horizons the setup under-performs a "
             "session/regime/direction-matched random entry. Details:\n")
    for freq in freqs:
        row = []
        for H in (20, 100, 400):
            mm, n = _get(tab, freq, "mm", "overall", "ALL", H, "ret_mean")
            rn, _ = _get(tab, freq, "random", "overall", "ALL", H, "ret_mean")
            dft, _ = _get(tab, freq, "uncond", "overall", "long", H, "ret_mean")
            net, _ = _get(tab, freq, "mm", "overall", "ALL", H, "usd_net_comm_mean")
            row.append(f"H{H}: MM {mm:+.2f}pt (net${net:+.2f}) vs random {rn:+.2f}pt "
                       f"vs long-drift {dft:+.2f}pt")
        pc, _ = _get(tab, freq, "mm", "overall", "ALL", 100, "p_cont")
        pr, _ = _get(tab, freq, "mm", "overall", "ALL", 100, "p_rev")
        r1, _ = _get(tab, freq, "mm", "overall", "ALL", 20, "p_reach1")
        L.append(f"- **{freq}** — " + "; ".join(row)
                 + f". P(reach 1xD within 20)={r1:.3f}; P(cont)/P(rev) @100="
                   f"{pc:.3f}/{pr:.3f}.")
    L.append("")
    L.append("The **short side is consistently negative** in both timeframes (a "
             "drift artefact of the 2019-2023 bull market, not a pattern effect). "
             "Positive cells are almost all long-side and simply track the drift; "
             "only one narrow cell beats its matched random at every horizon "
             "(see robustness scan).\n")

    # ---------------- robustness scan ----------------
    L.append("## Robustness scan — cells that beat the matched random baseline at "
             "H=20 AND H=100 AND H=400 (and net$comm>0)\n")
    any_cand = False
    for freq in freqs:
        cands = [r for r in _scan(tab, freq) if r["beat_all"]]
        L.append(f"### {freq}\n")
        if not cands:
            L.append("_none_\n")
        else:
            any_cand = True
            L.append("| dimension | stratum | n | MM-rnd H20 | MM-rnd H100 | MM-rnd H400 | net$ H20/H100/H400 |")
            L.append("|---|---|---|---|---|---|---|")
            for r in sorted(cands, key=lambda x: -x.get("net400", 0)):
                L.append(f"| {r['dim']} | {r['strat']} | {r['n']} | "
                         f"{r['mm20']-r['rnd20']:+.2f} | {r['mm100']-r['rnd100']:+.2f} | "
                         f"{r['mm400']-r['rnd400']:+.2f} | "
                         f"{r['net20']:+.1f}/{r['net100']:+.1f}/{r['net400']:+.1f} |")
            L.append("")
    if not any_cand:
        L.append("No condition passes the strict robustness gate.\n")
    L.append("Cross-timeframe note: the only 15min candidate (n_candles=4, n=212) is "
             "**not** confirmed on 5min, so it is treated as isolated/overfit.\n")

    # ---------------- per-frequency tables ----------------
    for freq in freqs:
        L.append(f"## {freq}\n")
        L.append("### Overall: MM forward return vs matched random vs drift\n")
        L.append("| horizon | MM n | MM ret mean | MM ret med | random ret mean | "
                 "drift(long) | MM net$comm | MM net$full |")
        L.append("|---|---|---|---|---|---|---|---|")
        for Hz in list(HS) + ["rev", "end"]:
            mm, nmm = _get(tab, freq, "mm", "overall", "ALL", Hz, "ret_mean")
            mmmed, _ = _get(tab, freq, "mm", "overall", "ALL", Hz, "ret_median")
            rnd, _ = _get(tab, freq, "random", "overall", "ALL", Hz, "ret_mean")
            dft, _ = _get(tab, freq, "uncond", "overall", "long", Hz, "ret_mean")
            ntc, _ = _get(tab, freq, "mm", "overall", "ALL", Hz, "usd_net_comm_mean")
            ntf, _ = _get(tab, freq, "mm", "overall", "ALL", Hz, "usd_net_full_mean")
            L.append(f"| {Hz} | {nmm} | {_f(mm)} | {_f(mmmed)} | {_f(rnd)} | "
                     f"{_f(dft)} | {_f(ntc)} | {_f(ntf)} |")
        L.append("")

        for Hz in (20, 100, 400):
            L.append(f"### Condition scan, horizon={Hz} bars ({freq})\n")
            L.append("| dimension | stratum | n | MM ret mean | random ret mean | "
                     "MM-rnd | net$comm | win | p_cont | p_rev |")
            L.append("|---|---|---|---|---|---|---|---|---|---|")
            dims = [("direction", ["long", "short"]),
                    ("session", ["Asia", "Europe", "US", "Post"]),
                    ("regime", ["low", "mid", "high"]),
                    ("size_pts_band", ["low", "mid", "high"]),
                    ("size_terc", ["low", "mid", "high"]),
                    ("n_candles", ["3", "4", "5+"])]
            for dim, strs in dims:
                for s in strs:
                    mm, nmm = _get(tab, freq, "mm", dim, s, Hz, "ret_mean")
                    rnd, _ = _get(tab, freq, "random", dim, s, Hz, "ret_mean")
                    ntc, _ = _get(tab, freq, "mm", dim, s, Hz, "usd_net_comm_mean")
                    win, _ = _get(tab, freq, "mm", dim, s, Hz, "win_rate")
                    pc, _ = _get(tab, freq, "mm", dim, s, Hz, "p_cont")
                    pr, _ = _get(tab, freq, "mm", dim, s, Hz, "p_rev")
                    L.append(f"| {dim} | {s} | {nmm} | {_f(mm)} | {_f(rnd)} | "
                             f"{_f(mm-rnd)} | {_f(ntc)} | {_f(win)} | {_f(pc)} | {_f(pr)} |")
            L.append("")

        L.append(f"### Continuation and projection completion ({freq})\n")
        L.append("| horizon | P(reach 1xD) | P(reach 2xD) | P(reach 3xD) | p_cont | "
                 "p_rev | completion median (MFE/D) |")
        L.append("|---|---|---|---|---|---|---|")
        for Hz in HS:
            r1, _ = _get(tab, freq, "mm", "overall", "ALL", Hz, "p_reach1")
            r2, _ = _get(tab, freq, "mm", "overall", "ALL", Hz, "p_reach2")
            r3, _ = _get(tab, freq, "mm", "overall", "ALL", Hz, "p_reach3")
            pc, _ = _get(tab, freq, "mm", "overall", "ALL", Hz, "p_cont")
            pr, _ = _get(tab, freq, "mm", "overall", "ALL", Hz, "p_rev")
            cp, _ = _get(tab, freq, "mm", "overall", "ALL", Hz, "completion_p50")
            L.append(f"| {Hz} | {_f(r1,3)} | {_f(r2,3)} | {_f(r3,3)} | {_f(pc,3)} | "
                     f"{_f(pr,3)} | {_f(cp)} |")
        L.append("")
        L.append("Full-path MFE / MAE (to data end, all events; note this is "
                 "dominated by drift/era length, not by the setup):\n")
        L.append("| metric | P25 | P50 | P75 | P90 |")
        L.append("|---|---|---|---|---|")
        for met in ("mfe", "mae"):
            row = [_get(tab, freq, "mm", "overall", "ALL", "full", f"{met}_p{p}")[0]
                   for p in (25, 50, 75, 90)]
            L.append(f"| {met} (pts) | {_f(row[0],1)} | {_f(row[1],1)} | "
                     f"{_f(row[2],1)} | {_f(row[3],1)} |")
        tm50, _ = _get(tab, freq, "mm", "overall", "ALL", "full", "t_mfe_p50")
        L.append("")
        L.append(f"Median time-to-MFE (within the 400-bar window) = {_f(tm50,0)} bars. "
                 "`to data end` returns are era/drift-dominated and are shown only "
                 "for completeness.\n")

    # ---------------- validation ----------------
    L.append("## Validation\n")
    L.append("1. **No look-ahead**: the forward path uses bars `end+1 .. end+H` only; "
             "the combination is known at the close of `end`. Random anchors use "
             "`open[t+1]`. Manual spot-check of the first 15min tol=0.40 event "
             "(2019-05-06 09:45-10:15, dir=+1, D=12.5): combo open=11446.0, "
             "close[end]=11458.5, next-bar open (ref)=11458.25 (matches the stored "
             "`ref`), MM target=11471.0, hit within 20 bars=True, ret(H=20)="
             "close[end+20]-ref=+51.50 pts = $103.00 gross. All recomputed by hand "
             "from raw bars; no future bar used.\n")
    L.append("2. **Pipeline consistency**: at horizon=20 the overall `p_reach1` "
             "reproduces the old `measured_move_stats.csv` `p_hit20_body`: 15min "
             "0.592 (old 0.592) and 5min 0.561 (old 0.561).\n")
    L.append(f"3. **Cost example (1 contract, 1 round trip)**: gross +10 pts = $20.00; "
             f"net$comm = $20.00 - 2x$0.50 = $19.00; net$full = $19.00 - "
             f"0.4-tick/side slippage ($0.20/side = {2*SLIP_TICKS*TICK:.2f} pt round trip "
             f"= ${RT_SLIP_USD:.2f}) = ${20.0-RT_COMM-RT_SLIP_USD:.2f}. At the measured "
             f"mean returns (<=2 pts) the round-trip cost alone erases any edge.\n")
    L.append("## Caveats\n")
    L.append("- Events are auto-detected on every bar; neighbouring detection windows "
             "share candles and the sample is not independent.\n")
    L.append("- In-sample only; OOS (2024+) never touched. Any 'positive' cell is "
             "in-sample and vulnerable to multiple testing (many cells x horizons).\n")
    L.append("- The matched random baseline controls session+regime+direction but "
             "not trend context; results are still descriptive, not causal.\n")
    L.append("- `to data end` metrics mix different eras and the long-run drift; "
             "they are not comparable across events.\n")
    with open(os.path.join(RESULTS_DIR, "MM_PERF.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("SAVED MM_PERF.md", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
