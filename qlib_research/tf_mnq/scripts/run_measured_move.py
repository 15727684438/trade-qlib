"""Measured-Move (MM) study — do >=3 approx-equal-body TREND candles reach a
target of double their body distance? (IN-SAMPLE only, 2019-05-06..2023-12-31)

Run:  python scripts/run_measured_move.py
Out:  results/measured_move_stats.csv      (per-condition stats)
      results/measured_move_events.csv     (matched combinations)
      results/MEASURED_MOVE.md

Causality: a combination is known only at the close of its last candle; all
forward measurement starts on the NEXT bar. OOS (2024+) is never touched.
"""

from __future__ import annotations

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
from tf_mnq.measured_move import (  # noqa: E402
    base_prob, extract_combos, forward_move_samples, measure_forward,
    n_candle_bucket, session_labels, tercile_labels,
)

RESULTS_DIR = os.path.join(_PKG_ROOT, "results")
IS_START, IS_END = "2019-05-06", "2023-12-31"
FREQS = ("15min", "5min", "1min")
TOLS = (0.25, 0.40, 0.60)
ATR_MULTS = (0.0, 0.3, 0.5)
NS = (5, 10, 20, 50)
N_MAX = max(NS)
BASE_SAMPLE = 200_000

EVENT_COLS = [
    "freq", "tol", "start_time", "end_time", "dir", "n", "n_bucket",
    "D", "D_sum", "mean_body", "atr_start", "D_atr", "atr_pct", "regime",
    "size_terc", "session", "target_body", "target_sum", "ref",
    "req_body", "req_sum",
] + [f"hit{N}_body" for N in NS] + [f"hit{N}_sum" for N in NS] + [
    "base5_body", "base10_body", "base20_body", "base50_body",
    "base5_sum", "base10_sum", "base20_sum", "base50_sum",
    "t_body", "t_sum", "t_adv", "adverse", "adverse_before", "mfe", "mae",
    "mfe_d", "mae_d", "base20_body_2D", "base20_sum_2D",
]

STAT_COLS = ["freq", "tol", "atr_mult", "strat", "group", "n"] + [
    f"p_hit{N}_body" for N in NS] + [f"p_hit{N}_sum" for N in NS] + [
    "base20_body", "base20_sum", "base20_body_2D", "base20_sum_2D",
    "edge20_body", "edge20_sum",
    "t_med_body", "t_q1_body", "t_q3_body",
    "adverse", "adverse_before", "mfe_med", "mae_med",
    "mfe_d_med", "mae_d_med", "D_atr_med", "atrpc_med",
]


def _empty_stats():
    return []


def agg_stats(ev: pd.DataFrame) -> dict:
    n = len(ev)
    out = {"n": n}
    for N in NS:
        out[f"p_hit{N}_body"] = float(ev[f"hit{N}_body"].mean()) if n else np.nan
        out[f"p_hit{N}_sum"] = float(ev[f"hit{N}_sum"].mean()) if n else np.nan
    out["base20_body"] = float(ev["base20_body"].mean()) if n else np.nan
    out["base20_sum"] = float(ev["base20_sum"].mean()) if n else np.nan
    out["base20_body_2D"] = float(ev["base20_body_2D"].mean()) if n else np.nan
    out["base20_sum_2D"] = float(ev["base20_sum_2D"].mean()) if n else np.nan
    out["edge20_body"] = out["p_hit20_body"] - out["base20_body"] if n else np.nan
    out["edge20_sum"] = out["p_hit20_sum"] - out["base20_sum"] if n else np.nan
    if n:
        t = ev.loc[ev["t_body"] <= 20, "t_body"].to_numpy(float)
        if t.size:
            out["t_med_body"] = float(np.median(t))
            out["t_q1_body"] = float(np.percentile(t, 25))
            out["t_q3_body"] = float(np.percentile(t, 75))
        else:
            out["t_med_body"] = out["t_q1_body"] = out["t_q3_body"] = np.nan
        out["adverse"] = float(ev["adverse"].mean())
        out["adverse_before"] = float(ev["adverse_before"].mean())
        out["mfe_med"] = float(np.nanmedian(ev["mfe"]))
        out["mae_med"] = float(np.nanmedian(ev["mae"]))
        out["mfe_d_med"] = float(np.nanmedian(ev["mfe_d"]))
        out["mae_d_med"] = float(np.nanmedian(ev["mae_d"]))
        out["D_atr_med"] = float(np.nanmedian(ev["D_atr"]))
        out["atrpc_med"] = float(np.nanmedian(ev["atr_pct"]))
    else:
        for k in ("t_med_body", "t_q1_body", "t_q3_body", "adverse",
                  "adverse_before", "mfe_med", "mae_med", "mfe_d_med",
                  "mae_d_med", "D_atr_med", "atrpc_med"):
            out[k] = np.nan
    return out


def strat_rows(freq, tol, atr_mult, ev, exp):
    """Build all statistics rows for one (freq, tol, atr_mult) cell."""
    rows = []

    def add(strat, group, sub):
        base = agg_stats(sub)
        base.update({"freq": freq, "tol": tol, "atr_mult": atr_mult,
                     "strat": strat, "group": group})
        rows.append(base)

    add("overall", "ALL", ev)
    if len(ev):
        add("direction", "long", ev[ev["dir"] > 0])
        add("direction", "short", ev[ev["dir"] < 0])
        for g in ("3", "4", "5+"):
            add("n_candles", g, ev[ev["n_bucket"] == g])
        for g in ("low", "mid", "high"):
            add("size_tercile", g, ev[ev["size_terc"] == g])
            add("regime", g, ev[ev["regime"] == g])
        for g in ("Asia", "Europe", "US", "Post"):
            add("session", g, ev[ev["session"] == g])
    exp.extend(rows)
    return rows


def main() -> int:
    t0 = time.perf_counter()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stats_rows_all = []
    events_all = []

    for freq in FREQS:
        tf0 = time.perf_counter()
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
        n = len(sl)
        print(f"[{freq}] IS bars={n}", flush=True)

        # ---- random-bar base distributions (same next-bar convention) --------
        max_start = n - 1 - N_MAX
        if max_start > 0:
            pool = np.arange(0, max_start + 1)
            rng = np.random.default_rng(7)
            if pool.size > BASE_SAMPLE:
                anchors = np.sort(rng.choice(pool, size=BASE_SAMPLE, replace=False))
            else:
                anchors = pool
        else:
            anchors = np.array([], dtype=np.int64)
        base_sorted = {}
        for N in NS:
            an = anchors[anchors <= n - 1 - N]
            base_sorted[N] = forward_move_samples(h, l, o, an, N)

        for tol in TOLS:
            cm = extract_combos(o, c, tol)
            ne = len(cm)
            if ne == 0:
                print(f"  [{freq}] tol={tol:g} no combos", flush=True)
                continue
            st = cm["start"].to_numpy(np.int64)
            en = cm["end"].to_numpy(np.int64)
            dr = cm["dir"].to_numpy(np.int64)
            D = cm["D"].to_numpy(float)
            Dsum = cm["D_sum"].to_numpy(float)
            mb = cm["mean_body"].to_numpy(float)
            fw = measure_forward(o, h, l, c, st, en, dr, D, Dsum, Ns=NS)

            ev = pd.DataFrame({
                "freq": freq, "tol": tol,
                "start_time": idx[st], "end_time": idx[en],
                "dir": dr, "n": cm["n"].to_numpy(np.int64),
                "n_bucket": n_candle_bucket(cm["n"].to_numpy()),
                "D": D, "D_sum": Dsum, "mean_body": mb,
                "atr_start": atr[st], "D_atr": D / np.where(atr[st] > 0, atr[st], np.nan),
                "atr_pct": atrp[st],
                "target_body": fw["target_body"], "target_sum": fw["target_sum"],
                "ref": fw["ref"], "t_body": fw["t_body"], "t_sum": fw["t_sum"],
                "t_adv": fw["t_adv"], "adverse": fw["adverse"],
                "adverse_before": fw["adverse_before"],
                "mfe": fw["mfe"], "mae": fw["mae"],
            })
            for N in NS:
                ev[f"hit{N}_body"] = fw[f"hit{N}_body"]
                ev[f"hit{N}_sum"] = fw[f"hit{N}_sum"]
            # required forward move from the next-bar reference + base rate
            req_body = dr * (fw["target_body"] - fw["ref"])
            req_sum = dr * (fw["target_sum"] - fw["ref"])
            ev["req_body"] = req_body
            ev["req_sum"] = req_sum
            long_m = (dr > 0)
            for N in NS:
                up_s, dn_s = base_sorted[N]
                bb = np.where(long_m, base_prob(up_s, req_body), base_prob(dn_s, req_body))
                bs = np.where(long_m, base_prob(up_s, req_sum), base_prob(dn_s, req_sum))
                ev[f"base{N}_body"] = bb
                ev[f"base{N}_sum"] = bs
            safe_D = np.where(D > 0, D, np.nan)
            ev["mfe_d"] = fw["mfe"] / safe_D
            ev["mae_d"] = fw["mae"] / safe_D
            up20, dn20 = base_sorted[20]
            b2d = np.where(long_m, base_prob(up20, 2.0 * D), base_prob(dn20, 2.0 * D))
            b2s = np.where(long_m, base_prob(up20, 2.0 * Dsum),
                           base_prob(dn20, 2.0 * Dsum))
            ev["base20_body_2D"] = b2d
            ev["base20_sum_2D"] = b2s
            ev["session"] = session_labels(idx[en])
            lab, _ = tercile_labels(ev["D_atr"].to_numpy(float))
            ev["size_terc"] = lab
            lab, _ = tercile_labels(ev["atr_pct"].to_numpy(float))
            ev["regime"] = lab

            events_all.append(ev[EVENT_COLS].copy())
            # incremental save of events
            pd.concat(events_all, ignore_index=True)[EVENT_COLS].to_csv(
                os.path.join(RESULTS_DIR, "measured_move_events.csv"), index=False)

            for mult in ATR_MULTS:
                if mult > 0:
                    keep = np.isfinite(ev["atr_start"].to_numpy(float)) & (
                        ev["mean_body"].to_numpy(float) >= mult * ev["atr_start"].to_numpy(float))
                    sub = ev.loc[keep]
                else:
                    sub = ev
                strat_rows(freq, tol, mult, sub, stats_rows_all)

            # incremental save of stats
            pd.DataFrame(stats_rows_all, columns=STAT_COLS).to_csv(
                os.path.join(RESULTS_DIR, "measured_move_stats.csv"), index=False)
            print(f"  [{freq}] tol={tol:g} combos={ne} "
                  f"hit20={np.nanmean(ev['hit20_body']):.3f} "
                  f"base20={np.nanmean(ev['base20_body']):.3f}", flush=True)

        print(f"  [{freq}] done {time.perf_counter()-tf0:.1f}s", flush=True)

    stats = pd.DataFrame(stats_rows_all, columns=STAT_COLS)
    stats.to_csv(os.path.join(RESULTS_DIR, "measured_move_stats.csv"), index=False)
    events = pd.concat(events_all, ignore_index=True) if events_all else pd.DataFrame()
    events.to_csv(os.path.join(RESULTS_DIR, "measured_move_events.csv"), index=False)
    print(f"TOTAL {time.perf_counter()-t0:.1f}s  stats_rows={len(stats)} "
          f"events={len(events)}", flush=True)

    _write_markdown(stats, events)
    return 0


# --------------------------------------------------------------------------- #
# markdown
# --------------------------------------------------------------------------- #
def _fmt(v, nd=3):
    try:
        if v is None or not np.isfinite(v):
            return "—"
        return f"{v:.{nd}f}"
    except Exception:
        return "—"


def _cell(stats, freq, tol, mult, strat, group):
    r = stats[(stats["freq"] == freq) & (stats["tol"] == tol)
              & (stats["atr_mult"] == mult) & (stats["strat"] == strat)
              & (stats["group"] == group)]
    if r.empty:
        return None
    return r.iloc[0]


def _write_markdown(stats, events):
    L = []
    L.append("# MEASURED MOVE — equal-length TREND candle combinations (IS only, "
             "2019-05-06..2023-12-31)\n")
    L.append("**Pattern**: >=3 consecutive same-direction candles with approximately "
             "equal bodies (each body within ``tol`` of the running segment mean). "
             "**D** = dir*(close[last]-open[first]); **MM target** = close[last] + "
             "dir*D (D_sum variant: close[last] + dir*D_sum). A combination is known "
             "only at the close of its last candle; the forward path is measured from "
             "the **next bar** (reference = next open). OOS (2024+) never touched.\n")
    L.append("Cost model corrected 2026-10-10: commission $0.50/side ($1.00 open+close) "
             "+ slippage $0.20/side; '1.5t' label retained, stress '3.0t' = $1.50/side slip. "
             "(This report is cost-free probabilities; the cost note applies to the "
             "downstream momentum/MM-performance reports.)\n")

    # ---------- base-rate comparison (overall, by freq/tol) ----------
    L.append("## Base-rate comparison (body target)\n")
    L.append("Random-bar base rate = P(a random bar's forward move within N bars "
             "reaches the same required distance from the next-bar reference), "
             "estimated on a 200k-bar random sample.\n")
    L.append("| freq | tol | N combos | P(hit≤5) | P(hit≤10) | P(hit≤20) | P(hit≤50) "
             "| base20 (req) | base20 (2*D) | edge20 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for freq in FREQS:
        for tol in TOLS:
            r = _cell(stats, freq, tol, 0.0, "overall", "ALL")
            if r is None:
                continue
            L.append(f"| {freq} | {tol:g} | {int(r['n'])} | "
                     f"{_fmt(r['p_hit5_body'])} | {_fmt(r['p_hit10_body'])} | "
                     f"{_fmt(r['p_hit20_body'])} | {_fmt(r['p_hit50_body'])} | "
                     f"{_fmt(r['base20_body'])} | {_fmt(r['base20_body_2D'])} | "
                     f"{_fmt(r['edge20_body'])} |")
    L.append("")

    # ---------- characteristics (15min primary) ----------
    L.append("## Characteristics table — 15min (which conditions RAISE MM hit "
             "probability)\n")
    for tol in TOLS:
        r = _cell(stats, "15min", tol, 0.0, "overall", "ALL")
        if r is None:
            continue
        L.append(f"### tol = {tol:g}  (N={int(r['n'])}, P(hit≤20)={_fmt(r['p_hit20_body'])}, "
                 f"base20={_fmt(r['base20_body'])}, P(touch start first)="
                 f"{_fmt(r['adverse_before'])})\n")
        L.append("| strat | group | N | P(hit≤5) | P(hit≤10) | P(hit≤20) | P(hit≤50) "
                 "| base20 | edge20 | touch≤20 | touch-first | MFE/D | MAE/D |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        order = [("direction", g) for g in ("long", "short")]
        order += [("n_candles", g) for g in ("3", "4", "5+")]
        order += [("size_tercile", g) for g in ("low", "mid", "high")]
        order += [("regime", g) for g in ("low", "mid", "high")]
        order += [("session", g) for g in ("Asia", "Europe", "US", "Post")]
        for strat, group in order:
            rr = _cell(stats, "15min", tol, 0.0, strat, group)
            if rr is None:
                continue
            L.append(f"| {strat} | {group} | {int(rr['n'])} | {_fmt(rr['p_hit5_body'])} | "
                     f"{_fmt(rr['p_hit10_body'])} | {_fmt(rr['p_hit20_body'])} | "
                     f"{_fmt(rr['p_hit50_body'])} | {_fmt(rr['base20_body'])} | "
                     f"{_fmt(rr['edge20_body'])} | {_fmt(rr['adverse'])} | "
                     f"{_fmt(rr['adverse_before'])} | "
                     f"{_fmt(rr['mfe_d_med'],2)} | {_fmt(rr['mae_d_med'],2)} |")
        L.append("")

    # ---------- D_sum variant ----------
    L.append("## D_sum target variant (15min, overall)\n")
    L.append("| tol | N | P(hit≤20) body | P(hit≤20) D_sum | base20 body | base20 D_sum |")
    L.append("|---|---|---|---|---|---|")
    for tol in TOLS:
        r = _cell(stats, "15min", tol, 0.0, "overall", "ALL")
        if r is None:
            continue
        L.append(f"| {tol:g} | {int(r['n'])} | {_fmt(r['p_hit20_body'])} | "
                 f"{_fmt(r['p_hit20_sum'])} | {_fmt(r['base20_body'])} | "
                 f"{_fmt(r['base20_sum'])} |")
    L.append("")

    # ---------- size filter switch ----------
    L.append("## Size filter switch (mean body >= k*ATR, 15min, overall)\n")
    L.append("| tol | k | N | P(hit≤20) | base20 | P(touch start first) |")
    L.append("|---|---|---|---|---|---|")
    for tol in TOLS:
        for k in ATR_MULTS:
            r = _cell(stats, "15min", tol, k, "overall", "ALL")
            if r is None:
                continue
            L.append(f"| {tol:g} | {k:g} | {int(r['n'])} | {_fmt(r['p_hit20_body'])} | "
                     f"{_fmt(r['base20_body'])} | {_fmt(r['adverse_before'])} |")
    L.append("")

    # ---------- timeframe comparison ----------
    L.append("## Timeframe comparison (tol=0.40, k=0, overall)\n")
    L.append("| freq | N | P(hit≤20) | base20 | edge20 | P(touch start first) | "
             "median bars-to-target |")
    L.append("|---|---|---|---|---|---|---|")
    for freq in FREQS:
        r = _cell(stats, freq, 0.40, 0.0, "overall", "ALL")
        if r is None:
            continue
        L.append(f"| {freq} | {int(r['n'])} | {_fmt(r['p_hit20_body'])} | "
                 f"{_fmt(r['base20_body'])} | {_fmt(r['edge20_body'])} | "
                 f"{_fmt(r['adverse_before'])} | {_fmt(r['t_med_body'],1)} |")
    L.append("")

    L.append("## Caveats\n")
    L.append("- **Overlapping / selection**: combinations come from every bar and "
             "neighbouring combinations share candles; hit probabilities are "
             "descriptive, not independent tests.\n")
    L.append("- **Base rate**: a random bar is required to travel the *same* distance "
             "from the next-bar reference; the full move from the combination start is "
             "2*D, but measurement begins one bar later, so the required forward move "
             "is target minus next open.\n")
    L.append("- **Definition-dependent**: looser `tol` admits noisier combinations; "
             "results move with the chosen tolerance.\n")
    L.append("- **IS only**: 2024-01..2026-10 (OOS) NOT touched. Any edge here is "
             "in-sample and likely overstated.\n")
    with open(os.path.join(RESULTS_DIR, "MEASURED_MOVE.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
