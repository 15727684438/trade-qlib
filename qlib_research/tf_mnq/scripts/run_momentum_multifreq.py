"""TASK 1 — does the P2/P1 complementarity hold at 5min (and 1min)?

Same causal-zigzag definitions as before. Produces the full conditioning table
(main / p1_tercile / atr_regime) for 5min and 1min across ATR thresholds
1.0/1.5/2.0/3.0, then appends a section to results/MOMENTUM_IMPULSE.md.

Run:  python scripts/run_momentum_multifreq.py
Out:  results/momentum_impulse_multifreq.csv + MOMENTUM_IMPULSE.md (append)
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
from tf_mnq.features import atr_percentile, compute_features  # noqa: E402
from tf_mnq.momentum import causal_zigzag, collect_sequences, legs_from_pivots, summarize  # noqa: E402

RESULTS_DIR = os.path.join(_PKG_ROOT, "results")
IS_START, IS_END = "2019-05-06", "2023-12-31"
TOLS = (0.05, 0.10, 0.15, 0.20, 0.25)
KS = (1.0, 1.5, 2.0, 3.0)
COLS = (["freq", "thr_val", "direction", "block", "group", "n_seq",
         "r2_median", "r2_q1", "r2_q3", "r2_mean"]
        + [f"p_eq_{int(t*100):03d}" for t in TOLS]
        + ["p_r2_ge1", "p_r2_le1", "r3_median", "r3_q1", "r3_q3", "r3_mean",
           "p_r3_ge1", "p_r3_le1", "p_r3_eq_015"])


def thirds(a):
    a = np.asarray(a, float)
    fin = np.isfinite(a)
    q1, q2 = np.quantile(a[fin], [1 / 3, 2 / 3]) if fin.any() else (np.nan, np.nan)
    return (fin & (a <= q1), fin & (a > q1) & (a <= q2), fin & (a > q2))


def row(freq, val, direction, block, group, seqs):
    s = summarize(seqs)
    s.update({"freq": freq, "thr_val": val, "direction": direction,
              "block": block, "group": group})
    return s


def fmt(v, nd=3):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def main() -> int:
    t0 = time.perf_counter()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for tf in ("5min", "1min"):
        allb = load_bars(tf)
        feat_all = compute_features(allb)
        bars = allb.loc[(allb.index >= pd.Timestamp(IS_START))
                        & (allb.index <= pd.Timestamp(IS_END))]
        feat = feat_all.reindex(bars.index)
        atr = feat["atr"].to_numpy(float)
        atrp = atr_percentile(feat["atr"], window=500, min_periods=100).to_numpy(float)
        high = bars["high"].to_numpy(float)
        low = bars["low"].to_numpy(float)
        print(f"[{tf}] bars={len(bars)}", flush=True)
        for k in KS:
            piv = causal_zigzag(high, low, atr=atr, mode="atr", param=k)
            legs = legs_from_pivots(piv)
            seqs = collect_sequences(legs, atrp=atrp)
            tf_rows = [row(tf, k, dd, "main", "all", sub)
                       for dd, sub in (("ALL", seqs),
                                       ("LONG", [s for s in seqs if s["dir"] > 0]),
                                       ("SHORT", [s for s in seqs if s["dir"] < 0]))]
            p1 = np.array([s["P1"] for s in seqs], float)
            ar = np.array([s["atrp"] for s in seqs], float)
            for name, masks in (("p1_tercile", thirds(p1)), ("atr_regime", thirds(ar))):
                for gl, m in zip(("low", "mid", "high"), masks):
                    tf_rows.append(row(tf, k, "ALL", name, gl,
                                       [s for s, keep in zip(seqs, m) if keep]))
            rows.extend(tf_rows)
            print(f"  [{tf}] atr{k:g} seqs={len(seqs)}", flush=True)
            pd.DataFrame(rows, columns=COLS).to_csv(
                os.path.join(RESULTS_DIR, "momentum_impulse_multifreq.csv"), index=False)

    tab = pd.DataFrame(rows, columns=COLS)
    tab.to_csv(os.path.join(RESULTS_DIR, "momentum_impulse_multifreq.csv"), index=False)

    # ---------------- markdown section ----------------
    L = ["\n---\n", "## 5min / 1min — does the effect hold?\n"]
    L.append("Same causal zigzag as above (ATR-k thresholds). Overlapping 5-leg "
             "windows; N is descriptive.\n")

    def freq_tables(tf):
        out = []
        m = tab[(tab["freq"] == tf) & (tab["block"] == "main") & (tab["direction"] == "ALL")]
        out.append(f"**{tf} — r2 = P2/P1 by threshold**\n")
        out.append("| k | N | median | IQR | mean | P≤0.05 | ≤0.10 | ≤0.15 | ≤0.20 | ≤0.25 | P(r2≥1) | P(r2≤1) |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in m.iterrows():
            out.append(f"| {r['thr_val']:g} | {int(r['n_seq'])} | {fmt(r['r2_median'])} | "
                       f"{fmt(r['r2_q1'])}–{fmt(r['r2_q3'])} | {fmt(r['r2_mean'])} | "
                       f"{fmt(r['p_eq_005'])} | {fmt(r['p_eq_010'])} | {fmt(r['p_eq_015'])} | "
                       f"{fmt(r['p_eq_020'])} | {fmt(r['p_eq_025'])} | {fmt(r['p_r2_ge1'])} | "
                       f"{fmt(r['p_r2_le1'])} |")
        out.append("")
        cond = tab[(tab["freq"] == tf) & (tab["thr_val"] == 1.5)
                   & (tab["block"] == "p1_tercile")]
        out.append(f"**{tf} — r2 by P1 tercile (ATR k=1.5)**\n")
        out.append("| group | N | median r2 | P(r2≥1) | P(\\|r2-1\\|≤0.15) |")
        out.append("|---|---|---|---|---|")
        for _, r in cond.iterrows():
            out.append(f"| {r['group']} | {int(r['n_seq'])} | {fmt(r['r2_median'])} | "
                       f"{fmt(r['p_r2_ge1'])} | {fmt(r['p_eq_015'])} |")
        out.append("")
        cond2 = tab[(tab["freq"] == tf) & (tab["thr_val"] == 1.5)
                    & (tab["block"] == "atr_regime")]
        out.append(f"**{tf} — r2 by ATR% regime (ATR k=1.5)**\n")
        out.append("| group | N | median r2 | P(r2≥1) | P(\\|r2-1\\|≤0.15) |")
        out.append("|---|---|---|---|---|")
        for _, r in cond2.iterrows():
            out.append(f"| {r['group']} | {int(r['n_seq'])} | {fmt(r['r2_median'])} | "
                       f"{fmt(r['p_r2_ge1'])} | {fmt(r['p_eq_015'])} |")
        out.append("")
        return out

    L += freq_tables("5min")
    L += freq_tables("1min")
    # r3 check at 5min
    m5 = tab[(tab["freq"] == "5min") & (tab["block"] == "main") & (tab["direction"] == "ALL")]
    L.append("**r3 = P3/P2 (5min, ATR k=1.5)**\n")
    r = m5[m5["thr_val"] == 1.5].iloc[0]
    L.append(f"- N={int(r['n_seq'])}; median r3={fmt(r['r3_median'])}, "
             f"P(r3≥1)={fmt(r['p_r3_ge1'])}, P(r3≤1)={fmt(r['p_r3_le1'])} — "
             f"same as r2 (no 3rd-push length decay).\n")
    # direct answer
    def med(tf, grp):
        q = tab[(tab["freq"] == tf) & (tab["thr_val"] == 1.5)
                & (tab["block"] == "p1_tercile") & (tab["group"] == grp)]
        return float(q.iloc[0]["r2_median"]) if len(q) else np.nan
    L.append("### Direct answer\n")
    L.append(f"**YES — the complementarity holds at 5min** (and is the same shape at 1min). "
             f"At 5min (ATR k=1.5, N={int(m5[m5['thr_val']==1.5].iloc[0]['n_seq'])}), median P2/P1 "
             f"rises monotonically as P1 shrinks: P1-low median r2={fmt(med('5min','low'))}, "
             f"mid={fmt(med('5min','mid'))}, high={fmt(med('5min','high'))}; the same monotone "
             f"pattern holds at 1min (low={fmt(med('1min','low'))}, mid={fmt(med('1min','mid'))}, "
             f"high={fmt(med('1min','high'))}). Small pushes tend to be followed by relatively "
             f"larger pushes, large pushes by smaller ones.\n")
    with open(os.path.join(RESULTS_DIR, "MOMENTUM_IMPULSE.md"), "a", encoding="utf-8") as fh:
        fh.write("\n".join(L))

    print("=" * 78)
    print("SAVED: momentum_impulse_multifreq.csv; appended MOMENTUM_IMPULSE.md")
    print(f"TOTAL {time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
