"""TASK A — empirical distribution of second-push vs first-push length.

Causal zigzag (ATR-k and percent thresholds) -> alternating legs -> 5-leg impulse
sequences (P1=leg0, P2=leg2, P3=leg4). Measures r2=P2/P1 and r3=P3/P2 on IS only.

Run:  python scripts/run_momentum_impulse.py
Out:  results/momentum_impulse_stats.csv, results/MOMENTUM_IMPULSE.md
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

CFGS = ([("atr", k) for k in (1.0, 1.5, 2.0, 3.0)]
        + [("pct", p) for p in (0.003, 0.005, 0.008)])
TOLS = (0.05, 0.10, 0.15, 0.20, 0.25)
COLS = (["freq", "thr_kind", "thr_val", "direction", "block", "group", "n_seq",
         "r2_median", "r2_q1", "r2_q3", "r2_mean"]
        + [f"p_eq_{int(t*100):03d}" for t in TOLS]
        + ["p_r2_ge1", "p_r2_le1",
           "r3_median", "r3_q1", "r3_q3", "r3_mean", "p_r3_ge1", "p_r3_le1",
           "p_r3_eq_015"])


def thr_label(kind, val):
    return f"atr{val:g}" if kind == "atr" else f"pct{val*100:g}"


def row_from(freq, kind, val, direction, block, group, seqs):
    s = summarize(seqs)
    s.update({"freq": freq, "thr_kind": kind, "thr_val": val,
              "direction": direction, "block": block, "group": group})
    return s


def thirds(arr):
    """Return (low_mask, mid_mask, high_mask) by terciles of finite values."""
    a = np.asarray(arr, float)
    fin = np.isfinite(a)
    q1, q2 = np.quantile(a[fin], [1 / 3, 2 / 3]) if fin.any() else (np.nan, np.nan)
    return (fin & (a <= q1), fin & (a > q1) & (a <= q2), fin & (a > q2))


def main() -> int:
    t0 = time.perf_counter()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []

    for tf in ("15min", "5min"):
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

        freq_rows = []
        for kind, val in CFGS:
            piv = causal_zigzag(high, low, atr=atr, mode=kind, param=val)
            legs = legs_from_pivots(piv)
            seqs = collect_sequences(legs, atrp=atrp)
            print(f"  [{tf}] {thr_label(kind, val):8s} pivots={len(piv):6d} "
                  f"legs={len(legs):6d} seqs={len(seqs):6d}", flush=True)

            for direction, sub in (("ALL", seqs),
                                   ("LONG", [s for s in seqs if s["dir"] > 0]),
                                   ("SHORT", [s for s in seqs if s["dir"] < 0])):
                freq_rows.append(row_from(tf, kind, val, direction, "main", "all", sub))

            # conditioning on P1 tercile / ATR% regime (ALL directions)
            p1 = np.array([s["P1"] for s in seqs], float)
            ar = np.array([s["atrp"] for s in seqs], float)
            for name, masks in (("p1_tercile", thirds(p1)), ("atr_regime", thirds(ar))):
                for glabel, m in zip(("low", "mid", "high"), masks):
                    sub = [s for s, keep in zip(seqs, m) if keep]
                    freq_rows.append(row_from(tf, kind, val, "ALL", name, glabel, sub))

            # incremental checkpoint
            pd.DataFrame(rows + freq_rows, columns=COLS).to_csv(
                os.path.join(RESULTS_DIR, "momentum_impulse_stats.csv"), index=False)
        rows.extend(freq_rows)

    tab = pd.DataFrame(rows, columns=COLS)
    tab.to_csv(os.path.join(RESULTS_DIR, "momentum_impulse_stats.csv"), index=False)

    # ---------------- markdown ----------------
    L = []
    L.append("# MOMENTUM — impulse-push statistics (IS only, 2019-05-06..2023-12-31)\n")
    L.append("Causal zigzag: a swing pivot is confirmed only after price reverses by "
             ">= threshold from the running extreme (ATR-k: threshold = k*ATR at the "
             "confirming bar; pct: threshold = x% of the extreme price). Alternating "
             "legs; every consecutive 5-leg window is an impulse sequence with "
             "P1=leg0, P2=leg2, P3=leg4. Overlapping windows are used (descriptive).\n")
    L.append("Cost model corrected 2026-10-10: commission $0.50/side ($1.00 open+close) "
             "+ slippage $0.20/side; '1.5t' label retained, stress '3.0t' = $1.50/side slip.\n")

    def fmt(v, nd=3):
        return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"

    main15 = tab[(tab["freq"] == "15min") & (tab["block"] == "main")
                 & (tab["direction"] == "ALL")]
    L.append("## Direct answer — P(2nd push ≈ 1st push length)\n")
    L.append("15min, across impulse definitions (r2 = P2/P1):\n")
    L.append("| threshold | N seq | median r2 | Q1–Q3 | mean | P(\\|r2-1\\|≤0.05) | ≤0.10 | ≤0.15 | ≤0.20 | ≤0.25 | P(r2≥1) | P(r2≤1) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in main15.iterrows():
        L.append(
            f"| {thr_label(r['thr_kind'], r['thr_val'])} | {int(r['n_seq'])} | "
            f"{fmt(r['r2_median'])} | {fmt(r['r2_q1'])}–{fmt(r['r2_q3'])} | "
            f"{fmt(r['r2_mean'])} | {fmt(r['p_eq_005'])} | {fmt(r['p_eq_010'])} | "
            f"{fmt(r['p_eq_015'])} | {fmt(r['p_eq_020'])} | {fmt(r['p_eq_025'])} | "
            f"{fmt(r['p_r2_ge1'])} | {fmt(r['p_r2_le1'])} |")
    L.append("")
    main5 = tab[(tab["freq"] == "5min") & (tab["block"] == "main")
                & (tab["direction"] == "ALL")]
    L.append("5min (secondary):\n")
    L.append("| threshold | N seq | median r2 | P(\\|r2-1\\|≤0.15) | P(r2≥1) | P(r2≤1) |")
    L.append("|---|---|---|---|---|---|")
    for _, r in main5.iterrows():
        L.append(f"| {thr_label(r['thr_kind'], r['thr_val'])} | {int(r['n_seq'])} | "
                 f"{fmt(r['r2_median'])} | {fmt(r['p_eq_015'])} | "
                 f"{fmt(r['p_r2_ge1'])} | {fmt(r['p_r2_le1'])} |")
    L.append("")
    L.append("**Headline (primary, 15min, ATR k=1.5)**: "
             + _headline(tab, "15min", "atr", 1.5) + "\n")
    L.append("## Does the data support \"avoid the 3rd push\"? (r3 = P3/P2)\n")
    L.append("| freq | threshold | N | median r3 | P(r3≥1) | P(r3≤1) | P(\\|r3-1\\|≤0.15) |")
    L.append("|---|---|---|---|---|---|---|")
    for _, r in pd.concat([main15, main5]).iterrows():
        L.append(f"| {r['freq']} | {thr_label(r['thr_kind'], r['thr_val'])} | "
                 f"{int(r['n_seq'])} | {fmt(r['r3_median'])} | {fmt(r['p_r3_ge1'])} | "
                 f"{fmt(r['p_r3_le1'])} | {fmt(r['p_r3_eq_015'])} |")
    L.append("")
    L.append("## Conditioning (15min, ATR k=1.5): does equality depend on context?\n")
    L.append("| block | group | N | median r2 | P(\\|r2-1\\|≤0.15) | P(r2≥1) |")
    L.append("|---|---|---|---|---|---|")
    cond = tab[(tab["freq"] == "15min") & (tab["thr_kind"] == "atr")
               & (tab["thr_val"] == 1.5) & (tab["block"].isin(["p1_tercile", "atr_regime"]))]
    for _, r in cond.iterrows():
        L.append(f"| {r['block']} | {r['group']} | {int(r['n_seq'])} | "
                 f"{fmt(r['r2_median'])} | {fmt(r['p_eq_015'])} | {fmt(r['p_r2_ge1'])} |")
    L.append("")
    L.append("## Caveats\n")
    L.append("- Results are **definition-dependent**: the zigzag threshold sets what "
             "counts as a push, and equality probability moves with it.\n")
    L.append("- 5-leg windows overlap, so sequence counts are not independent; treat "
             "the probabilities as descriptive, not as significance tests.\n")
    L.append("- OOS (2024-01..2026-10) NOT touched.")
    L.append("")
    with open(os.path.join(RESULTS_DIR, "MOMENTUM_IMPULSE.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))

    print("=" * 78)
    print("SAVED: momentum_impulse_stats.csv, MOMENTUM_IMPULSE.md")
    print(f"TOTAL {time.perf_counter()-t0:.1f}s")
    return 0


def _headline(tab, freq, kind, val):
    r = tab[(tab["freq"] == freq) & (tab["thr_kind"] == kind)
            & (tab["thr_val"] == val) & (tab["block"] == "main")
            & (tab["direction"] == "ALL")].iloc[0]
    return (f"N={int(r['n_seq'])} sequences; median P2/P1={r['r2_median']:.3f}, "
            f"mean={r['r2_mean']:.3f}, P(|r2-1|<=0.15)={r['p_eq_015']:.3f}, "
            f"P(|r2-1|<=0.25)={r['p_eq_025']:.3f}, P(r2>=1)={r['p_r2_ge1']:.3f}, "
            f"P(r2<=1)={r['p_r2_le1']:.3f}.")


if __name__ == "__main__":
    raise SystemExit(main())
