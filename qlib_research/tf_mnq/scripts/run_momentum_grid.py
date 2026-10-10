"""TASK B — "second-push momentum" prototype grid (IS only).

After push1 completes and a correction forms, enter on the break of the push1
extreme (start of push2); target = entry +/- f*P1; stop beyond the correction
extreme with a >=2.0*ATR risk floor; at most one trade per impulse sequence
(never enter the 3rd push). Long AND short. 15min decisions / 1min fills.

Run:  python scripts/run_momentum_grid.py
Out:  results/momentum_grid.csv, momentum_trades_<tag>.csv, MOMENTUM_IMPULSE.md (append)
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
from tf_mnq.engine import CostCfg  # noqa: E402
from tf_mnq.features import compute_features  # noqa: E402
from tf_mnq.momentum import causal_zigzag, run_second_push  # noqa: E402

RESULTS_DIR = os.path.join(_PKG_ROOT, "results")
IS_START, IS_END = "2019-05-06", "2023-12-31"
COST = CostCfg(slip_ticks=1.5, commission_per_side=0.75, tick=0.25, point_value=2.0)
COST3 = CostCfg(slip_ticks=3.0, commission_per_side=0.75, tick=0.25, point_value=2.0)
PV, ACCOUNT, DD_GATE = 2.0, 50000.0, 1500.0

VARIANTS = [
    dict(tag="k1.5_f0.618", kind="atr", val=1.5, f=0.618),
    dict(tag="k1.5_f0.75", kind="atr", val=1.5, f=0.75),
    dict(tag="k1.5_f1.0", kind="atr", val=1.5, f=1.0),
    dict(tag="k1.0_f1.0", kind="atr", val=1.0, f=1.0),
    dict(tag="k2.0_f1.0", kind="atr", val=2.0, f=1.0),
    dict(tag="pct0.5_f1.0", kind="pct", val=0.005, f=1.0),
]


def pf_of(pnl):
    pnl = np.asarray(pnl, float)
    if not pnl.size:
        return np.nan
    w = pnl[pnl > 0].sum()
    l = -pnl[pnl < 0].sum()
    return float(w / l) if l > 0 else (np.inf if w > 0 else 0.0)


def side_stats(trades, side):
    sub = [t for t in trades if t["side"] == side]
    pnl = np.array([t["pnl_usd_net"] for t in sub], float) if sub else np.array([])
    return len(sub), pf_of(pnl), (float(pnl.sum()) if pnl.size else 0.0)


def metrics_row(res, cost_label, tag, ppy, kind, val, f):
    idx = res["equity"].index
    eq = res["equity"].to_numpy(float)
    cap = float(res["capital"])
    eq_usd = ACCOUNT + (eq - 1.0) * cap          # @1 contract
    tr = res["trades"]
    pnl = np.array([t["pnl_usd_net"] for t in tr], float) if tr else np.array([])
    peak = np.maximum.accumulate(eq_usd)
    maxdd1 = float((peak - eq_usd).max()) if len(eq_usd) else 0.0
    maxdd2 = 2.0 * maxdd1
    ret = np.diff(eq_usd) / eq_usd[:-1] if len(eq_usd) > 1 else np.array([])
    sd = ret.std(ddof=1) if ret.size > 1 else np.nan
    sharpe = ret.mean() / sd * np.sqrt(ppy) if (np.isfinite(sd) and sd > 0) else np.nan
    ln, lpf, lnet = side_stats(tr, 1)
    sn, spf, snet = side_stats(tr, -1)
    rs = res["reasons"]
    return {
        "tag": tag, "cost": cost_label, "zigzag": f"{kind}{val:g}", "f": f,
        "n": int(len(tr)), "PF": pf_of(pnl) if pnl.size else np.nan,
        "win": float((pnl > 0).mean()) if pnl.size else np.nan,
        "expectancy_usd": float(pnl.mean()) if pnl.size else np.nan,
        "worst_trade_usd": float(pnl.min()) if pnl.size else np.nan,
        "net_profit_usd": float(pnl.sum()) if pnl.size else 0.0,
        "Sharpe_usd": sharpe, "MaxDD_usd_1c": maxdd1, "MaxDD_usd_2c": maxdd2,
        "target_hits": int(rs.get("target", 0)), "stop_hits": int(rs.get("stop", 0)),
        "end_hits": int(rs.get("end", 0)),
        "long_n": ln, "long_PF": lpf, "long_net_usd": lnet,
        "short_n": sn, "short_PF": spf, "short_net_usd": snet,
    }


def main() -> int:
    t0 = time.perf_counter()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    allb = load_bars("15min")
    feat_all = compute_features(allb)
    bars = allb.loc[(allb.index >= pd.Timestamp(IS_START))
                    & (allb.index <= pd.Timestamp(IS_END))].copy()
    feat = feat_all.reindex(bars.index)
    ppy = len(bars) / ((bars.index[-1] - bars.index[0]).days / 365.25)
    bars1 = load_bars("1min", start=IS_START, end=IS_END)
    print(f"IS 15min={len(bars)} 1min={len(bars1)} ppy={ppy:.1f}", flush=True)

    high = bars["high"].to_numpy(float)
    low = bars["low"].to_numpy(float)
    atr = feat["atr"].to_numpy(float)

    piv_cache = {}

    def pivots_for(kind, val):
        key = (kind, val)
        if key not in piv_cache:
            piv_cache[key] = causal_zigzag(high, low, atr=atr, mode=kind, param=val)
        return piv_cache[key]

    rows = []
    for v in VARIANTS:
        t1 = time.perf_counter()
        piv = pivots_for(v["kind"], v["val"])
        for cost, clabel in ((COST, "1.5t"), (COST3, "3.0t")):
            res = run_second_push(bars, bars1, feat, piv, cost, f=v["f"],
                                  stop_atr_floor=2.0, freq_minutes=15)
            row = metrics_row(res, clabel, v["tag"], ppy, v["kind"], v["val"], v["f"])
            rows.append(row)
            if clabel == "1.5t":
                pd.DataFrame(res["trades"]).to_csv(
                    os.path.join(RESULTS_DIR, f"momentum_trades_{v['tag']}.csv"),
                    index=False)
            print(f"  {v['tag']:12s} {clabel} n={row['n']:4d} PF={row['PF']:.3f} "
                  f"win={row['win']:.3f} net=${row['net_profit_usd']:.0f} "
                  f"Sharpe={row['Sharpe_usd']:.2f} DD1=${row['MaxDD_usd_1c']:.0f} "
                  f"DD2=${row['MaxDD_usd_2c']:.0f} T/{row['target_hits']} "
                  f"S/{row['stop_hits']} ({time.perf_counter()-t1:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "momentum_grid.csv"),
                                  index=False)

    grid = pd.DataFrame(rows)
    # 3t stress of the best-by-net config (all already run at 3t)
    def get(tag, cost, col):
        return float(grid[(grid["tag"] == tag) & (grid["cost"] == cost)][col].iloc[0])

    tags = [v["tag"] for v in VARIANTS]
    ok1 = [t for t in tags if get(t, "1.5t", "MaxDD_usd_1c") <= DD_GATE]
    ok2 = [t for t in tags if get(t, "1.5t", "MaxDD_usd_2c") <= DD_GATE
           and get(t, "3.0t", "MaxDD_usd_2c") <= DD_GATE]
    best1 = max(ok1, key=lambda t: get(t, "1.5t", "net_profit_usd")) if ok1 else None
    best2 = max(ok2, key=lambda t: get(t, "1.5t", "net_profit_usd")) if ok2 else None

    print("=" * 78)
    print(f"best @1c<=${DD_GATE:.0f} 1.5t: {best1} "
          f"net=${get(best1,'1.5t','net_profit_usd'):.0f}" if best1
          else f"best @1c<=${DD_GATE:.0f} 1.5t: NONE")
    print(f"best @2c<=${DD_GATE:.0f} both costs: {best2} "
          f"1.5t net=${get(best2,'1.5t','net_profit_usd'):.0f} "
          f"3t net=${get(best2,'3.0t','net_profit_usd'):.0f}" if best2
          else f"best @2c<=${DD_GATE:.0f} both costs: NONE")
    for t in tags:
        print(f"  dd {t:12s} 1.5t DD1=${get(t,'1.5t','MaxDD_usd_1c'):.0f} "
              f"DD2=${get(t,'1.5t','MaxDD_usd_2c'):.0f} | 3t "
              f"DD1=${get(t,'3.0t','MaxDD_usd_1c'):.0f} "
              f"DD2=${get(t,'3.0t','MaxDD_usd_2c'):.0f}")

    # ---------------- append markdown ----------------
    L = []
    L.append("\n---\n")
    L.append("## TASK B — second-push prototype (15min decisions / 1min fills)\n")
    L.append("Rule: after push1 (causal zigzag leg0) + correction (leg1), enter on the "
             "break of the push1 extreme (start of push2); target = entry +/- f*P1; stop "
             "beyond the correction extreme, risk floored at 2.0*ATR; one trade per "
             "impulse sequence (no 3rd push). $0.75/side, $2/pt, tick $0.50.\n")
    L.append("**Best @1c<=$1500 (1.5t)**: " + (f"`{best1}` net=${get(best1,'1.5t','net_profit_usd'):.0f}" if best1 else "NONE") + "\n")
    L.append("**Best @2c<=$1500 at BOTH costs**: " + (f"`{best2}`" if best2 else "NONE") + "\n")
    L.append("| tag | cost | n | PF | win | exp$ | net$ | Sharpe$ | MaxDD1c$ | MaxDD2c$ | worst$ | target | stop | end | L n/PF/net$ | S n/PF/net$ |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in grid.iterrows():
        L.append(
            f"| {r['tag']} | {r['cost']} | {int(r['n'])} | {r['PF']:.3f} | {r['win']:.3f} | "
            f"${r['expectancy_usd']:.2f} | ${r['net_profit_usd']:.0f} | {r['Sharpe_usd']:.2f} | "
            f"${r['MaxDD_usd_1c']:.0f} | ${r['MaxDD_usd_2c']:.0f} | ${r['worst_trade_usd']:.0f} | "
            f"{int(r['target_hits'])} | {int(r['stop_hits'])} | {int(r['end_hits'])} | "
            f"{int(r['long_n'])}/{r['long_PF']:.3f}/${r['long_net_usd']:.0f} | "
            f"{int(r['short_n'])}/{r['short_PF']:.3f}/${r['short_net_usd']:.0f} |")
    L.append("")
    with open(os.path.join(RESULTS_DIR, "MOMENTUM_IMPULSE.md"), "a", encoding="utf-8") as fh:
        fh.write("\n".join(L))

    print("SAVED: momentum_grid.csv, momentum_trades_*.csv; appended MOMENTUM_IMPULSE.md")
    print(f"TOTAL {time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
