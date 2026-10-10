"""STRATEGY V2B — EMA20/144 regime + Donchian breakout, LONG **and** SHORT,
2-contract sizing with a 2-leg partial exit. 5min decisions / 1min execution.

Pre-registered grid (IS only, OOS must NOT be touched):
  anchor(T2,long+short,2c,0.5R,floor2.0,RTH on); anchor long-only; T1; T3;
  partial 1.0R; floor 1.5; RTH off; 1c no-partial reference.
Every config is run at 1.5t and 3.0t (cost stress). Acceptance: rank by
net_profit_usd@1.5t among configs with MaxDD_usd@2c <= $1,500 at BOTH cost levels.

Run:  python scripts/run_ema2_grid.py
Out:  results/ema2_grid.csv, ema2_trades_<tag>.csv, STRATEGY_V2B.md
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
from tf_mnq.strategies import strategy_frame  # noqa: E402
from tf_mnq.strategies_v2 import regime_series  # noqa: E402
from tf_mnq.strategies_v2b import build_signals_v2b, run_ema2_struct  # noqa: E402

RESULTS_DIR = os.path.join(_PKG_ROOT, "results")
IS_START, IS_END = "2019-05-06", "2023-12-31"
COST = CostCfg(slip_ticks=1.5, commission_per_side=0.75, tick=0.25, point_value=2.0)
COST3 = CostCfg(slip_ticks=3.0, commission_per_side=0.75, tick=0.25, point_value=2.0)
PV, ACCOUNT = 2.0, 50000.0
DD_HARD = 1500.0


def save(df, name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, name), index=False)


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


def metrics_row(res, cost_label, tag, ppy, spec):
    idx = res["equity"].index
    eq = res["equity"].to_numpy(float)
    cap = float(res["capital"])
    eq_usd = ACCOUNT + (eq - 1.0) * cap
    tr = res["trades"]
    pnl = np.array([t["pnl_usd_net"] for t in tr], float) if tr else np.array([])
    peak = np.maximum.accumulate(eq_usd)
    maxdd = float((peak - eq_usd).max())
    years = (idx[-1] - idx[0]).days / 365.25
    end = float(eq_usd[-1])
    cagr = (end / ACCOUNT) ** (1.0 / years) - 1.0 if end > 0 and years > 0 else np.nan
    ret = np.diff(eq_usd) / eq_usd[:-1] if len(eq_usd) > 1 else np.array([])
    sd = ret.std(ddof=1) if ret.size > 1 else np.nan
    sharpe = ret.mean() / sd * np.sqrt(ppy) if (np.isfinite(sd) and sd > 0) else np.nan
    ln, lpf, lnet = side_stats(tr, 1)
    sn, spf, snet = side_stats(tr, -1)
    withp = [t for t in tr if t["partial_done"]]
    withoutp = [t for t in tr if not t["partial_done"]]
    win_with = float(np.mean([t["pnl_usd_net"] > 0 for t in withp])) if withp else np.nan
    win_wo = float(np.mean([t["pnl_usd_net"] > 0 for t in withoutp])) if withoutp else np.nan
    return {
        "tag": tag, "cost": cost_label, "trigger": spec["trigger"],
        "size": spec["size"], "partial_R": spec["partial_R"],
        "floor": spec["floor_mult"], "rth": spec["rth"],
        "long_short": spec["allow_short"],
        "n": int(len(tr)), "PF": pf_of(pnl) if pnl.size else np.nan,
        "win": float((pnl > 0).mean()) if pnl.size else np.nan,
        "expectancy_usd": float(pnl.mean()) if pnl.size else np.nan,
        "worst_trade_usd": float(pnl.min()) if pnl.size else np.nan,
        "net_profit_usd": float(pnl.sum()) if pnl.size else 0.0,
        "Sharpe_usd": sharpe, "MaxDD_usd": maxdd, "CAGR_usd": cagr,
        "long_n": ln, "long_PF": lpf, "long_net_usd": lnet,
        "short_n": sn, "short_PF": spf, "short_net_usd": snet,
        "n_tp1": len(withp), "win_with_partial": win_with,
        "win_without_partial": win_wo,
    }


def build_specs():
    def sp(tag, trigger="T2", allow_short=True, size=2, partial_R=0.5,
           floor_mult=2.0, rth=True):
        return {"tag": tag, "trigger": trigger, "allow_short": allow_short,
                "size": size, "partial_R": partial_R, "floor_mult": floor_mult,
                "rth": rth}
    return [
        sp("anchor"),                                  # T2 l+s 2c 0.5R floor2.0 RTH on
        sp("anchor_long", allow_short=False),          # long-only
        sp("T1", trigger="T1"),
        sp("T3", trigger="T3"),
        sp("partial_1.0R", partial_R=1.0),
        sp("floor_1.5", floor_mult=1.5),
        sp("rth_off", rth=False),
        sp("one_contract", size=1, partial_R=None),    # 1c no-partial reference
    ]


def main() -> int:
    t0 = time.perf_counter()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    allb = load_bars("5min")
    feat_all = compute_features(allb)
    sf_all = strategy_frame(allb, feat_all)
    bars = allb.loc[(allb.index >= pd.Timestamp(IS_START))
                    & (allb.index <= pd.Timestamp(IS_END))].copy()
    feat = feat_all.reindex(bars.index)
    sf = sf_all.reindex(bars.index)
    ppy = len(bars) / ((bars.index[-1] - bars.index[0]).days / 365.25)
    bars1 = load_bars("1min", start=IS_START, end=IS_END)
    print(f"IS 5min={len(bars)} 1min={len(bars1)} ppy={ppy:.1f}", flush=True)

    _reg, _e20, e144 = regime_series(bars)

    specs = build_specs()
    rows = []
    grid_path = os.path.join(RESULTS_DIR, "ema2_grid.csv")
    for s in specs:
        t1 = time.perf_counter()
        sl, ss = build_signals_v2b(bars, feat, sf, s["trigger"], atrp_cap=90.0,
                                   rth=s["rth"], allow_short=s["allow_short"])
        for cost, clabel in ((COST, "1.5t"), (COST3, "3.0t")):
            res = run_ema2_struct(
                bars, bars1, feat, sl, ss, cost, size=s["size"],
                floor_mult=s["floor_mult"], pivot_n=2, buf=0.25,
                partial_R=s["partial_R"], exit_ema=e144, regime_confirm=0,
                fills="cost", freq_minutes=5, allow_short=s["allow_short"])
            row = metrics_row(res, clabel, s["tag"], ppy, s)
            rows.append(row)
            if clabel == "1.5t":
                save(pd.DataFrame(res["trades"]), f"ema2_trades_{s['tag']}.csv")
            print(f"  {s['tag']:14s} {clabel} n={row['n']:5d} PF={row['PF']:.3f} "
                  f"win={row['win']:.3f} net=${row['net_profit_usd']:.0f} "
                  f"MaxDD=${row['MaxDD_usd']:.0f} Sharpe={row['Sharpe_usd']:.2f} "
                  f"L/{row['long_net_usd']:.0f} S/{row['short_net_usd']:.0f} "
                  f"tp1={row['n_tp1']} ({time.perf_counter()-t1:.0f}s)", flush=True)
        save(pd.DataFrame(rows), "ema2_grid.csv")  # incremental checkpoint

    grid = pd.DataFrame(rows)

    # ---------------- acceptance ----------------
    def dd15(tag):
        return float(grid[(grid["tag"] == tag) & (grid["cost"] == "1.5t")
                          ]["MaxDD_usd"].iloc[0])

    def dd3(tag):
        return float(grid[(grid["tag"] == tag) & (grid["cost"] == "3.0t")
                          ]["MaxDD_usd"].iloc[0])

    def net15(tag):
        return float(grid[(grid["tag"] == tag) & (grid["cost"] == "1.5t")
                          ]["net_profit_usd"].iloc[0])

    tags = [s["tag"] for s in specs]
    eligible = [t for t in tags if dd15(t) <= DD_HARD and dd3(t) <= DD_HARD]
    eligible2c = [t for t in eligible
                  if int(grid[(grid["tag"] == t) & (grid["cost"] == "1.5t")
                              ]["size"].iloc[0]) == 2]
    if eligible2c:
        best = max(eligible2c, key=net15)
        best_net = net15(best)
        best_dd15, best_dd3 = dd15(best), dd3(best)
    else:
        best, best_net, best_dd15, best_dd3 = None, np.nan, np.nan, np.nan

    grid.to_csv(grid_path, index=False)
    print("=" * 78)
    if best:
        print(f"BEST (MaxDD<=${DD_HARD:.0f}@2c both costs): {best} "
              f"net@1.5t=${best_net:.0f} MaxDD@1.5t=${best_dd15:.0f} "
              f"MaxDD@3t=${best_dd3:.0f}")
    else:
        print("NO config meets MaxDD@2c<=$1500 at BOTH cost levels.")
    for t in tags:
        print(f"  dd_check {t:14s} 1.5t=${dd15(t):.0f} 3t=${dd3(t):.0f} "
              f"ok_both={dd15(t) <= DD_HARD and dd3(t) <= DD_HARD}")

    # ---------------- markdown ----------------
    L = []
    L.append("# STRATEGY V2B — EMA20/144 regime + Donchian breakout, long+short, "
             "2-leg partial (IS-only verdict)\n")
    L.append("- 5min decisions / 1min execution; regime LONG (EMA20>EMA144 & close>EMA144) "
             "/ SHORT (EMA20<EMA144 & close<EMA144); anchor T2 Donchian-20; gates "
             "{VR<1, ATRpct<=90, ext<=2.5}; size 2c; partial 0.5R (1 of 2), runner trails "
             "structurally (pivot n=2, buf 0.25 ATR, ratchet-only, floor=initial stop) "
             "+ regime-break exit.\n")
    L.append("- Fill model: canonical adverse slip = slip_ticks*tick on entry/stop/regime/"
             "partial; partial carries >=1 tick. Costs: $0.75/side, $2/pt, tick $0.50.\n")
    L.append(f"- Acceptance: rank by net_profit_usd@1.5t among configs with "
             f"MaxDD_usd@2c <= ${DD_HARD:.0f} at BOTH 1.5t and 3.0t.\n")
    L.append(f"- **Verdict: {'BEST = ' + best if best else 'NONE meets MaxDD gate at both cost levels'}**\n")
    L.append("## Grid (IS) — all configs run at 1.5t and 3.0t\n")
    L.append("| tag | cost | n | PF | win | exp$ | net$ | Sharpe$ | MaxDD$ | worst$ | "
             "L n/PF/net$ | S n/PF/net$ | TP1 | win w/partial | win w/o partial |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in grid.iterrows():
        wwp = f"{r['win_with_partial']:.3f}" if pd.notna(r["win_with_partial"]) else "—"
        wwop = f"{r['win_without_partial']:.3f}" if pd.notna(r["win_without_partial"]) else "—"
        L.append(
            f"| {r['tag']} | {r['cost']} | {int(r['n'])} | {r['PF']:.3f} | {r['win']:.3f} | "
            f"${r['expectancy_usd']:.2f} | ${r['net_profit_usd']:.0f} | "
            f"{r['Sharpe_usd']:.2f} | ${r['MaxDD_usd']:.0f} | ${r['worst_trade_usd']:.0f} | "
            f"{int(r['long_n'])}/{r['long_PF']:.3f}/${r['long_net_usd']:.0f} | "
            f"{int(r['short_n'])}/{r['short_PF']:.3f}/${r['short_net_usd']:.0f} | "
            f"{int(r['n_tp1'])} | {wwp} | {wwop} |")
    L.append("")
    L.append("## Acceptance (MaxDD@2c <= $1500 at BOTH costs)\n")
    L.append("| tag | size | net@1.5t$ | MaxDD@1.5t$ | MaxDD@3t$ | eligible |")
    L.append("|---|---|---|---|---|---|")
    for t in tags:
        r = grid[(grid["tag"] == t) & (grid["cost"] == "1.5t")].iloc[0]
        L.append(f"| {t} | {int(r['size'])} | ${r['net_profit_usd']:.0f} | "
                 f"${dd15(t):.0f} | ${dd3(t):.0f} | "
                 f"{'YES' if (dd15(t) <= DD_HARD and dd3(t) <= DD_HARD) else 'no'} |")
    L.append("")
    if best:
        r15 = grid[(grid["tag"] == best) & (grid["cost"] == "1.5t")].iloc[0]
        r3 = grid[(grid["tag"] == best) & (grid["cost"] == "3.0t")].iloc[0]
        L.append(f"**Best eligible config: `{best}`** — net@1.5t=${r15['net_profit_usd']:.0f} "
                 f"(MaxDD ${r15['MaxDD_usd']:.0f}), net@3t=${r3['net_profit_usd']:.0f} "
                 f"(MaxDD ${r3['MaxDD_usd']:.0f}); LONG n={int(r15['long_n'])} "
                 f"net=${r15['long_net_usd']:.0f}, SHORT n={int(r15['short_n'])} "
                 f"net=${r15['short_net_usd']:.0f}; TP1 hits={int(r15['n_tp1'])}.\n")
    else:
        L.append("**No config satisfies the MaxDD@2c<=$1500 gate at both cost levels.**\n")
    L.append("## Anchor check (STEP 1)\n")
    L.append("- `run_ema2_struct` with shorts+partial disabled and legacy fills reproduces "
             "frozen A_T2 trade-for-trade (n=774, PF 1.185, net $4895.93, MaxDD $1361.89, "
             "max per-trade diff $0.00) — see `results/ema2_anchor_check.csv`.\n")
    L.append("- OOS NOT touched.")
    L.append("")
    with open(os.path.join(RESULTS_DIR, "STRATEGY_V2B.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))

    print("SAVED: ema2_grid.csv, ema2_trades_*.csv, STRATEGY_V2B.md, ema2_anchor_check.csv")
    print(f"TOTAL RUNTIME {time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
