"""TASK 2 — optimise the second-push entry/exit using the P1 complementarity.

Dynamic target E[P2|P1] (log-linear fit + P1-tercile median) vs fixed f*P1 refs,
with partial exits, structural trailing runner, direction and gate variants, on
15min (primary) and 5min. IS only, OOS never touched.

Run:  python scripts/run_momentum_opt.py
Out:  results/momentum_opt_grid.csv, momentum_opt_trades_<tag>.csv, MOMENTUM_OPT.md
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
from tf_mnq.momentum import (causal_zigzag, collect_sequences, fit_dyn_mult,  # noqa: E402
                             legs_from_pivots, run_second_push_opt, tercile_med_mult)

RESULTS_DIR = os.path.join(_PKG_ROOT, "results")
IS_START, IS_END = "2019-05-06", "2023-12-31"
COST = CostCfg(slip_ticks=1.5, commission_per_side=0.75, tick=0.25, point_value=2.0)
COST3 = CostCfg(slip_ticks=3.0, commission_per_side=0.75, tick=0.25, point_value=2.0)
PV, ACCOUNT, DD_GATE = 2.0, 50000.0, 1500.0
K = 1.5


def pf_of(pnl):
    pnl = np.asarray(pnl, float)
    if not pnl.size:
        return np.nan
    w = pnl[pnl > 0].sum(); l = -pnl[pnl < 0].sum()
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
    dd1 = float((peak - eq_usd).max()) if len(eq_usd) else 0.0
    ret = np.diff(eq_usd) / eq_usd[:-1] if len(eq_usd) > 1 else np.array([])
    sd = ret.std(ddof=1) if ret.size > 1 else np.nan
    sharpe = ret.mean() / sd * np.sqrt(ppy) if (np.isfinite(sd) and sd > 0) else np.nan
    ln, lpf, lnet = side_stats(tr, 1)
    sn, spf, snet = side_stats(tr, -1)
    thr = sum(1 for t in tr if t.get("hit_target"))
    stp = sum(1 for t in tr if t["exit_reason"] == "stop")
    prt = sum(1 for t in tr if t.get("partial_done"))
    return {
        "tag": tag, "cost": cost_label, "freq": spec["freq"], "target": spec["target"],
        "size": spec["size"], "partial": spec["partial"], "dir": spec["dir"],
        "p1_tercile": spec.get("p1_tercile"), "regime": spec.get("regime", False),
        "atr_cap": spec.get("atr_cap"), "trail": spec.get("trail", True),
        "n": int(len(tr)), "PF": pf_of(pnl) if pnl.size else np.nan,
        "win": float((pnl > 0).mean()) if pnl.size else np.nan,
        "expectancy_usd": float(pnl.mean()) if pnl.size else np.nan,
        "worst_trade_usd": float(pnl.min()) if pnl.size else np.nan,
        "net_profit_usd": float(pnl.sum()) if pnl.size else 0.0,
        "Sharpe_usd": sharpe, "MaxDD_usd_1c": dd1, "MaxDD_usd_2c": 2.0 * dd1,
        "target_hits": int(thr), "stop_hits": int(stp), "partial_hits": int(prt),
        "long_n": ln, "long_PF": lpf, "long_net_usd": lnet,
        "short_n": sn, "short_PF": spf, "short_net_usd": snet,
    }


def spec(tag, freq, target, *, size=2, partial=True, allow_long=True,
         allow_short=True, p1_tercile=None, regime=False, atr_cap=None, trail=True):
    d = "B" if (allow_long and allow_short) else ("L" if allow_long else "S")
    return {"tag": tag, "freq": freq, "target": target, "size": size,
            "partial": partial, "allow_long": allow_long, "allow_short": allow_short,
            "dir": d, "p1_tercile": p1_tercile, "regime": regime,
            "atr_cap": atr_cap, "trail": trail}


def build_specs():
    S = []
    for f in (0.618, 0.75, 1.0, 1.618):
        S.append(spec(f"15_f{f:g}", "15min", f"f{f:g}"))
    S.append(spec("15_dyn", "15min", "dyn"))
    S.append(spec("15_tercile", "15min", "tercile"))
    S.append(spec("15_dyn_lowP1", "15min", "dyn", p1_tercile=0))
    S.append(spec("15_dyn_highP1", "15min", "dyn", p1_tercile=2))
    S.append(spec("15_dyn_regime", "15min", "dyn", regime=True))
    S.append(spec("15_dyn_atr90", "15min", "dyn", atr_cap=90.0))
    S.append(spec("15_dyn_long", "15min", "dyn", allow_long=True, allow_short=False))
    S.append(spec("15_dyn_short", "15min", "dyn", allow_long=False, allow_short=True))
    S.append(spec("15_dyn_1c_target", "15min", "dyn", size=1, partial=False, trail=False))
    # 5min: dynamic vs best-fixed refs + 1c reference
    for f in (0.75, 1.0):
        S.append(spec(f"5_f{f:g}", "5min", f"f{f:g}"))
    S.append(spec("5_dyn", "5min", "dyn"))
    S.append(spec("5_dyn_1c_target", "5min", "dyn", size=1, partial=False, trail=False))
    return S


def main() -> int:
    t0 = time.perf_counter()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    ctx = {}
    for tf in ("15min", "5min"):
        allb = load_bars(tf)
        feat_all = compute_features(allb)
        bars = allb.loc[(allb.index >= pd.Timestamp(IS_START))
                        & (allb.index <= pd.Timestamp(IS_END))].copy()
        feat = feat_all.reindex(bars.index)
        ppy = len(bars) / ((bars.index[-1] - bars.index[0]).days / 365.25)
        bars1 = load_bars("1min", start=IS_START, end=IS_END)
        atr = feat["atr"].to_numpy(float)
        piv = causal_zigzag(bars["high"].to_numpy(float), bars["low"].to_numpy(float),
                            atr=atr, mode="atr", param=K)
        legs = legs_from_pivots(piv)
        seqs = collect_sequences(legs, atrp=None)
        p1 = np.array([s["P1"] for s in seqs], float)
        q1, q2 = np.quantile(p1, [1 / 3, 2 / 3])
        dyn_no, coef = fit_dyn_mult(seqs)
        ter_fn, meds = tercile_med_mult(seqs, (q1, q2))
        ctx[tf] = {"bars": bars, "feat": feat, "bars1": bars1, "piv": piv,
                   "ppy": ppy, "dyn": dyn_no, "ter": ter_fn, "bounds": (q1, q2)}
        print(f"[{tf}] bars={len(bars)} legs={len(legs)} seqs={len(seqs)} "
              f"P1 q1/q2={q1:.1f}/{q2:.1f} dyn a,b={coef} tercile_med={meds}", flush=True)

    specs = build_specs()
    rows = []
    for sp in specs:
        t1 = time.perf_counter()
        c = ctx[sp["freq"]]
        tm = None; dyn = None
        if sp["target"].startswith("f"):
            tm = float(sp["target"][1:])
        elif sp["target"] == "dyn":
            dyn = c["dyn"]
        else:
            dyn = c["ter"]
        res = run_second_push_opt(
            c["bars"], c["bars1"], c["feat"], c["piv"], COST,
            target_mult=tm, dyn_fn=dyn, partial=sp["partial"], size=sp["size"],
            allow_long=sp["allow_long"], allow_short=sp["allow_short"],
            p1_tercile=sp["p1_tercile"], p1_bounds=c["bounds"],
            regime_gate=sp["regime"], atr_cap=sp["atr_cap"], trail=sp["trail"],
            freq_minutes=15 if sp["freq"] == "15min" else 5)
        row = metrics_row(res, "1.5t", sp["tag"], c["ppy"], sp)
        rows.append(row)
        pd.DataFrame(res["trades"]).to_csv(
            os.path.join(RESULTS_DIR, f"momentum_opt_trades_{sp['tag']}.csv"), index=False)
        print(f"  {sp['tag']:16s} n={row['n']:5d} PF={row['PF']:.3f} win={row['win']:.3f} "
              f"net=${row['net_profit_usd']:.0f} Sharpe={row['Sharpe_usd']:.2f} "
              f"DD1=${row['MaxDD_usd_1c']:.0f} DD2=${row['MaxDD_usd_2c']:.0f} "
              f"T/{row['target_hits']} S/{row['stop_hits']} "
              f"L{row['long_net_usd']:.0f}/S{row['short_net_usd']:.0f} "
              f"({time.perf_counter()-t1:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "momentum_opt_grid.csv"),
                                  index=False)

    grid = pd.DataFrame(rows)

    def get(tag, col):
        return float(grid[grid["tag"] == tag][col].iloc[0])

    ok1 = [r["tag"] for r in rows if r["MaxDD_usd_1c"] <= DD_GATE]
    ok2 = [r["tag"] for r in rows if r["MaxDD_usd_2c"] <= DD_GATE]
    best1 = max(ok1, key=lambda t: get(t, "net_profit_usd")) if ok1 else None
    best2 = max(ok2, key=lambda t: get(t, "net_profit_usd")) if ok2 else None
    best_any = max([r["tag"] for r in rows], key=lambda t: get(t, "net_profit_usd"))

    # 3t stress of the best-by-net config (and of best1 if it exists and differs)
    stress_tags = []
    if best1:
        stress_tags.append(best1)
    if best_any not in stress_tags:
        stress_tags.append(best_any)
    stress_rows = []
    for tag in stress_tags:
        sp = next(s for s in specs if s["tag"] == tag)
        c = ctx[sp["freq"]]
        tm = float(sp["target"][1:]) if sp["target"].startswith("f") else None
        dyn = None if tm is not None else (c["dyn"] if sp["target"] == "dyn" else c["ter"])
        res = run_second_push_opt(
            c["bars"], c["bars1"], c["feat"], c["piv"], COST3,
            target_mult=tm, dyn_fn=dyn, partial=sp["partial"], size=sp["size"],
            allow_long=sp["allow_long"], allow_short=sp["allow_short"],
            p1_tercile=sp["p1_tercile"], p1_bounds=c["bounds"],
            regime_gate=sp["regime"], atr_cap=sp["atr_cap"], trail=sp["trail"],
            freq_minutes=15 if sp["freq"] == "15min" else 5)
        row = metrics_row(res, "3.0t", tag, c["ppy"], sp)
        stress_rows.append(row)
        print(f"  [3t] {tag:16s} n={row['n']} PF={row['PF']:.3f} "
              f"net=${row['net_profit_usd']:.0f} DD1=${row['MaxDD_usd_1c']:.0f}", flush=True)
    grid = pd.concat([grid, pd.DataFrame(stress_rows)], ignore_index=True)
    grid.to_csv(os.path.join(RESULTS_DIR, "momentum_opt_grid.csv"), index=False)

    # dynamic vs best fixed
    def best_fixed(freq):
        fixed = grid[(grid["freq"] == freq) & (grid["cost"] == "1.5t")
                     & (grid["tag"].str.contains(r"^15_f|^5_f"))]
        if not len(fixed):
            return None
        return fixed.loc[fixed["net_profit_usd"].idxmax(), "tag"]
    comparisons = []
    for freq, dyn_tag in (("15min", "15_dyn"), ("5min", "5_dyn")):
        bf = best_fixed(freq)
        if bf:
            comparisons.append((freq, bf, dyn_tag,
                                get(bf, "net_profit_usd"), get(dyn_tag, "net_profit_usd"),
                                get(bf, "MaxDD_usd_1c"), get(dyn_tag, "MaxDD_usd_1c")))

    print("=" * 78)
    print(f"best @1c<=1500: {best1}  best @2c<=1500: {best2}  best any: {best_any}")
    for freq, bf, dt, bfn, dnn, bfd, dnd in comparisons:
        print(f"  [{freq}] best-fixed={bf} net=${bfn:.0f} DD1=${bfd:.0f} | "
              f"dyn net=${dnn:.0f} DD1=${dnd:.0f}")

    # ---------------- MOMENTUM_OPT.md ----------------
    L = ["# MOMENTUM OPT — dynamic (P2|P1) target vs fixed f*P1 (IS only)\n"]
    L.append("Second-push entry (break of push1 extreme during push2), stop beyond the "
             "correction extreme with a 2.0*ATR floor, partial at the target (1 of 2 "
             "contracts, >=1 tick adverse), runner structural trail (pivot n=2, buf 0.25 "
             "ATR, ratchet-only). Causal zigzag ATR k=1.5. $0.75/side, $2/pt.\n")
    L.append(f"- **Best @1c<=$1500**: {best1 if best1 else 'NONE'}\n")
    L.append(f"- **Best @2c<=$1500**: {best2 if best2 else 'NONE'}\n")
    L.append(f"- **Best by net (ungated)**: `{best_any}` net=${get(best_any,'net_profit_usd'):.0f} "
             f"(MaxDD@1c ${get(best_any,'MaxDD_usd_1c'):.0f})\n")
    L.append("## Dynamic vs best fixed target\n")
    L.append("| freq | best fixed | fixed net$ | dyn net$ | fixed MaxDD1c$ | dyn MaxDD1c$ | dyn beats fixed (net)? |")
    L.append("|---|---|---|---|---|---|---|")
    for freq, bf, dt, bfn, dnn, bfd, dnd in comparisons:
        L.append(f"| {freq} | {bf} | ${bfn:.0f} | ${dnn:.0f} | ${bfd:.0f} | ${dnd:.0f} | "
                 f"{'YES' if dnn > bfn else 'NO'} |")
    L.append("")
    L.append("## Grid (IS, 1.5t; plus 3t stress of best)\n")
    L.append("| tag | cost | freq | target | dir | n | PF | win | exp$ | net$ | Sharpe$ | MaxDD1c$ | MaxDD2c$ | target | stop | partial | L net$ | S net$ |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in grid.iterrows():
        L.append(
            f"| {r['tag']} | {r['cost']} | {r['freq']} | {r['target']} | {r['dir']} | "
            f"{int(r['n'])} | {r['PF']:.3f} | {r['win']:.3f} | ${r['expectancy_usd']:.2f} | "
            f"${r['net_profit_usd']:.0f} | {r['Sharpe_usd']:.2f} | ${r['MaxDD_usd_1c']:.0f} | "
            f"${r['MaxDD_usd_2c']:.0f} | {int(r['target_hits'])} | {int(r['stop_hits'])} | "
            f"{int(r['partial_hits'])} | ${r['long_net_usd']:.0f} | ${r['short_net_usd']:.0f} |")
    L.append("")
    L.append("## Verdict\n")
    L.append(f"- {'At least one config satisfies MaxDD@1c<=$1500.' if best1 else 'NO config satisfies MaxDD@1c<=$1500.'} "
             f"{'At least one satisfies @2c.' if best2 else 'NO config satisfies MaxDD@2c<=$1500 either.'}\n")
    L.append("- OOS NOT touched.")
    L.append("")
    with open(os.path.join(RESULTS_DIR, "MOMENTUM_OPT.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))

    print(f"SAVED: momentum_opt_grid.csv, momentum_opt_trades_*.csv, MOMENTUM_OPT.md")
    print(f"TOTAL {time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
