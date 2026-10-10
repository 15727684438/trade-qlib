"""FROZEN single OOS evaluation — config A_T2 (from the 15min V2 grid).

ONE config, ONE touch. No tuning, no grid, no second config on OOS.

STEP 1: reproduce the IS metrics of A_T2 and compare to the frozen grid row.
STEP 2: run A_T2 once on OOS = 2024-01-01..2026-10-02 and score the pre-registered
        gates.

Run:  python scripts/run_oos_A_T2.py
Out:  results/oos_A_T2_is_check.csv, oos_A_T2.csv, oos_A_T2_trades.csv,
      section appended to results/STRATEGY_V2.md
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
from tf_mnq.strategies_v2 import build_signals_v2, regime_series, run_ema_struct  # noqa: E402

RESULTS_DIR = os.path.join(_PKG_ROOT, "results")
IS_START, IS_END = "2019-05-06", "2023-12-31"
OOS_START, OOS_END = "2024-01-01", "2026-10-02"
COST = CostCfg(slip_ticks=1.5, commission_per_side=0.75, tick=0.25, point_value=2.0)
COST3 = CostCfg(slip_ticks=3.0, commission_per_side=0.75, tick=0.25, point_value=2.0)
PV, ACCOUNT = 2.0, 50000.0
ANCH = dict(pivot_n=2, buf=0.25, floor_mult=2.0, act_mult=0.0,
            regime_confirm=0, breakeven_atr=None)
IS_EXPECT = dict(n=774, PF=1.1854617937806449, win=0.36175710594315247,
                 net=4895.9267484771735, MaxDD=1361.8885113636861,
                 Sharpe=0.724259381747148)
DD_GATE = 1500.0
Z = 1.6  # gate 5 multiplier


def pf_of(pnl):
    pnl = np.asarray(pnl, float)
    if not pnl.size:
        return np.nan
    w = pnl[pnl > 0].sum()
    l = -pnl[pnl < 0].sum()
    return float(w / l) if l > 0 else (np.inf if w > 0 else 0.0)


def metrics(res, ppy):
    idx = res["equity"].index
    eq = res["equity"].to_numpy(float)
    cap = float(res["capital"])
    eq_usd = ACCOUNT + (eq - 1.0) * cap
    tr = res["trades"]
    pnl = np.array([t["pnl_usd_net"] for t in tr], float) if tr else np.array([])
    peak = np.maximum.accumulate(eq_usd)
    maxdd1 = float((peak - eq_usd).max()) if len(eq_usd) else 0.0
    years = (idx[-1] - idx[0]).days / 365.25
    end = float(eq_usd[-1])
    cagr = (end / ACCOUNT) ** (1.0 / years) - 1.0 if end > 0 and years > 0 else np.nan
    ret = np.diff(eq_usd) / eq_usd[:-1] if len(eq_usd) > 1 else np.array([])
    sd = ret.std(ddof=1) if ret.size > 1 else np.nan
    sharpe = ret.mean() / sd * np.sqrt(ppy) if (np.isfinite(sd) and sd > 0) else np.nan
    mar = cagr / (maxdd1 / ACCOUNT) if (np.isfinite(cagr) and maxdd1 > 0) else np.nan
    return {
        "n": int(len(tr)), "PF": pf_of(pnl) if pnl.size else np.nan,
        "win": float((pnl > 0).mean()) if pnl.size else np.nan,
        "net_profit_usd": float(pnl.sum()) if pnl.size else 0.0,
        "expectancy_usd": float(pnl.mean()) if pnl.size else np.nan,
        "worst_trade_usd": float(pnl.min()) if pnl.size else np.nan,
        "Sharpe_usd": sharpe, "MaxDD_usd_1c": maxdd1, "MaxDD_usd_2c": 2.0 * maxdd1,
        "CAGR_usd": cagr, "MAR": mar, "years": years,
    }


def run_config(allb, feat_all, sf_all, cost, p0, p1):
    bars = allb.loc[(allb.index >= pd.Timestamp(p0)) & (allb.index <= pd.Timestamp(p1))].copy()
    feat = feat_all.reindex(bars.index)
    sf = sf_all.reindex(bars.index)
    ppy = len(bars) / ((bars.index[-1] - bars.index[0]).days / 365.25)
    bars1 = load_bars("1min", start=p0, end=p1)
    # regime / signal on the FULL causal history, then sliced (live-faithful)
    _, _, e144_full = regime_series(allb)
    sig_full, _, _ = build_signals_v2(allb, feat_all, sf_all, "T2", rth=False)
    e144 = e144_full.reindex(bars.index)
    sig = sig_full.reindex(bars.index)
    res = run_ema_struct(bars, bars1, feat, sig, cost, exit_ema=e144, **ANCH)
    return res, ppy, bars, sig


def main() -> int:
    t0 = time.perf_counter()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    allb = load_bars("15min")
    feat_all = compute_features(allb)
    sf_all = strategy_frame(allb, feat_all)
    print(f"full 15min bars={len(allb)}  {allb.index[0]} .. {allb.index[-1]}", flush=True)

    # ---------------- STEP 1: IS reproduction ----------------
    res_is, ppy_is, bars_is, _ = run_config(allb, feat_all, sf_all, COST, IS_START, IS_END)
    m_is = metrics(res_is, ppy_is)
    checks = [
        ("n", m_is["n"], IS_EXPECT["n"], 0),
        ("PF", m_is["PF"], IS_EXPECT["PF"], 1e-6),
        ("win", m_is["win"], IS_EXPECT["win"], 1e-6),
        ("net_profit_usd", m_is["net_profit_usd"], IS_EXPECT["net"], 1e-6),
        ("MaxDD_usd", m_is["MaxDD_usd_1c"], IS_EXPECT["MaxDD"], 1e-6),
        ("Sharpe_usd", m_is["Sharpe_usd"], IS_EXPECT["Sharpe"], 1e-6),
    ]
    ok = True
    rows = []
    for name, got, exp, tol in checks:
        diff = abs(float(got) - float(exp))
        passed = diff <= tol
        ok = ok and passed
        rows.append({"metric": name, "computed": got, "expected": exp,
                     "abs_diff": diff, "tol": tol, "pass": passed})
        print(f"  IS check {name:16s} got={got:.6f} exp={exp:.6f} diff={diff:.3e} "
              f"{'OK' if passed else 'FAIL'}", flush=True)
    ck = pd.DataFrame(rows)
    ck.to_csv(os.path.join(RESULTS_DIR, "oos_A_T2_is_check.csv"), index=False)
    if not ok:
        print("STEP 1 FAILED — reproduction mismatch. STOPPING (no OOS).")
        return 1
    print("STEP 1 OK — IS reproduces the frozen A_T2 grid row.", flush=True)

    # ---------------- STEP 2: single OOS run ----------------
    res_oos, ppy_oos, bars_oos, _ = run_config(
        allb, feat_all, sf_all, COST, OOS_START, OOS_END)
    m_oos = metrics(res_oos, ppy_oos)
    tr = res_oos["trades"]
    pnl = np.array([t["pnl_usd_net"] for t in tr], float) if tr else np.array([])

    # per-year
    per_year = {}
    for y in (2024, 2025, 2026):
        p = np.array([t["pnl_usd_net"] for t in tr
                      if pd.Timestamp(t["entry_time"]).year == y], float)
        per_year[y] = {"n": int(p.size), "PF": pf_of(p) if p.size else np.nan,
                       "net": float(p.sum()) if p.size else 0.0}

    # 3t (engine uses fixed tick fills, so identical to 1.5t — reported for the record)
    res_oos3, _, _, _ = run_config(allb, feat_all, sf_all, COST3, OOS_START, OOS_END)
    m_oos3 = metrics(res_oos3, ppy_oos)

    # ---------------- gates ----------------
    g1 = bool(m_oos["MaxDD_usd_1c"] <= DD_GATE)
    g2 = bool(m_oos["expectancy_usd"] > 0)
    g3 = bool(m_oos["PF"] >= 1.05)
    g4 = bool(all(per_year[y]["PF"] >= 1.0 for y in (2024, 2025, 2026)))
    g5 = bool(m_oos["MaxDD_usd_1c"] <= Z * IS_EXPECT["MaxDD"])
    overall = bool(g1 and g2 and g3 and g4 and g5)

    out = {
        "config": "A_T2", "os_start": OOS_START, "os_end": OOS_END,
        "n": m_oos["n"], "PF": m_oos["PF"], "win": m_oos["win"],
        "net_profit_usd": m_oos["net_profit_usd"],
        "expectancy_usd": m_oos["expectancy_usd"],
        "Sharpe_usd": m_oos["Sharpe_usd"], "MaxDD_usd_1c": m_oos["MaxDD_usd_1c"],
        "MaxDD_usd_2c": m_oos["MaxDD_usd_2c"],
        "worst_trade_usd": m_oos["worst_trade_usd"], "CAGR_usd": m_oos["CAGR_usd"],
        "MAR": m_oos["MAR"],
        "net_3t": m_oos3["net_profit_usd"], "PF_3t": m_oos3["PF"],
        "MaxDD_1c_3t": m_oos3["MaxDD_usd_1c"],
        "y2024_n": per_year[2024]["n"], "y2024_PF": per_year[2024]["PF"],
        "y2024_net": per_year[2024]["net"],
        "y2025_n": per_year[2025]["n"], "y2025_PF": per_year[2025]["PF"],
        "y2025_net": per_year[2025]["net"],
        "y2026_n": per_year[2026]["n"], "y2026_PF": per_year[2026]["PF"],
        "y2026_net": per_year[2026]["net"],
        "gate1_dd1c<=1500": g1, "gate2_exp>0": g2, "gate3_PF>=1.05": g3,
        "gate4_yearly_PF>=1.0": g4, "gate5_dd<=1.6xIS": g5,
        "OVERALL_PASS": overall,
        "IS_n": m_is["n"], "IS_PF": m_is["PF"], "IS_win": m_is["win"],
        "IS_net": m_is["net_profit_usd"], "IS_MaxDD_1c": m_is["MaxDD_usd_1c"],
        "IS_Sharpe": m_is["Sharpe_usd"],
    }
    pd.DataFrame([out]).to_csv(os.path.join(RESULTS_DIR, "oos_A_T2.csv"), index=False)
    pd.DataFrame(tr).to_csv(os.path.join(RESULTS_DIR, "oos_A_T2_trades.csv"), index=False)

    print("=" * 78)
    print(f"OOS A_T2  n={m_oos['n']} PF={m_oos['PF']:.3f} win={m_oos['win']:.3f} "
          f"net=${m_oos['net_profit_usd']:.0f} exp=${m_oos['expectancy_usd']:.2f} "
          f"Sharpe={m_oos['Sharpe_usd']:.2f} MaxDD1c=${m_oos['MaxDD_usd_1c']:.0f} "
          f"MaxDD2c=${m_oos['MaxDD_usd_2c']:.0f} worst=${m_oos['worst_trade_usd']:.0f}")
    print("per-year:", {y: (per_year[y]["n"], round(per_year[y]["PF"], 3)
                             if np.isfinite(per_year[y]["PF"]) else None,
                             round(per_year[y]["net"])) for y in per_year})
    for name, val in (("gate1 DD1c<=1500", g1), ("gate2 exp>0", g2),
                      ("gate3 PF>=1.05", g3), ("gate4 yearly PF>=1.0", g4),
                      ("gate5 DD<=1.6xIS", g5)):
        print(f"  {name:22s}: {'PASS' if val else 'FAIL'}")
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")

    # ---------------- append STRATEGY_V2.md ----------------
    L = []
    L.append("\n---\n")
    L.append("## OOS verdict — A_T2 (single pre-registered touch, 2024-01-01..2026-10-02)\n")
    L.append(f"- IS reproduction: **{'OK' if ok else 'FAILED'}** "
             f"(n={m_is['n']}, PF {m_is['PF']:.3f}, win {m_is['win']:.3f}, "
             f"net ${m_is['net_profit_usd']:.2f}, MaxDD ${m_is['MaxDD_usd_1c']:.2f}, "
             f"Sharpe {m_is['Sharpe_usd']:.3f}) — matches frozen grid row A_T2.\n")
    L.append("| period | n | PF | win | net$ | expectancy$ | Sharpe$ | MaxDD@1c$ | MaxDD@2c$ | worst$ | CAGR$ | MAR |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    L.append(f"| IS 2019-05..2023-12 | {m_is['n']} | {m_is['PF']:.3f} | {m_is['win']:.3f} | "
             f"${m_is['net_profit_usd']:.0f} | ${m_is['expectancy_usd']:.2f} | {m_is['Sharpe_usd']:.2f} | "
             f"${m_is['MaxDD_usd_1c']:.0f} | ${m_is['MaxDD_usd_2c']:.0f} | ${m_is['worst_trade_usd']:.0f} | "
             f"{m_is['CAGR_usd']*100:.2f}% | {m_is['MAR']:.2f} |")
    L.append(f"| OOS 2024-01..2026-10 | {m_oos['n']} | {m_oos['PF']:.3f} | {m_oos['win']:.3f} | "
             f"${m_oos['net_profit_usd']:.0f} | ${m_oos['expectancy_usd']:.2f} | {m_oos['Sharpe_usd']:.2f} | "
             f"${m_oos['MaxDD_usd_1c']:.0f} | ${m_oos['MaxDD_usd_2c']:.0f} | ${m_oos['worst_trade_usd']:.0f} | "
             f"{m_oos['CAGR_usd']*100:.2f}% | {m_oos['MAR']:.2f} |")
    L.append("")
    L.append(f"- 3t for the record: net ${m_oos3['net_profit_usd']:.0f}, PF {m_oos3['PF']:.3f}, "
             f"MaxDD@1c ${m_oos3['MaxDD_usd_1c']:.0f} (identical to 1.5t: `run_ema_struct` uses "
             f"fixed 1/2/1-tick fills, so cost.slip_ticks does not bind here).\n")
    L.append("### Per calendar year (OOS)\n")
    L.append("| year | n | PF | net$ |")
    L.append("|---|---|---|---|")
    for y in (2024, 2025, 2026):
        pfv = f"{per_year[y]['PF']:.3f}" if np.isfinite(per_year[y]["PF"]) else "—"
        L.append(f"| {y} | {per_year[y]['n']} | {pfv} | ${per_year[y]['net']:.0f} |")
    L.append("")
    L.append("### Pre-registered gates\n")
    L.append("| # | gate | actual | verdict |")
    L.append("|---|---|---|---|")
    L.append(f"| 1 | OOS MaxDD@1c <= $1,500 | ${m_oos['MaxDD_usd_1c']:.0f} | {'PASS' if g1 else 'FAIL'} |")
    L.append(f"| 2 | OOS expectancy > 0 | ${m_oos['expectancy_usd']:.2f} | {'PASS' if g2 else 'FAIL'} |")
    L.append(f"| 3 | OOS PF >= 1.05 | {m_oos['PF']:.3f} | {'PASS' if g3 else 'FAIL'} |")
    L.append(f"| 4 | every year PF >= 1.0 | 2024={per_year[2024]['PF']:.3f}, "
             f"2025={per_year[2025]['PF']:.3f}, 2026={per_year[2026]['PF']:.3f} | "
             f"{'PASS' if g4 else 'FAIL'} |")
    L.append(f"| 5 | OOS MaxDD@1c <= 1.6x IS (${Z*IS_EXPECT['MaxDD']:.0f}) | "
             f"${m_oos['MaxDD_usd_1c']:.0f} | {'PASS' if g5 else 'FAIL'} |")
    L.append("")
    L.append(f"**OVERALL OOS: {'PASS' if overall else 'FAIL'}** — "
             f"{'all five gates hold' if overall else 'at least one gate failed; the V2 family is closed (no second touch, no nudging, no re-tuning).'}\n")
    L.append(f"### IS -> OOS degradation\n")
    L.append(f"- PF {m_is['PF']:.3f} -> {m_oos['PF']:.3f}; win {m_is['win']:.3f} -> "
             f"{m_oos['win']:.3f}; MaxDD@1c ${m_is['MaxDD_usd_1c']:.0f} -> "
             f"${m_oos['MaxDD_usd_1c']:.0f}; expectancy ${m_is['expectancy_usd']:.2f} -> "
             f"${m_oos['expectancy_usd']:.2f}.\n")
    with open(os.path.join(RESULTS_DIR, "STRATEGY_V2.md"), "a", encoding="utf-8") as fh:
        fh.write("\n".join(L))

    print("SAVED: oos_A_T2_is_check.csv, oos_A_T2.csv, oos_A_T2_trades.csv; appended STRATEGY_V2.md")
    print(f"TOTAL {time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
