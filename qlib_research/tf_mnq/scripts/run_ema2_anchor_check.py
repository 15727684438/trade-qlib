"""STEP 1 verification for STRATEGY V2B.

Proves the generalised ``run_ema2_struct`` engine reproduces the frozen long-only
V2 A_T2 anchor (15min decisions, 1min execution) trade-for-trade when the SHORT
and 2-leg partial paths are switched off, using the legacy fill conventions.

Run:  python scripts/run_ema2_anchor_check.py
Out:  results/ema2_anchor_check.csv
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
from tf_mnq.strategies_v2b import build_signals_v2b, run_ema2_struct  # noqa: E402

RESULTS_DIR = os.path.join(_PKG_ROOT, "results")
IS_START, IS_END = "2019-05-06", "2023-12-31"
COST = CostCfg(slip_ticks=1.5, commission_per_side=0.75, tick=0.25, point_value=2.0)
PV, ACCOUNT = 2.0, 50000.0
ANCH = dict(pivot_n=2, buf=0.25, floor_mult=2.0, act_mult=0.0,
            regime_confirm=0, breakeven_atr=None)


def pf_of(pnl):
    pnl = np.asarray(pnl, float)
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
    maxdd = float((peak - eq_usd).max())
    years = (idx[-1] - idx[0]).days / 365.25
    end = float(eq_usd[-1])
    cagr = (end / ACCOUNT) ** (1.0 / years) - 1.0 if end > 0 and years > 0 else np.nan
    ret = np.diff(eq_usd) / eq_usd[:-1] if len(eq_usd) > 1 else np.array([])
    sd = ret.std(ddof=1) if ret.size > 1 else np.nan
    sharpe = ret.mean() / sd * np.sqrt(ppy) if (np.isfinite(sd) and sd > 0) else np.nan
    return {"n": len(tr), "PF": pf_of(pnl) if len(pnl) else np.nan,
            "win": float((pnl > 0).mean()) if len(pnl) else np.nan,
            "net_usd": float(pnl.sum()) if len(pnl) else 0.0,
            "MaxDD_usd": maxdd, "Sharpe_usd": sharpe,
            "expectancy_usd": float(pnl.mean()) if len(pnl) else np.nan,
            "worst_trade_usd": float(pnl.min()) if len(pnl) else np.nan,
            "CAGR_usd": cagr}


def main():
    t0 = time.perf_counter()
    allb = load_bars("15min")
    feat_all = compute_features(allb)
    sf_all = strategy_frame(allb, feat_all)
    bars = allb.loc[(allb.index >= pd.Timestamp(IS_START))
                    & (allb.index <= pd.Timestamp(IS_END))].copy()
    feat = feat_all.reindex(bars.index)
    sf = sf_all.reindex(bars.index)
    ppy = len(bars) / ((bars.index[-1] - bars.index[0]).days / 365.25)
    bars1 = load_bars("1min", start=IS_START, end=IS_END)
    print(f"IS 15min={len(bars)} 1min={len(bars1)} ppy={ppy:.1f}")

    _regime, _e20r, e144 = regime_series(bars)
    sig_ref, _e20, _e144 = build_signals_v2(bars, feat, sf, "T2", rth=False)
    ref = run_ema_struct(bars, bars1, feat, sig_ref, COST, exit_ema=e144, **ANCH)

    sl, ss = build_signals_v2b(bars, feat, sf, "T2", atrp_cap=90.0, rth=False,
                               allow_short=False)
    sig_match = bool((sl.reindex(bars.index).fillna(False)
                      == sig_ref.reindex(bars.index).fillna(False)).all())
    new = run_ema2_struct(bars, bars1, feat, sl, ss, COST, size=1, floor_mult=2.0,
                          pivot_n=2, buf=0.25, partial_R=None, exit_ema=e144,
                          regime_confirm=0, fills="legacy", freq_minutes=15,
                          allow_short=False)

    m_ref, m_new = metrics(ref, ppy), metrics(new, ppy)
    print("signal_match:", sig_match)
    print("REF(legacy run_ema_struct):", {k: round(v, 3) if isinstance(v, float) else v
                                          for k, v in m_ref.items()})
    print("NEW(run_ema2_struct)      :", {k: round(v, 3) if isinstance(v, float) else v
                                          for k, v in m_new.items()})

    # trade-for-trade comparison
    tr_ref = pd.DataFrame(ref["trades"])
    tr_new = pd.DataFrame(new["trades"])
    same_n = len(tr_ref) == len(tr_new)
    max_dpnl = np.nan
    if same_n and len(tr_ref):
        max_dpnl = float((tr_ref["pnl_usd_net"] - tr_new["pnl_usd_net"]).abs().max())
        times_eq = bool((pd.to_datetime(tr_ref["entry_time"]).to_numpy()
                         == pd.to_datetime(tr_new["entry_time"]).to_numpy()).all())
        exits_eq = bool((pd.to_datetime(tr_ref["exit_time"]).to_numpy()
                         == pd.to_datetime(tr_new["exit_time"]).to_numpy()).all())
    else:
        times_eq = exits_eq = False

    rows = []
    for tag, m in (("A_T2_legacy_ref", m_ref), ("A_T2_new_engine", m_new)):
        r = {"tag": tag}
        r.update(m)
        rows.append(r)
    check = pd.DataFrame(rows)
    check["same_n"] = same_n
    check["entry_times_equal"] = times_eq
    check["exit_times_equal"] = exits_eq
    check["max_abs_pnl_diff_usd"] = max_dpnl
    check["signal_match"] = sig_match
    os.makedirs(RESULTS_DIR, exist_ok=True)
    check.to_csv(os.path.join(RESULTS_DIR, "ema2_anchor_check.csv"), index=False)

    tol = 1e-6
    ok = (same_n and sig_match and times_eq and exits_eq
          and (max_dpnl is not None and max_dpnl <= tol)
          and abs(m_ref["net_usd"] - m_new["net_usd"]) <= tol
          and abs(m_ref["MaxDD_usd"] - m_new["MaxDD_usd"]) <= tol)
    print("=" * 70)
    print("ANCHOR CHECK:", "PASS" if ok else "FAIL",
          f"(same_n={same_n}, max_abs_pnl_diff={max_dpnl})")
    print(f"TOTAL {time.perf_counter()-t0:.1f}s")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
