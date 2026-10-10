"""M2-15min reporting: full performance analytics + curve quality + plots.

For each evaluated strategy (T1' sealed, T1' MAR-best, T1' smoothness-best, B2g,
B2, B0) writes results/metrics_full_15min.csv, results/equity_curves_15min.csv,
results/zseries_15min.csv and PNG plots under results/figs/. IN-SAMPLE ONLY.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(_PKG_ROOT))

from tf_mnq import IS_END, IS_START  # noqa: E402
from tf_mnq.analytics import (  # noqa: E402
    beta_alpha_ir,
    curve_quality,
    drawdown_z,
    monthly_returns,
    monthly_z,
    plot_equity_dd_monthly,
    returns_metrics,
    rolling_return_z,
    trades_metrics,
)
from tf_mnq.data import load_bars  # noqa: E402
from tf_mnq.engine import CostCfg, buy_and_hold  # noqa: E402
from tf_mnq.features import compute_features  # noqa: E402
from tf_mnq.strategies import build_signal, run_t1, strategy_frame, veto_blocked  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "run_m1b_diagnostics", os.path.join(_SCRIPT_DIR, "run_m1b_diagnostics.py")
)
_m1b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m1b)
run_breakout = _m1b.run_breakout

RESULTS_DIR = os.path.join(_PKG_ROOT, "results")
CONFIG_DIR = os.path.join(_PKG_ROOT, "config")
FIGS_DIR = os.path.join(RESULTS_DIR, "figs")
COST = CostCfg(slip_ticks=1.5, commission_per_side=0.75, tick=0.25, point_value=2.0)
FREQ = "15min"


def print_frame(title, df, floatfmt="%.4f"):
    print(title)
    if df.empty:
        print("  (empty)")
    else:
        with pd.option_context("display.width", 260, "display.max_columns", 80):
            print(df.to_string(index=False, float_format=lambda x: floatfmt % x))
    print()


def read_kv(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#") or ":" not in s:
                continue
            k, v = s.split(":", 1)
            out[k.strip()] = v.strip().strip("'\"")
    return out


def parse_seal(path):
    kv = read_kv(path)
    trig = kv.get("trigger", "b_pullback_resumption")
    return {
        "pivot_n": int(kv.get("pivot_n", 2)),
        "buffer_mult": float(kv.get("buffer_mult", 0.25)),
        "stop_mult": float(kv.get("stop_mult", 1.5)),
        "depth_mult": float(kv.get("depth_mult", 1.0)),
        "atr_pct": float(kv.get("atr_pct", 90)),
        "trigger": trig,
    }


def config_from_row(r):
    return {"pivot_n": int(r.pivot_n), "buffer_mult": float(r.buffer_mult),
            "stop_mult": float(r.stop_mult), "depth_mult": float(r.depth_mult),
            "atr_pct": float(r.atr_pct)}


def z_summary(eq, ppy):
    mr = monthly_returns(eq)
    mz = monthly_z(eq)
    rz = rolling_return_z(eq, ppy)
    dz = drawdown_z(eq)
    return {
        "monthly_ret_mean": float(mr.mean()) if len(mr) else np.nan,
        "monthly_ret_std": float(mr.std(ddof=1)) if len(mr) > 1 else np.nan,
        "monthly_z_last": float(mz.iloc[-1]) if len(mz) else np.nan,
        "rolling_return_z_last": float(rz.iloc[-1]) if np.isfinite(rz.iloc[-1]) else np.nan,
        "rolling_return_z_mean": float(rz.mean()),
        "pct_rolling_z_lt_m2": float((rz < -2).mean()),
        "drawdown_z_last": float(dz.iloc[-1]),
        "drawdown_z_min": float(dz.min()),
    }


def z_series(eq, ppy):
    mr = monthly_returns(eq)
    rz = rolling_return_z(eq, ppy)
    dz = drawdown_z(eq)
    out = pd.DataFrame({
        "monthly_z": monthly_z(eq),
        "rolling_return_z": rz.resample("ME").last(),
        "drawdown_z": dz.resample("ME").last(),
        "monthly_ret": mr,
    })
    return out


def full_row(name, res, bars, b0_eq, ppy):
    eq = res["equity"]
    rm = returns_metrics(eq, ppy, bars.index)
    tm = trades_metrics(res["trades"], len(bars), ppy)
    ba = beta_alpha_ir(eq, b0_eq, ppy)
    cq = curve_quality(eq, ppy)
    zs = z_summary(eq, ppy)
    row = {"strategy": name, "freq": FREQ, "n_bars": len(bars)}
    row.update(rm)
    row.update(tm)
    row.update(ba)
    row.update(zs)
    row.update(cq)
    return row, eq, res["trades"]


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGS_DIR, exist_ok=True)

    dec = read_kv(os.path.join(RESULTS_DIR, "m1b15_decisions.yaml"))
    trigger = dec.get("d2_primary", "b_pullback_resumption")
    return_kv = dec.get("d3_keep", "")
    comps = []
    for tok, comp in (("VR", "vr"), ("ATRpct", "vol")):
        if tok in return_kv:
            comps.append(comp)
    comps = sorted(set(comps)) or ["vol", "vr"]

    seal = parse_seal(os.path.join(CONFIG_DIR, "t1_sealed_15min.yaml"))
    trigger = seal.get("trigger", trigger)
    grid = pd.read_csv(os.path.join(RESULTS_DIR, "m2_15min_grid.csv"))
    mar_best = config_from_row(grid.loc[grid["MAR"].idxmax()])
    smooth_best = config_from_row(grid.loc[grid["smoothness_score"].idxmax()])
    print(f"trigger={trigger} vetoes={comps}")
    print(f"sealed params   : {seal}")
    print(f"MAR-best params : {mar_best} (MAR={grid['MAR'].max():.4f})")
    print(f"smooth params   : {smooth_best} (score={grid['smoothness_score'].max():.4f})\n")

    bars = load_bars(FREQ, start=IS_START, end=IS_END)
    bars1 = load_bars("1min", start=IS_START, end=IS_END)
    feat = compute_features(bars)
    sf = strategy_frame(bars, feat)
    atr = feat["atr"]
    ppy = len(bars) / ((bars.index[-1] - bars.index[0]).days / 365.25)
    assert bars.index.max() <= pd.Timestamp(IS_END)

    b0 = buy_and_hold(bars, COST, atr=atr)
    b0_eq = b0["equity"]

    t0 = time.perf_counter()
    strategies = {}
    configs = [("T1p_sealed", seal), ("T1p_marbest", mar_best), ("T1p_smooth", smooth_best)]
    for name, cfg in configs:
        sig = build_signal(bars, feat, atr, trigger=trigger, depth_mult=cfg["depth_mult"],
                           veto_components=comps, atr_pct_cap=cfg["atr_pct"])
        strategies[name] = run_t1(bars, bars1, feat, sig, COST, pivot_n=cfg["pivot_n"],
                                  buffer_mult=cfg["buffer_mult"], stop_mult=cfg["stop_mult"],
                                  freq_minutes=15)
    blocked = veto_blocked(bars, feat, sf, components=comps)
    strategies["B2g"] = run_breakout(bars, atr, COST, allow_short=False, veto_long=blocked)
    strategies["B2"] = run_breakout(bars, atr, COST, allow_short=False)
    strategies["B0"] = b0

    rows = []
    curves = {}
    zframes = {}
    trade_ledgers = {}
    for name, res in strategies.items():
        row, eq, trades = full_row(name, res, bars, b0_eq, ppy)
        rows.append(row)
        curves[name] = eq
        zframes[name] = z_series(eq, ppy)
        trade_ledgers[name] = trades

    metrics = pd.DataFrame(rows)
    # order columns: identity first, then risk, trades, z, curve quality
    front = ["strategy", "freq", "n_bars"]
    metrics = metrics[front + [c for c in metrics.columns if c not in front]]

    eq_curves = pd.concat(curves, axis=1)
    eq_curves.columns = list(curves.keys())
    eq_curves.to_csv(os.path.join(RESULTS_DIR, "equity_curves_15min.csv"))

    # z-series: long frame (month, strategy, metric)
    zparts = []
    for name, zf in zframes.items():
        z = zf.copy()
        z.index.name = "month"
        z["strategy"] = name
        zparts.append(z.reset_index())
    zlong = pd.concat(zparts, ignore_index=True)
    zlong.to_csv(os.path.join(RESULTS_DIR, "zseries_15min.csv"), index=False)

    metrics.to_csv(os.path.join(RESULTS_DIR, "metrics_full_15min.csv"), index=False)

    # grid z-score (MAR z already in grid; recompute defensively)
    gz = grid.copy()
    mstd = gz["MAR"].std(ddof=1)
    gz["mar_z"] = (gz["MAR"] - gz["MAR"].mean()) / mstd if mstd else 0.0
    gz.to_csv(os.path.join(RESULTS_DIR, "m2_15min_grid.csv"), index=False)

    # plots
    for name, eq in curves.items():
        plot_equity_dd_monthly(eq, os.path.join(FIGS_DIR, f"{name}_equity.png"), name)
    print(f"plots -> {FIGS_DIR}{os.sep}*_equity.png\n")

    show = ["strategy", "profit_factor", "win_rate", "n_trades", "CAGR", "ann_vol", "Sharpe", "Sortino",
            "Calmar", "MAR", "Ulcer", "MaxDD", "max_dd_duration_days", "time_underwater",
            "recovery_factor", "beta_vs_B0", "alpha_ann_vs_B0", "info_ratio_vs_B0",
            "expectancy_R", "time_in_market"]
    print_frame("METRICS_FULL (15min IS @1.5 ticks)", metrics[show])

    show2 = ["strategy", "r2_log_equity_vs_time", "pct_positive_months", "pct_positive_rolling_12m",
             "worst_12m_return", "worst_month", "best_month", "equity_curve_autocorr",
             "monthly_z_last", "rolling_return_z_last", "drawdown_z_last", "pct_rolling_z_lt_m2"]
    print_frame("CURVE QUALITY + Z (15min IS)", metrics[show2])

    # acceptance @1.5 ticks (T1' sealed)
    t1 = metrics[metrics["strategy"] == "T1p_sealed"].iloc[0]
    pf_ok = bool(t1["profit_factor"] >= 1.5)
    hit_ok = bool(t1["win_rate"] >= 0.30)
    print("=" * 78)
    print("ACCEPTANCE @1.5 ticks (T1' sealed, IS only)")
    print("=" * 78)
    print(f"  PF >= 1.5       : {'PASS' if pf_ok else 'FAIL'}  (PF={t1['profit_factor']:.4f})")
    print(f"  win rate >= 30% : {'PASS' if hit_ok else 'FAIL'}  (win={t1['win_rate']:.4f})")
    print(f"  n_trades        : {int(t1['n_trades'])} (no hard floor; report n)")
    sr = metrics[metrics["strategy"] != "B0"]
    best_pf = sr.loc[sr["profit_factor"].idxmax()]
    print(f"  best PF in set  : {best_pf['strategy']} PF={best_pf['profit_factor']:.4f} "
          f"win={best_pf['win_rate']:.4f} MAR={best_pf['MAR']:.4f}")
    print("=" * 78)
    print(f"runtime={time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
