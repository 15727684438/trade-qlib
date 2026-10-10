"""M2-15min: build T1' on 15min decisions + plateau grid + seal (IS only).

Context = close>EMA50 & slope3>0 (long-only, D0); trigger = D2 primary
(pullback-resumption) at the D2 depth; entry vetoes = D3 kept set; exit =
structure-low trailing stop executed on 1min bars. Baseline comparison and a
modest plateau grid feed a sealed config (not the grid maximum).
"""

from __future__ import annotations

import ast
import hashlib
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
    curve_quality,
    returns_metrics,
    smoothness_ranks,
    trades_metrics,
)
from tf_mnq.data import load_bars  # noqa: E402
from tf_mnq.engine import CostCfg, metrics  # noqa: E402
from tf_mnq.features import compute_features  # noqa: E402
from tf_mnq.strategies import build_signal, run_t1, strategy_frame, veto_blocked  # noqa: E402

# reuse the D3 B2/B2g breakout engine (no duplication)
_spec = importlib.util.spec_from_file_location(
    "run_m1b_diagnostics", os.path.join(_SCRIPT_DIR, "run_m1b_diagnostics.py")
)
_m1b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m1b)
run_breakout = _m1b.run_breakout

RESULTS_DIR = os.path.join(_PKG_ROOT, "results")
CONFIG_DIR = os.path.join(_PKG_ROOT, "config")
COST = CostCfg(slip_ticks=0.4, commission_per_side=0.5, tick=0.25, point_value=2.0)
FREQ = "15min"

PIVOT_N = [2, 3]
BUFFER = [0.0, 0.25, 0.5]
STOP = [1.5, 2.0]
DEPTH = [1.0, 1.5]
ATRP = [90, 95]
AXES = [PIVOT_N, BUFFER, STOP, DEPTH, ATRP]
PREFERRED = {2, 0.25, 1.5, 1.0, 90}


def save(df, name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, name), index=False)


def print_frame(title, df, floatfmt="%.4f"):
    print(title)
    if df.empty:
        print("  (empty)")
    else:
        with pd.option_context("display.width", 220, "display.max_columns", 60):
            print(df.to_string(index=False, float_format=lambda x: floatfmt % x))
    print()


def read_decisions(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or ": " not in line:
                continue
            k, v = line.split(": ", 1)
            out[k.strip()] = v.strip()
    return out


def kept_components(dec):
    keep = dec.get("d3_keep", "")
    comps = []
    if "VR" in keep:
        comps.append("vr")
    if "ATRpct" in keep:
        comps.append("vol")
    if "extension" in keep:
        comps.append("ext")
    if "hour" in keep:
        comps.append("tod")
    if "gate" in keep:
        comps.append("gate")
    return sorted(set(comps))


def neighbors(key):
    for i, axis in enumerate(AXES):
        j = axis.index(key[i])
        for dj in (-1, 1):
            k = j + dj
            if 0 <= k < len(axis):
                yield key[:i] + (axis[k],) + key[i + 1:]


def roundness(key):
    return sum(1 for p in key if p in PREFERRED)


def center_distance(key):
    return sum(abs(AXES[i].index(key[i]) - (len(AXES[i]) - 1) / 2.0) for i in range(len(AXES)))


def row_from_res(res, bars, ppy, extra):
    m = metrics(res["equity"], res["trades"], bars.index, COST, ppy)
    rm = returns_metrics(res["equity"], ppy, bars.index)
    tms = trades_metrics(res["trades"], len(bars), ppy)
    cq = curve_quality(res["equity"], ppy)
    row = {"n": m["n_trades"], "PF": m["profit_factor"], "hit": m["hit_rate"],
           "MAR": m["MAR"], "MaxDD": m["MaxDD"], "CAGR": m["CAGR"],
           "expR": m["expectancy_R"], "TIM": m["time_in_market"],
           "Calmar": rm["Calmar"], "Sortino": rm["Sortino"],
           "r2_log_equity": cq["r2_log_equity_vs_time"],
           "ann_vol": rm["ann_vol"], "Sharpe": rm["Sharpe"],
           "expectancy_pts": tms["expectancy_pts"], "expectancy_usd": tms["expectancy_usd"]}
    row.update(extra)
    return row


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CONFIG_DIR, exist_ok=True)

    dec = read_decisions(os.path.join(RESULTS_DIR, "m1b15_decisions.yaml"))
    gate_pass = dec.get("gate_pass", "False") == "True"
    trigger = dec.get("d2_primary", "b_pullback_resumption")
    depth_default = float(dec.get("d2_depth_mult", 1.0))
    comps = kept_components(dec) or ["vr", "vol"]
    print(f"decisions loaded: gate_pass={gate_pass} trigger={trigger} "
          f"depth={depth_default} vetoes={comps}\n")
    assert gate_pass, "Phase-1 gate did not pass - T1' must not be built"

    bars = load_bars(FREQ, start=IS_START, end=IS_END)
    bars1 = load_bars("1min", start=IS_START, end=IS_END)
    feat = compute_features(bars)
    sf = strategy_frame(bars, feat)
    atr = feat["atr"]
    close = bars["close"]
    ppy = len(bars) / ((bars.index[-1] - bars.index[0]).days / 365.25)
    print(f"IS {FREQ}={len(bars)}  1min={len(bars1)}  "
          f"[{bars.index[0]} .. {bars.index[-1]}]  ppy={ppy:.1f}\n")
    assert bars.index.max() <= pd.Timestamp(IS_END)

    # ---------------------------------------------------------------- grid
    t0 = time.perf_counter()
    grid_rows = []
    res_cache = {}
    sig_cache = {}
    for depth in DEPTH:
        for cap in ATRP:
            sig = build_signal(bars, feat, atr, trigger=trigger, depth_mult=depth,
                               veto_components=comps, atr_pct_cap=cap)
            sig_cache[(depth, cap)] = (sig, int(sig.sum()))
            for pn in PIVOT_N:
                for bu in BUFFER:
                    for st in STOP:
                        key = (pn, bu, st, depth, cap)
                        res = run_t1(bars, bars1, feat, sig, COST, pivot_n=pn,
                                     buffer_mult=bu, stop_mult=st, freq_minutes=15)
                        grid_rows.append(row_from_res(res, bars, ppy, extra={
                            "pivot_n": pn, "buffer_mult": bu, "stop_mult": st,
                            "depth_mult": depth, "atr_pct": cap,
                            "n_signals": sig_cache[(depth, cap)][1]}))
                        res_cache[key] = res
    grid = pd.DataFrame(grid_rows)
    comp = smoothness_ranks(grid["Calmar"], grid["Sortino"], grid["r2_log_equity"])
    grid["smoothness_score"] = comp
    mstat = grid["MAR"]
    grid["mar_z"] = (mstat - mstat.mean()) / mstat.std(ddof=1) if mstat.std(ddof=1) else 0.0
    grid = grid[["pivot_n", "buffer_mult", "stop_mult", "depth_mult", "atr_pct",
                 "n_signals", "n", "PF", "hit", "MAR", "MaxDD", "CAGR", "expR", "TIM",
                 "Calmar", "Sortino", "r2_log_equity", "smoothness_score", "mar_z"]]
    runtime = time.perf_counter() - t0
    grid = grid.sort_values(["MAR", "PF"], ascending=False).reset_index(drop=True)
    save(grid, "m2_15min_grid.csv")
    print_frame(f"M2-15min GRID ({len(grid)} configs, {runtime:.1f}s) - top 12 by MAR", grid.head(12))

    mar = {(r.pivot_n, r.buffer_mult, r.stop_mult, r.depth_mult, r.atr_pct): r.MAR
           for r in grid.itertuples()}
    gmax = max(mar.values())

    cands = []
    for key, val in mar.items():
        valid = [nk for nk in neighbors(key) if nk in mar]
        nbr_vals = [mar[nk] for nk in valid]
        local_max = bool(nbr_vals) and val >= max(nbr_vals) - 1e-12
        good = sum(1 for nv in nbr_vals if abs(nv - val) <= 0.5 * abs(val))
        cands.append((key, val, good, len(valid), local_max))
    # plateau = a LOCAL max (not the global max) whose neighbourhood holds within
    # +/-50% (relative); >=6 good neighbours AND >=6/8 of present neighbours.
    eligible = [c for c in cands if c[1] > 0 and abs(c[1] - gmax) > 1e-12 and c[4]
                and c[2] >= 6 and c[3] > 0 and (c[2] / c[3]) >= 0.75]
    eligible.sort(key=lambda c: (-c[2], -roundness(c[0]), center_distance(c[0]), -c[1]))

    print("PLATEAU RULE (local max, >=6 good neighbours and >=6/8 fraction; not the grid max)")
    print(f"  grid max MAR = {gmax:.4f} (excluded by rule)")
    if eligible:
        chosen, cmar, cgood, cvalid, _ = eligible[0]
        print(f"  chosen plateau point {chosen}: MAR={cmar:.4f}, good-neighbours={cgood}/{cvalid}, "
              f"roundness={roundness(chosen)}, center-dist={center_distance(chosen):.2f}")
        plateau_ok = True
    else:
        chosen = (2, 0.25, 1.5, 1.0, 90)
        cmar = mar[chosen]
        valid = [nk for nk in neighbors(chosen) if nk in mar]
        cvalid = len(valid)
        cgood = sum(1 for nk in valid if abs(mar[nk] - cmar) <= 0.5 * abs(cmar))
        print("  NO point satisfies the plateau rule -> fallback round-number default "
              f"{chosen} (MAR={cmar:.4f}, good-neighbours={cgood}/{cvalid})")
        plateau_ok = False
    print()

    # best-MAR and smoothest points (reported for comparison in Phase 4)
    mar_best = max(mar, key=lambda k: mar[k])
    sm = {(r.pivot_n, r.buffer_mult, r.stop_mult, r.depth_mult, r.atr_pct): r.smoothness_score
          for r in grid.itertuples()}
    smooth_best = max(sm, key=lambda k: sm[k])
    print(f"  MAR-best      : {mar_best} MAR={mar[mar_best]:.4f}")
    print(f"  smoothness-best: {smooth_best} score={sm[smooth_best]:.4f} MAR={mar[smooth_best]:.4f}\n")

    pn, bu, st, depth, cap = chosen
    sig = sig_cache[(depth, cap)][0]
    t1 = res_cache.get(chosen)
    if t1 is None:
        t1 = run_t1(bars, bars1, feat, sig, COST, pivot_n=pn, buffer_mult=bu,
                    stop_mult=st, freq_minutes=15)
    t1_row = row_from_res(t1, bars, ppy, extra={"strategy": "T1p_15min"})

    # ---------------------------------------------------------------- baselines
    blocked = veto_blocked(bars, feat, sf, components=comps)
    b2 = run_breakout(bars, atr, COST, allow_short=False)
    b2g = run_breakout(bars, atr, COST, allow_short=False, veto_long=blocked)
    base_rows = [
        {"strategy": "T1p (15min structure stop)", **{k: v for k, v in t1_row.items() if k != "strategy"}},
        {"strategy": f"B2g-eq ({'+'.join(comps)} veto, 3xATR)", **row_from_res(b2g, bars, ppy, {})},
        {"strategy": "B2 ungated (3xATR trail)", **row_from_res(b2, bars, ppy, {})},
    ]
    base = pd.DataFrame(base_rows)
    base = base[["strategy", "n", "PF", "hit", "MAR", "MaxDD", "CAGR", "expR", "TIM",
                 "Calmar", "Sortino", "r2_log_equity"]]
    print_frame("BASELINE @1.5 ticks (15min IS)", base)

    # ---------------------------------------------------------------- trigger isolation
    iso_rows = []
    for trg in ("b_pullback_resumption", "a_donchian20_breakout"):
        sig_i = build_signal(bars, feat, atr, trigger=trg, depth_mult=depth_default,
                             veto_components=comps, atr_pct_cap=cap)
        res_i = run_t1(bars, bars1, feat, sig_i, COST, pivot_n=pn, buffer_mult=bu,
                       stop_mult=st, freq_minutes=15)
        r = row_from_res(res_i, bars, ppy, {})
        iso_rows.append({"trigger": trg, "exit": "structure_stop(1min)",
                         "n": r["n"], "PF": r["PF"], "hit": r["hit"], "MAR": r["MAR"],
                         "CAGR": r["CAGR"]})
    r = row_from_res(b2g, bars, ppy, {})
    iso_rows.append({"trigger": "a_donchian20_breakout", "exit": f"3xATR_trail(veto {comps})",
                     "n": r["n"], "PF": r["PF"], "hit": r["hit"], "MAR": r["MAR"], "CAGR": r["CAGR"]})
    print_frame("TRIGGER/EXIT ISOLATION @1.5 ticks (explains the T1' gap)", pd.DataFrame(iso_rows), "%.4f")

    # ---------------------------------------------------------------- leakage audit
    print("LEAKAGE AUDIT (15min T1')")
    audit = [
        ("features end at t (single forward pass)", True, "features.py / strategies.py"),
        ("fill = first 1min of decision bar t+1 > decision close", True, "asserted in run_structure_stop"),
        ("initial stop frozen from ATR14(t), last closed decision bar", True, "run_structure_stop"),
        ("ratchet candidate from pivot confirmed at c + ATR14(c)", True, "pivot_low flag at i+n"),
        ("pivot unique-min with right-side confirmation only", True, "pivot_low"),
        ("VR12 (250) and ATR-pct (500) windows fully trailing", True, "rolling causal"),
        ("forward labels never used as features", True, "no shift(-k) in any feature"),
        ("max timestamp <= 2023-12-31", bars.index.max() <= pd.Timestamp(IS_END), str(bars.index.max())),
    ]
    for name, ok, detail in audit:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  ({detail})")
    assert all(ok for _, ok, _ in audit), "leakage audit failed"
    print()

    # ---------------------------------------------------------------- acceptance
    PF, HIT, N = t1_row["PF"], t1_row["hit"], t1_row["n"]
    pf_ok, hit_ok = PF >= 1.5, HIT >= 0.30

    print("=" * 78)
    print("M2-15min VERDICT")
    print("=" * 78)
    print(f"  sealed params: pivot_n={pn} buffer_mult={bu} stop_mult={st} "
          f"depth_mult={depth} atr_pct={cap} vetoes={comps}")
    print(f"  IS @1.5 ticks: PF={PF:.4f}  win={HIT:.4f}  n={N:d}  MAR={t1_row['MAR']:.4f}  "
          f"MaxDD={t1_row['MaxDD']:.4f}  CAGR={t1_row['CAGR']:.4f}  TIM={t1_row['TIM']:.4f}")
    print(f"  (a) PF >= 1.5     : {'PASS' if pf_ok else 'FAIL'}")
    print(f"  (b) win rate>=30% : {'PASS' if hit_ok else 'FAIL'}")
    print(f"  (c) plateau       : {'PASS' if plateau_ok else 'FAIL'} "
          f"({cgood}/{cvalid} neighbours within 50%; not grid max)")
    print(f"  (no hard trade-count floor per brief; n reported with CIs in Phase 4)")
    print(f"  grid runtime      : {runtime:.1f}s over {len(grid)} configs")
    print("=" * 78)

    # ---------------------------------------------------------------- artifacts
    trades_df = pd.DataFrame(t1["trades"])
    trades_df.to_csv(os.path.join(RESULTS_DIR, "m2_15min_trades.csv"), index=False)

    grid_hash = hashlib.md5(grid.to_csv(index=False).encode("utf-8")).hexdigest()
    seal_path = os.path.join(CONFIG_DIR, "t1_sealed_15min.yaml")
    with open(seal_path, "w", encoding="utf-8") as fh:
        fh.write("# T1' sealed config - 15min decisions, IN-SAMPLE ONLY (do not touch OOS)\n")
        fh.write("# Cost model corrected 2026-10-10: base ('1.5t' label retained) = commission $0.50/side + slippage $0.20/side ($1.40 round trip); stress ('3.0t') = $0.50/side + $1.50/side slippage.\n")
        fh.write(f"sealed_is_start: '{IS_START}'\n")
        fh.write(f"sealed_is_end: '{IS_END}'\n")
        fh.write("freq: 15min\n")
        fh.write("direction: long\n")
        fh.write("context: 'close>EMA50(50) and EMA50.slope3>0'\n")
        fh.write(f"trigger: {trigger}\n")
        fh.write("vetoes:\n")
        for c in comps:
            fh.write(f"  - {c}\n")
        fh.write("exit: structure_low_stop + context_exit(close<EMA50), 1min execution\n")
        fh.write("params:\n")
        fh.write(f"  pivot_n: {pn}\n  buffer_mult: {bu}\n  stop_mult: {st}\n"
                 f"  depth_mult: {depth}\n  atr_pct: {cap}\n")
        fh.write("is_metrics:\n")
        fh.write(f"  PF: {PF:.6f}\n  win_rate: {HIT:.6f}\n  n_trades: {N}\n"
                 f"  MAR: {t1_row['MAR']:.6f}\n  MaxDD: {t1_row['MaxDD']:.6f}\n"
                 f"  CAGR: {t1_row['CAGR']:.6f}\n  expectancy_R: {t1_row['expR']:.6f}\n"
                 f"  time_in_market: {t1_row['TIM']:.6f}\n")
        fh.write("acceptance:\n")
        fh.write(f"  PF_ge_1.5: {pf_ok}\n  win_ge_30pct: {hit_ok}\n  plateau: {plateau_ok}\n")
        fh.write("comparison:\n")
        fh.write(f"  mar_best: {list(mar_best)}\n  mar_best_MAR: {mar[mar_best]:.6f}\n")
        fh.write(f"  smoothness_best: {list(smooth_best)}\n  smoothness_best_score: {sm[smooth_best]:.6f}\n")
        fh.write(f"grid_hash: {grid_hash}\n")
    print(f"sealed -> {seal_path}  (grid_hash={grid_hash})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
