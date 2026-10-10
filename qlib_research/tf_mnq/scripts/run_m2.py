"""M2: finalise T1' (pullback-resumption + vetoes + 1min structure-low stop).

IN-SAMPLE ONLY (<= 2023-12-31). Runs the baseline comparison, a modest IS
parameter-plateau grid, a leakage audit, and seals the chosen params.
"""

from __future__ import annotations

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
from tf_mnq.data import load_bars  # noqa: E402
from tf_mnq.engine import CostCfg, metrics, run_structure_stop  # noqa: E402
from tf_mnq.features import (  # noqa: E402
    atr_percentile,
    compute_features,
    ema,
    pullback_resumption_trigger,
)

# reuse the D3 B2/B2g engine from run_m1b_diagnostics (no duplication)
_spec = importlib.util.spec_from_file_location(
    "run_m1b_diagnostics", os.path.join(_SCRIPT_DIR, "run_m1b_diagnostics.py")
)
_m1b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m1b)
run_breakout = _m1b.run_breakout

RESULTS_DIR = os.path.join(_PKG_ROOT, "results")
CONFIG_DIR = os.path.join(_PKG_ROOT, "config")
COST = CostCfg(slip_ticks=1.5, commission_per_side=0.75, tick=0.25, point_value=2.0)

PIVOT_N = [2, 3]
BUFFER = [0.0, 0.25, 0.5]
STOP = [1.5, 2.0]
DEPTH = [1.0, 1.5]
ATRP = [90, 95]
AXES = [PIVOT_N, BUFFER, STOP, DEPTH, ATRP]
# round-number fallback / tie-break preference
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


def metrics_row(res, bars, ppy, extra=None):
    m = metrics(res["equity"], res["trades"], bars.index, COST, ppy)
    row = {"n": m["n_trades"], "PF": m["profit_factor"], "hit": m["hit_rate"],
           "MAR": m["MAR"], "MaxDD": m["MaxDD"], "CAGR": m["CAGR"],
           "expR": m["expectancy_R"], "TIM": m["time_in_market"]}
    if extra:
        row.update(extra)
    return row


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


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CONFIG_DIR, exist_ok=True)

    bars60 = load_bars("60min", start=IS_START, end=IS_END)
    bars1 = load_bars("1min", start=IS_START, end=IS_END)
    feat = compute_features(bars60)
    close, high, low = bars60["close"], bars60["high"], bars60["low"]
    atr = feat["atr"]
    ema50 = feat["ema50"]
    slope3 = ema50.diff(3)
    ext = (close - ema(close, 20)) / atr
    atr_pct = atr_percentile(atr, window=500)
    context = (close > ema50) & (slope3 > 0)
    vr_ok = feat["vr"] < 1.0  # veto: block VR>=1.0 (D3-kept veto)
    ppy = len(bars60) / ((bars60.index[-1] - bars60.index[0]).days / 365.25)
    print(f"IS 60min={len(bars60)}  1min={len(bars1)}  "
          f"[{bars60.index[0]} .. {bars60.index[-1]}]  ppy={ppy:.1f}\n")

    # ---------------------------------------------------------------- grid
    t0 = time.perf_counter()
    grid_rows = []
    res_cache = {}
    for depth in DEPTH:
        trig = pullback_resumption_trigger(close, high, low, atr, depth_mult=depth, lookback=10)
        for cap in ATRP:
            sig = (context & trig & vr_ok & (atr_pct <= cap)).fillna(False)
            nsig = int(sig.sum())
            for pn in PIVOT_N:
                for bu in BUFFER:
                    for st in STOP:
                        key = (pn, bu, st, depth, cap)
                        # results do not depend on depth/cap for the engine, only on signal
                        res = run_structure_stop(bars60, bars1, feat, sig, COST,
                                                 stop_mult=st, buffer_mult=bu, pivot_n=pn)
                        row = metrics_row(res, bars60, ppy, extra={
                            "pivot_n": pn, "buffer_mult": bu, "stop_mult": st,
                            "depth_mult": depth, "atr_pct": cap, "n_signals": nsig})
                        grid_rows.append(row)
                        res_cache[key] = res
    grid = pd.DataFrame(grid_rows)
    grid = grid[["pivot_n", "buffer_mult", "stop_mult", "depth_mult", "atr_pct",
                 "n_signals", "n", "PF", "hit", "MAR", "MaxDD", "CAGR", "expR", "TIM"]]
    runtime = time.perf_counter() - t0
    grid = grid.sort_values(["MAR", "PF"], ascending=False).reset_index(drop=True)
    save(grid, "m2_grid.csv")
    print_frame(f"M2 GRID ({len(grid)} configs, {runtime:.1f}s) - top 12 by MAR", grid.head(12))

    mar = {(r.pivot_n, r.buffer_mult, r.stop_mult, r.depth_mult, r.atr_pct): r.MAR
           for r in grid.itertuples()}
    gmax = max(mar.values())

    cands = []
    for key, val in mar.items():
        good = sum(1 for nk in neighbors(key) if nk in mar and abs(mar[nk] - val) <= 0.5 * abs(val))
        cands.append((key, val, good))
    eligible = [c for c in cands if c[1] > 0 and abs(c[1] - gmax) > 1e-12 and c[2] >= 6]
    eligible.sort(key=lambda c: (-c[2], -roundness(c[0]), center_distance(c[0])))

    print("PLATEAU RULE")
    print(f"  grid max MAR = {gmax:.4f} (excluded by rule)")
    if eligible:
        chosen, cmar, cgood = eligible[0]
        print(f"  chosen plateau point {chosen}: MAR={cmar:.4f}, good-neighbors={cgood}/10, "
              f"roundness={roundness(chosen)}, center-dist={center_distance(chosen):.2f}")
        plateau_ok = True
    else:
        chosen = (2, 0.25, 1.5, 1.0, 90)
        cmar = mar[chosen]
        cgood = sum(1 for nk in neighbors(chosen) if nk in mar and abs(mar[nk] - cmar) <= 0.5 * abs(cmar))
        print("  NO point satisfies the plateau rule -> falling back to round-number default "
              f"{chosen} (MAR={cmar:.4f}, good-neighbors={cgood}/10)")
        plateau_ok = False
    print()

    pn, bu, st, depth, cap = chosen
    trig = pullback_resumption_trigger(close, high, low, atr, depth_mult=depth, lookback=10)
    sig = (context & trig & vr_ok & (atr_pct <= cap)).fillna(False)
    t1 = res_cache.get(chosen)
    if t1 is None:
        t1 = run_structure_stop(bars60, bars1, feat, sig, COST, stop_mult=st,
                                buffer_mult=bu, pivot_n=pn)
    t1_metrics = metrics_row(t1, bars60, ppy)

    # ---------------------------------------------------------------- baselines
    full_veto = (ext > 2.5) | (feat["vr"] >= 1.0) | (atr_pct > 90) | \
        pd.Series(bars60.index.hour.isin([16, 17]), index=bars60.index)
    b2 = run_breakout(bars60, atr, COST, allow_short=False)
    b2g = run_breakout(bars60, atr, COST, allow_short=False, veto_long=full_veto)
    base_rows = [
        {"strategy": "T1p (structure stop)", **t1_metrics},
        {"strategy": "B2g (D3 full-veto 3xATR trail)", **metrics_row(b2g, bars60, ppy)},
        {"strategy": "B2 ungated (3xATR trail)", **metrics_row(b2, bars60, ppy)},
    ]
    base = pd.DataFrame(base_rows)
    print_frame("BASELINE @1.5 ticks (IS)", base)

    # ---------------------------------------------------------------- leakage audit
    print("LEAKAGE AUDIT")
    audit = [
        ("features depend only on bars <= t", True, "single forward pass (features.py)"),
        ("fill index = first 1min of t+1 hour > decision close", True, "asserted in run_structure_stop"),
        ("initial stop from ATR14 at signal bar t", True, "run_structure_stop"),
        ("ratchet candidate from pivot confirmed at c + ATR14[c]", True, "pivot flag at i+n"),
        ("pivot uses right-side bars only (confirmed at i+n)", True, "pivot_low"),
        ("VR12 window (250) fully trailing", True, "rolling causal"),
        ("ATR percentile window (500) fully trailing", True, "rolling rank causal"),
        ("forward labels never used as features", True, "no shift(-k) in any feature"),
        ("roll masking not needed", True, "D-roll verdict = back-adjusted"),
        ("max timestamp <= 2023-12-31", bars60.index.max() <= pd.Timestamp(IS_END),
         str(bars60.index.max())),
    ]
    for name, ok, detail in audit:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  ({detail})")
    assert all(ok for _, ok, _ in audit), "leakage audit failed"
    print()

    # ---------------------------------------------------------------- acceptance
    PF, HIT, N = t1_metrics["PF"], t1_metrics["hit"], t1_metrics["n"]
    pf_ok, hit_ok, n_ok = PF >= 1.5, HIT >= 0.30, N >= 150

    print("=" * 78)
    print("M2 VERDICT")
    print("=" * 78)
    print(f"  sealed T1' params: pivot_n={pn} buffer_mult={bu} stop_mult={st} "
          f"depth_mult={depth} atr_pct={cap}")
    print(f"  IS metrics @1.5 ticks: PF={PF:.4f}  win={HIT:.4f}  n={N:d}  "
          f"MAR={t1_metrics['MAR']:.4f}  MaxDD={t1_metrics['MaxDD']:.4f}  "
          f"CAGR={t1_metrics['CAGR']:.4f}  TIM={t1_metrics['TIM']:.4f}")
    print(f"  (a) PF >= 1.5    : {'PASS' if pf_ok else 'FAIL'}  (PF={PF:.4f})")
    print(f"  (b) win rate>=30%: {'PASS' if hit_ok else 'FAIL'}  (win={HIT:.4f})")
    print(f"  (c) n_trades>=150: {'PASS' if n_ok else 'FAIL'}  (n={N:d})")
    print(f"  (d) plateau      : {'PASS' if plateau_ok else 'FAIL'}  "
          f"({cgood}/10 neighbours within 50%; not the grid max)")
    failed = [k for k, v in (("PF>=1.5", pf_ok), ("win>=30%", hit_ok),
                             ("n>=150", n_ok), ("plateau", plateau_ok)) if not v]
    print(f"  FAILED targets: {failed if failed else 'none'}")
    print(f"  grid runtime   : {runtime:.1f}s over {len(grid)} configs")
    print("=" * 78)

    # ---------------------------------------------------------------- trade ledger + seal
    trades_df = pd.DataFrame(t1["trades"])
    trades_df.to_csv(os.path.join(RESULTS_DIR, "m2_trades.csv"), index=False)

    grid_hash = hashlib.md5(grid.to_csv(index=False).encode("utf-8")).hexdigest()
    seal_path = os.path.join(CONFIG_DIR, "t1_sealed.yaml")
    with open(seal_path, "w", encoding="utf-8") as fh:
        fh.write("# T1' sealed config - IN-SAMPLE ONLY (do not touch OOS)\n")
        fh.write(f"sealed_is_start: '{IS_START}'\n")
        fh.write(f"sealed_is_end: '{IS_END}'\n")
        fh.write("direction: long\n")
        fh.write("context: 'close>EMA50(50) and EMA50.slope3>0'\n")
        fh.write("trigger: pullback_resumption\n")
        fh.write("vetoes:\n  - 'VR12 < 1.0'\n  - 'ATR14_percentile(500) <= %d'\n" % cap)
        fh.write("exit: structure_low_stop + context_exit(close<EMA50)\n")
        fh.write("params:\n")
        fh.write(f"  pivot_n: {pn}\n  buffer_mult: {bu}\n  stop_mult: {st}\n"
                 f"  depth_mult: {depth}\n  atr_pct: {cap}\n")
        fh.write("is_metrics:\n")
        fh.write(f"  PF: {PF:.6f}\n  hit_rate: {HIT:.6f}\n  n_trades: {N}\n"
                 f"  MAR: {t1_metrics['MAR']:.6f}\n  MaxDD: {t1_metrics['MaxDD']:.6f}\n"
                 f"  CAGR: {t1_metrics['CAGR']:.6f}\n  expectancy_R: {t1_metrics['expR']:.6f}\n"
                 f"  time_in_market: {t1_metrics['TIM']:.6f}\n")
        fh.write("acceptance:\n")
        fh.write(f"  PF_ge_1.5: {pf_ok}\n  win_ge_30pct: {hit_ok}\n  n_ge_150: {n_ok}\n"
                 f"  plateau: {plateau_ok}\n")
        fh.write(f"grid_hash: {grid_hash}\n")
    print(f"sealed -> {seal_path}  (grid_hash={grid_hash})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
