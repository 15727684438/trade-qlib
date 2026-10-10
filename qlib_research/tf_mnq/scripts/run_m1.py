"""Milestone M1 runner.

IN-SAMPLE ONLY: every feature, label and metric is restricted to
2019-05-06 .. 2023-12-31. OOS (2024+) is never loaded or evaluated.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tf_mnq import IS_END, IS_START  # noqa: E402
from tf_mnq.data import load_bars  # noqa: E402
from tf_mnq.engine import CostCfg, buy_and_hold, metrics, run_target_position  # noqa: E402
from tf_mnq.features import compute_features  # noqa: E402
from tf_mnq import rolls  # noqa: E402

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
SLIPS = (0.4, 1.0, 3.0)


def gate_separation(bars60: pd.DataFrame, feat: pd.DataFrame, horizons=(12, 24)) -> pd.DataFrame:
    close = bars60["close"]
    rows = []
    for h in horizons:
        fwd_ret = close.shift(-h) / close - 1.0
        fwd_eff = (close.shift(-h) - close).abs() / close.diff().abs().rolling(
            h, min_periods=h
        ).sum().shift(-h)
        tmp = pd.DataFrame({"gate": feat["gate"], "fwd_ret": fwd_ret, "fwd_eff": fwd_eff}).dropna()
        grp = tmp.groupby("gate").agg(
            count=("fwd_eff", "size"),
            mean_fwd_ret=("fwd_ret", "mean"),
            mean_fwd_path_eff=("fwd_eff", "mean"),
        )
        for gate, r in grp.iterrows():
            rows.append(
                {
                    "H": h,
                    "gate": int(gate),
                    "count": int(r["count"]),
                    "mean_fwd_ret": float(r["mean_fwd_ret"]),
                    "mean_fwd_path_eff": float(r["mean_fwd_path_eff"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["H", "gate"]).reset_index(drop=True)


def print_gate_table(gs: pd.DataFrame) -> None:
    print("GATE SEPARATION (forward labels, IS only)")
    print(f"  {'H':>3}  {'gate':>4}  {'count':>7}  {'mean_fwd_ret':>13}  {'mean_fwd_eff':>13}")
    for _, r in gs.iterrows():
        print(
            f"  {int(r['H']):>3}  {int(r['gate']):>4}  {int(r['count']):>7}  "
            f"{r['mean_fwd_ret']:>13.6f}  {r['mean_fwd_path_eff']:>13.6f}"
        )
    print()


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    bars60 = load_bars("60min", start=IS_START, end=IS_END)
    day = load_bars("day", start=IS_START, end=IS_END)
    print(f"IS 60min bars: {len(bars60)}  [{bars60.index[0].date()} .. {bars60.index[-1].date()}]")
    print(f"IS day bars  : {len(day)}  [{day.index[0].date()} .. {day.index[-1].date()}]")
    assert bars60.index.max() <= pd.Timestamp(IS_END)

    feat = compute_features(bars60)
    atr = feat["atr"]

    diag = rolls.report(day)

    gs = gate_separation(bars60, feat)
    print_gate_table(gs)
    gs.to_csv(os.path.join(RESULTS_DIR, "m1_gate_separation.csv"), index=False)

    years = (bars60.index[-1] - bars60.index[0]).days / 365.25
    ppy = len(bars60) / years
    print(f"periods_per_year (60min) = {ppy:.1f}  over {years:.2f} calendar years\n")

    # B1 target: long while the gate is ON and direction is up, else flat (causal)
    target_b1 = ((feat["gate"] == 1) & (feat["dir"] > 0)).astype(int)

    metric_rows = []
    results = {}
    for slip in SLIPS:
        cost = CostCfg(slip_ticks=slip, commission_per_side=0.5, tick=0.25, point_value=2.0)
        for name, res in (
            ("B1", run_target_position(bars60, target_b1, cost, allow_short=False, atr=atr)),
            ("B0", buy_and_hold(bars60, cost, atr=atr)),
        ):
            m = metrics(res["equity"], res["trades"], bars60.index, cost, ppy)
            results[(name, slip)] = (res, m)
            metric_rows.append({"strategy": name, "slip_ticks": slip, **m})

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(os.path.join(RESULTS_DIR, "m1_metrics.csv"), index=False)

    # ---- sanity: B0 total return ~= last/first - 1 net of one round trip ---- #
    first, last = float(bars60["close"].iloc[0]), float(bars60["close"].iloc[-1])
    c0 = CostCfg(slip_ticks=0.4, commission_per_side=0.5, tick=0.25, point_value=2.0)
    slip_pts = c0.slip_ticks * c0.tick
    expected = (
        last / first
        - 1.0
        - 2.0 * c0.commission_per_side / (first * c0.point_value)
        - 2.0 * slip_pts / first
    )
    got = results[("B0", 0.4)][1]["total_return"]
    assert abs(got - expected) < 1e-9, f"B0 sanity failed: got {got}, expected {expected}"
    print(f"B0 sanity OK: total_return {got:.6f} == {expected:.6f}\n")

    # ---- sanity: B1 trade ledger reconciles with its equity curve ---- #
    res_b1, _ = results[("B1", 0.4)]
    ledger = sum(t["pnl_usd_net"] for t in res_b1["trades"])
    equity_delta = (float(res_b1["equity"].iloc[-1]) - float(res_b1["equity"].iloc[0])) * res_b1[
        "capital"
    ]
    assert abs(ledger - equity_delta) < 1e-6 * res_b1["capital"], (
        f"B1 ledger/equity mismatch: ledger {ledger}, equity {equity_delta}"
    )
    print(f"B1 ledger sanity OK: trades ${ledger:.2f} == equity ${equity_delta:.2f}\n")

    print("METRICS (IS only)")
    print(
        f"  {'strat':>5}  {'slip':>5}  {'CAGR':>9}  {'MaxDD':>8}  {'MAR':>8}  "
        f"{'PF':>7}  {'hit':>7}  {'avgW$':>8}  {'avgL$':>8}  {'n':>4}  {'TIM':>6}  {'expR':>7}"
    )
    for r in metric_rows:
        print(
            f"  {r['strategy']:>5}  {r['slip_ticks']:>5.1f}  {r['CAGR']:>9.4f}  {r['MaxDD']:>8.4f}  "
            f"{r['MAR']:>8.3f}  {r['profit_factor']:>7.3f}  {r['hit_rate']:>7.3f}  "
            f"{r['avg_win']:>8.2f}  {r['avg_loss']:>8.2f}  {r['n_trades']:>4d}  "
            f"{r['time_in_market']:>6.3f}  {r['expectancy_R']:>7.3f}"
        )
    print()

    # ---- M1 VERDICT ---- #
    def eff(h, gate):
        row = gs[(gs["H"] == h) & (gs["gate"] == gate)]
        return float(row["mean_fwd_path_eff"].iloc[0]) if len(row) else float("nan")

    mono = all(eff(h, 1) > eff(h, 0) for h in (12, 24))
    m15 = results[("B1", 0.4)][1]
    m30 = results[("B1", 3.0)][1]
    pass_base = np.isfinite(m15["MAR"]) and m15["MAR"] > 0
    pass_stress = np.isfinite(m30["MAR"]) and m30["MAR"] > 0
    n_tr = m15["n_trades"]

    print("=" * 78)
    print("M1 VERDICT")
    print("=" * 78)
    print(f"(a) gate separation monotone (gate1 fwd-eff > gate0) for H=12,24 : {mono}")
    print(f"    H=12 eff(g1)-eff(g0) = {eff(12,1) - eff(12,0):+.6f}")
    print(f"    H=24 eff(g1)-eff(g0) = {eff(24,1) - eff(24,0):+.6f}")
    print(f"(b) B1 IS MAR > 0 @1.5t : {pass_base}  (MAR={m15['MAR']:.3f})")
    print(f"    B1 IS MAR > 0 @3.0t : {pass_stress}  (MAR={m30['MAR']:.3f})")
    print(f"(c) B1 IS trade count @1.5t : {n_tr}")
    print("ROLL:", diag["verdict"])
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
