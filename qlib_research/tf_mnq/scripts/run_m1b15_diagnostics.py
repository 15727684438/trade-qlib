"""M1b 15min diagnostic suite (IN-SAMPLE ONLY, <= 2023-12-31).

Parameterises the existing 60min M1b suite for the 15min decision timeframe:
  D0 default side (conditional drift, annualised on the 15min calendar)
  D1 feature deciles (ER20, ADX14, VR12, ATR14-pct, |EMA50 slope|/ATR,
     extension=(close-EMA20)/ATR) vs canonical_R at H=12 (~3h) and H=24 (~6h)
  D2 trigger families (Donchian-20 breakout vs pullback-resumption) + depth
  D3 B2 (Donchian-20 + 3xATR trail) single vetoes + full stack @1.5 ticks
  D4 hour-of-day + RTH vs Globex entry windows
  D5 ATR-pct deciles x expectancy
  D6 persistence (acf, P(stay ON|ON), durations)
  D7 forward path-efficiency decomposition by gate state

Everything is computed on the 15min IS slice; forward labels never cross the
freeze. Saves results/m1b15_*.csv and a DECISIONS_15MIN block.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(_PKG_ROOT))

from tf_mnq import IS_END, IS_START  # noqa: E402
from tf_mnq.data import load_bars  # noqa: E402
from tf_mnq.engine import CostCfg, metrics, run_target_position  # noqa: E402
from tf_mnq.features import compute_features  # noqa: E402
from tf_mnq.strategies import strategy_frame  # noqa: E402

# reuse the D1/D3 primitives from the 60min suite (no duplication)
_spec = importlib.util.spec_from_file_location(
    "run_m1b_diagnostics", os.path.join(_SCRIPT_DIR, "run_m1b_diagnostics.py")
)
_m1b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m1b)
canonical_R = _m1b.canonical_R
stat_block = _m1b.stat_block
weekly_bootstrap_ci = _m1b.weekly_bootstrap_ci
classify_deciles = _m1b.classify_deciles
run_breakout = _m1b.run_breakout

RESULTS_DIR = os.path.join(_PKG_ROOT, "results")
FREQ = "15min"
H_LIST = (12, 24)          # ~3h and ~6h in 15min bars
H_LABEL = 24
COST = CostCfg(slip_ticks=0.4, commission_per_side=0.5, tick=0.25, point_value=2.0)
TOD_HOURS = (16, 17)       # ET 16:00-18:00 veto


def save(df: pd.DataFrame, name: str) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, name), index=False)


def print_frame(title: str, df: pd.DataFrame, floatfmt: str = "%.4f") -> None:
    print(title)
    if df.empty:
        print("  (empty)")
    else:
        with pd.option_context("display.width", 220, "display.max_columns", 60):
            print(df.to_string(index=False, float_format=lambda x: floatfmt % x))
    print()


def rth_mask(index: pd.DatetimeIndex) -> pd.Series:
    """Regular Trading Hours (US equity 09:30-16:00 ET) vs Globex."""
    mins = index.hour * 60 + index.minute
    return pd.Series((mins >= 570) & (mins < 960), index=index)


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    bars = load_bars(FREQ, start=IS_START, end=IS_END)
    feat = compute_features(bars)
    sf = strategy_frame(bars, feat)
    close, high, low = bars["close"], bars["high"], bars["low"]
    atr = feat["atr"]
    ema50 = feat["ema50"]
    slope3 = ema50.diff(3)
    dirn = np.sign(slope3)
    years = (bars.index[-1] - bars.index[0]).days / 365.25
    ppy = len(bars) / years
    print(f"IS {FREQ} bars={len(bars)}  [{bars.index[0]} .. {bars.index[-1]}]  "
          f"ppy(15min calendar)={ppy:.1f}\n")
    assert bars.index.max() <= pd.Timestamp(IS_END)

    decisions: dict = {"freq": FREQ, "ppy": round(ppy, 3)}

    # ------------------------------------------------------------------ D0
    d0_rows = []
    for h in H_LIST:
        fwd = (close.shift(-h) / close - 1.0)
        for label, mask in (("LONG (dir>0)", dirn > 0), ("SHORT (dir<0)", dirn < 0)):
            m = fwd[mask].dropna()
            ann = float(m.mean() / h * ppy)
            side_contrib = ann if label.startswith("LONG") else -ann
            d0_rows.append({"H": h, "side": label, "n": int(m.size),
                            "mean_fwd_ret": float(m.mean()),
                            "annualized_drift": ann, "side_contribution": side_contrib})
    d0 = pd.DataFrame(d0_rows)
    print_frame(f"D0  conditional drift annualized (mean/H * ppy={ppy:.0f})", d0, "%.6f")
    short_contrib = float(d0[(d0["H"] == H_LABEL) & (d0["side"].str.startswith("SHORT"))]["side_contribution"].iloc[0])
    decisions["D0"] = ("long-only (short drift negative on IS)" if short_contrib <= 0
                       else "long+short allowed (short drift positive)")
    print(f"DECISION D0: {decisions['D0']}  (short side_contrib@H24={short_contrib:+.6f})\n")
    save(d0, "m1b15_d0_drift.csv")

    # ------------------------------------------------------------------ D1
    feats = {
        "ER20": feat["er"],
        "ADX14": feat["adx"],
        "VR12": feat["vr"],
        "ATR14_pct": sf["atr_pct"],
        "slopeAbsATR": sf["slope3"].abs() / atr,
        "extensionATR": sf["ext"],
    }
    d1_rows = []
    d1_by_h: dict = {h: {} for h in H_LIST}
    for h in H_LIST:
        R_dir = canonical_R(bars, dirn, atr, COST, horizon=h)
        for fname, fs in feats.items():
            data = pd.DataFrame({"f": fs, "R": R_dir}).dropna()
            if len(data) < 200:
                d1_by_h[h][fname] = "drop"
                continue
            data["decile"] = pd.qcut(data["f"], 10, labels=False, duplicates="drop")
            means = []
            for d, sub in data.groupby("decile"):
                st = stat_block(sub["R"])
                lo, hi = weekly_bootstrap_ci(sub["R"])
                means.append(st["mean_R"])
                d1_rows.append({"H": h, "feature": fname, "decile": int(d),
                                "n": st["n"], "mean_R": st["mean_R"],
                                "median_R": st["median_R"], "PF": st["PF"],
                                "hit": st["hit"], "ci_lo": lo, "ci_hi": hi})
            d1_by_h[h][fname] = classify_deciles(np.array(means))
    d1 = pd.DataFrame(d1_rows)
    print_frame("D1  feature deciles vs canonical_R (H=12 ~3h / H=24 ~6h)", d1, "%.4f")
    d1_dec = {}
    for fname in feats:
        c12, c24 = d1_by_h[12][fname], d1_by_h[H_LABEL][fname]
        if "band" in (c12, c24):
            d1_dec[fname] = "band"
        elif c24 == "threshold" or (c12 == "threshold" and c24 != "drop"):
            d1_dec[fname] = "threshold"
        else:
            d1_dec[fname] = "drop"
        print(f"  D1 {fname:>12}: H12={c12:<9} H24={c24:<9} -> {d1_dec[fname]}")
    decisions["D1"] = d1_dec
    print("DECISION D1: thresholds from monotone features; bands for hump-shaped; drop flat/negative\n")
    save(d1, "m1b15_d1_feature_deciles.csv")

    # ------------------------------------------------------------------ D2
    context_long = sf["context_long"]
    don_hi20 = high.rolling(20, min_periods=20).max().shift(1)
    ev_a = context_long & (close > don_hi20)
    peak10 = high.rolling(10, min_periods=10).max()
    depth = (peak10 - low) / atr
    prior_pb = depth.rolling(5, min_periods=1).max()
    ev_b = context_long & (prior_pb >= 1.0) & (close > high.shift(1))
    d2_rows = []
    for name, ev in (("a_donchian20_breakout", ev_a), ("b_pullback_resumption", ev_b)):
        R = canonical_R(bars, (ev & context_long).astype(int), atr, COST, horizon=H_LABEL)
        st = stat_block(R)
        st["exp_x_sqrtn"] = st["mean_R"] * np.sqrt(st["n"]) if st["n"] else np.nan
        d2_rows.append({"trigger": name, **st})
    d2 = pd.DataFrame(d2_rows)
    evb_bars = bars[ev_b & context_long]
    d2_depth = pd.DataFrame()
    depth_choice = 1.0
    if len(evb_bars):
        db = prior_pb.reindex(evb_bars.index).dropna()
        Rb = canonical_R(bars, (ev_b & context_long).astype(int), atr, COST,
                         horizon=H_LABEL).reindex(db.index)
        bins = pd.cut(db, [-np.inf, 1, 2, 3, np.inf], labels=["<1", "1-2", "2-3", ">3"])
        rows = []
        for lab, sub in Rb.groupby(bins, observed=False):
            st = stat_block(sub)
            rows.append({"depth_ATR": str(lab), **st})
            if st["n"] >= 20:
                rows[-1]["_lab"] = str(lab)
        d2_depth = pd.DataFrame(rows)
        edge = {"<1": 0.0, "1-2": 1.0, "2-3": 2.0, ">3": 3.0}
        # shallowest bucket with positive expectancy (robust, avoids overfitting
        # the small-n deep bucket); goes a little further than the grid range.
        pos = [r for r in rows if r["n"] >= 20 and np.isfinite(r["mean_R"]) and r["mean_R"] > 0]
        depth_choice = min(edge[r["depth_ATR"]] for r in pos) if pos else 1.0
    print_frame("D2  trigger families (long context, R @H=24)", d2, "%.4f")
    print_frame("D2b pullback-resumption by depth bucket", d2_depth.drop(columns=["_lab"], errors="ignore"), "%.4f")
    primary = d2.sort_values("exp_x_sqrtn", ascending=False)["trigger"].iloc[0]
    decisions["D2_primary"] = primary
    decisions["D2_depth_mult"] = depth_choice
    print(f"DECISION D2: primary trigger = {primary}; pullback depth threshold = {depth_choice} (bucket best mean_R)\n")
    save(d2, "m1b15_d2_triggers.csv")
    save(d2_depth.drop(columns=["_lab"], errors="ignore"), "m1b15_d2_depth.csv")

    # ------------------------------------------------------------------ D3
    ext_veto_long = sf["ext"] > 2.5
    vr_veto = feat["vr"] >= 1.0
    vol_veto = sf["atr_pct"] > 90
    tod_veto = pd.Series(bars.index.hour.isin(list(TOD_HOURS)), index=bars.index)
    old_gate_veto = feat["gate"] != 1

    def veto_l(s):
        return s.reindex(bars.index).fillna(True)

    veto_defs = {
        "extension>2.5ATR": veto_l(ext_veto_long),
        "VR>=1.0": veto_l(vr_veto),
        "ATRpct>90": veto_l(vol_veto),
        "hour16-18ET": veto_l(tod_veto),
        "old_gate!=1": veto_l(old_gate_veto),
    }
    full = veto_l(ext_veto_long | vr_veto | vol_veto | tod_veto)

    rows3 = []

    def add_row(name, res, ungated_n=None):
        m = metrics(res["equity"], res["trades"], bars.index, COST, ppy)
        nratio = m["n_trades"] / ungated_n if ungated_n else 1.0
        rows3.append({"variant": name, "n": m["n_trades"], "n_ratio": nratio, "MAR": m["MAR"],
                      "PF": m["profit_factor"], "MaxDD": m["MaxDD"], "hit": m["hit_rate"],
                      "CAGR": m["CAGR"], "expR": m["expectancy_R"]})
        return m

    b2_long = run_breakout(bars, atr, COST, allow_short=False)
    m_base = add_row("B2_long (ungated)", b2_long)
    base_n = m_base["n_trades"]
    for vname, vser in veto_defs.items():
        res = run_breakout(bars, atr, COST, allow_short=False, veto_long=vser)
        add_row(f"B2+{vname}", res, ungated_n=base_n)
    b2_full = run_breakout(bars, atr, COST, allow_short=False, veto_long=full)
    add_row("B2g=FULL stack", b2_full, ungated_n=base_n)
    d3 = pd.DataFrame(rows3)
    print_frame("D3  B2 Donchian-20 + 3xATR trailing stop, veto deltas @1.5 ticks", d3, "%.4f")

    base_m = rows3[0]
    single = []
    for r in rows3[1:5]:  # ext, vr, vol, tod (old_gate excluded from kept logic)
        delta_pf = r["PF"] - base_m["PF"]
        rel_mar = (r["MAR"] - base_m["MAR"]) / abs(base_m["MAR"]) if base_m["MAR"] else np.nan
        keep = r["n_ratio"] >= 0.60 and (delta_pf >= 0.05 or rel_mar >= 0.15)
        single.append((r["variant"], delta_pf, rel_mar, r["n_ratio"], keep))
    best_single = max([s for s in single if s[4]] or single, key=lambda x: x[2])
    full_row = [r for r in rows3 if r["variant"] == "B2g=FULL stack"][0]
    full_beats = full_row["MAR"] >= base_m["MAR"] and (
        full_row["PF"] >= best_single[1] + base_m["PF"]
        or full_row["MAR"] >= best_single[2] * abs(base_m["MAR"]) + base_m["MAR"]
    )
    decisions["D3_keep"] = [s[0] for s in single if s[4]] or ["none"]
    decisions["D3_stack"] = ("FULL stack" if full_beats else f"best single veto only ({best_single[0]})")
    inc_pf = full_row["PF"] - base_m["PF"]
    inc_mar = full_row["MAR"] - base_m["MAR"]
    gate_pos = bool(inc_pf > 0 or inc_mar > 0)
    decisions["D3_gate_increment_positive"] = gate_pos
    decisions["D3_gate_increment"] = {"dPF": round(inc_pf, 4), "dMAR": round(inc_mar, 4)}
    print("DECISION D3: kept per rule ->", decisions["D3_keep"])
    print(f"  FULL stack beats best single? {full_beats} -> {decisions['D3_stack']}")
    print(f"  GATE: full-veto-stack (a.k.a. gate-as-veto) increment B2->B2g: "
          f"dPF={inc_pf:+.4f} dMAR={inc_mar:+.4f} positive={gate_pos}\n")
    save(d3, "m1b15_d3_vetoes.csv")

    # ------------------------------------------------------------------ D4
    fwd12 = close.shift(-12) / close - 1.0
    hours = bars.index.hour
    d4_agg = pd.DataFrame({"hour": hours, "fwd_ret": fwd12}).dropna().groupby("hour").agg(
        n=("fwd_ret", "size"), mean_fwd_ret=("fwd_ret", "mean"), vol_fwd_ret=("fwd_ret", "std"))
    target_b1 = ((feat["gate"] == 1) & (feat["dir"] > 0)).astype(int)
    b1_res = run_target_position(bars, target_b1, COST, allow_short=False, atr=atr)
    tr = pd.DataFrame([{"hour": t["entry_time"].hour, "pnl": t["pnl_usd_net"]} for t in b1_res["trades"]])
    if len(tr):
        hb = tr.groupby("hour").agg(b1_n=("pnl", "size"), b1_mean_pnl=("pnl", "mean"))
        hb["b1_entry_share"] = hb["b1_n"] / hb["b1_n"].sum()
    else:
        hb = pd.DataFrame()
    d4 = d4_agg.join(hb, how="left").reset_index()
    rth = rth_mask(bars.index)
    sess = pd.DataFrame({"session": np.where(rth, "RTH", "Globex"),
                         "fwd_ret": fwd12, "b1_pnl": np.nan})
    b1_by_time = pd.Series(np.nan, index=bars.index)
    for t in b1_res["trades"]:
        b1_by_time.loc[t["entry_time"]] = t["pnl_usd_net"]
    sess["b1_pnl"] = b1_by_time
    d4s = sess.dropna(subset=["fwd_ret"]).groupby("session").agg(
        n=("fwd_ret", "size"), mean_fwd_ret=("fwd_ret", "mean"), vol_fwd_ret=("fwd_ret", "std"))
    b1_sess = sess.dropna(subset=["b1_pnl"]).groupby("session").agg(
        b1_n=("b1_pnl", "size"), b1_mean_pnl=("b1_pnl", "mean"))
    d4s = d4s.join(b1_sess, how="left").reset_index()
    print_frame("D4  forward-12 return / vol / B1 realized PnL by ET hour", d4, "%.6f")
    print_frame("D4b RTH (09:30-16:00 ET) vs Globex", d4s, "%.6f")
    bad = d4[(d4["mean_fwd_ret"] < 0) & (d4["b1_entry_share"].fillna(0) >= 0.05)]
    if len(bad):
        hrs = sorted(int(x) for x in bad["hour"])
        decisions["D4"] = f"restrict entries away from ET hours {hrs} (neg fwd drift & >=5% of B1 trades)"
    else:
        decisions["D4"] = "no time-of-day entry restriction justified by fwd drift"
    print(f"DECISION D4: {decisions['D4']}\n")
    save(d4, "m1b15_d4_hourly.csv")
    save(d4s, "m1b15_d4_session.csv")

    # ------------------------------------------------------------------ D5
    ev_primary = ev_a if primary.startswith("a") else ev_b
    prim_dir = (ev_primary & context_long).astype(int)
    Rprim = canonical_R(bars, prim_dir, atr, COST, horizon=H_LABEL)
    data5 = pd.DataFrame({"atr_pct": sf["atr_pct"], "R": Rprim}).dropna()
    rows5 = []
    if len(data5) > 100:
        data5["decile"] = pd.qcut(data5["atr_pct"], 10, labels=False, duplicates="drop")
        for d, sub in data5.groupby("decile"):
            rows5.append({"decile": int(d), **stat_block(sub["R"])})
    d5 = pd.DataFrame(rows5)
    print_frame("D5  ATR14-percentile deciles x expectancy (primary trigger, H=24)", d5, "%.4f")
    if len(d5) >= 3:
        peak_dec = int(d5.loc[d5["mean_R"].idxmax(), "decile"])
        peak = float(d5["mean_R"].max())
        top = float(d5["mean_R"].iloc[-1])
        bot = float(d5["mean_R"].iloc[0])
        if top < peak - 1e-9 and peak_dec < d5["decile"].max():
            decisions["D5"] = (f"vol band: top ATR decile underperforms peak ({top:.3f} vs {peak:.3f}) -> cap ATR-pct")
        elif top >= bot:
            decisions["D5"] = "ATR monotone -> no vol band"
        else:
            decisions["D5"] = "ATR flat/negative -> drop vol from veto set"
    else:
        decisions["D5"] = "insufficient trigger sample"
    print(f"DECISION D5: {decisions['D5']}\n")
    save(d5, "m1b15_d5_volband.csv")

    # ------------------------------------------------------------------ D6
    rows6 = []
    thr = {"ER20": 0.30, "ADX14": 20.0, "VR12": 1.05}
    for fname, fs in (("ER20", feat["er"]), ("ADX14", feat["adx"]), ("VR12", feat["vr"])):
        row = {"feature": fname, "threshold": thr[fname]}
        for lag in (1, 4, 12, 24):
            row[f"acf{lag}"] = float(fs.autocorr(lag))
        above = (fs >= thr[fname])
        both = above & above.shift(-1)
        row["P_stayON"] = float(both.sum() / above.sum()) if above.sum() else np.nan
        arr = above.to_numpy()
        runs_on, runs_off = [], []
        cur = arr[0]
        length = 1
        for v in arr[1:]:
            if v == cur:
                length += 1
            else:
                (runs_on if cur else runs_off).append(length)
                cur = v
                length = 1
        (runs_on if cur else runs_off).append(length)
        row["med_dur_ON"] = float(np.median(runs_on)) if runs_on else np.nan
        row["med_dur_OFF"] = float(np.median(runs_off)) if runs_off else np.nan
        rows6.append(row)
    d6 = pd.DataFrame(rows6)
    print_frame("D6  persistence of regime features (15min)", d6, "%.4f")
    low_pers = [r["feature"] for _, r in d6.iterrows() if r["acf12"] < 0.5 or r["med_dur_ON"] < 4]
    decisions["D6"] = f"entry-time filters only (never state machine): {low_pers or ['none']}"
    print(f"DECISION D6: {decisions['D6']}\n")
    save(d6, "m1b15_d6_persistence.csv")

    # ------------------------------------------------------------------ D7
    fwdvol = close.pct_change().rolling(H_LABEL, min_periods=H_LABEL).std().shift(-H_LABEL)
    fwdnet = (close.shift(-H_LABEL) - close).abs()
    patheff = fwdnet / close.diff().abs().rolling(H_LABEL, min_periods=H_LABEL).sum().shift(-H_LABEL)
    d7rows = []
    for g in (0, 1):
        m = feat["gate"] == g
        d7rows.append({"gate": g, "n": int((m & patheff.notna()).sum()),
                       "mean_fwd_vol": float(fwdvol[m].dropna().mean()),
                       "mean_abs_drift": float(fwdnet[m].dropna().mean()),
                       "mean_path_eff": float(patheff[m].dropna().mean())})
    d7 = pd.DataFrame(d7rows)
    print_frame("D7  forward path-efficiency decomposition by gate @H=24", d7, "%.6f")
    eff_inv = d7.loc[1, "mean_path_eff"] < d7.loc[0, "mean_path_eff"]
    drift_inv = d7.loc[1, "mean_abs_drift"] > d7.loc[0, "mean_abs_drift"]
    vol_inv = d7.loc[1, "mean_fwd_vol"] > d7.loc[0, "mean_fwd_vol"]
    if eff_inv and drift_inv and vol_inv:
        decisions["D7"] = "CONFIRMED vol artifact: gate-ON higher fwd vol + |drift| but lower path-efficiency ratio"
    elif eff_inv:
        decisions["D7"] = "REFUTED/partial: inversion not explained by vol denominator"
    else:
        decisions["D7"] = "no inversion observed"
    print(f"DECISION D7: {decisions['D7']}\n")
    save(d7, "m1b15_d7_efficiency_decomp.csv")

    # ------------------------------------------------------------------ FINAL
    b2_metrics = metrics(b2_long["equity"], b2_long["trades"], bars.index, COST, ppy)
    b2g_metrics = metrics(b2_full["equity"], b2_full["trades"], bars.index, COST, ppy)
    print_frame("B2 vs B2g @1.5 ticks (15min, IS)", pd.DataFrame([
        {"strategy": "B2 (ungated)", "n": b2_metrics["n_trades"], "MAR": b2_metrics["MAR"],
         "PF": b2_metrics["profit_factor"], "MaxDD": b2_metrics["MaxDD"], "hit": b2_metrics["hit_rate"],
         "CAGR": b2_metrics["CAGR"], "expR": b2_metrics["expectancy_R"]},
        {"strategy": "B2g (full veto stack)", "n": b2g_metrics["n_trades"], "MAR": b2g_metrics["MAR"],
         "PF": b2g_metrics["profit_factor"], "MaxDD": b2g_metrics["MaxDD"], "hit": b2g_metrics["hit_rate"],
         "CAGR": b2g_metrics["CAGR"], "expR": b2g_metrics["expectancy_R"]},
    ]), "%.4f")

    print("=" * 78)
    print("DECISIONS_15MIN")
    print("=" * 78)
    print(f"  ppy (15min calendar)  : {ppy:.1f}")
    print(f"  D0 default side       : {decisions['D0']}")
    print(f"  D1 feature roles      : {decisions['D1']}")
    print(f"  D2 primary trigger    : {decisions['D2_primary']}")
    print(f"  D2 depth threshold    : {decisions['D2_depth_mult']}")
    print(f"  D3 vetoes kept        : {decisions['D3_keep']}")
    print(f"  D3 stack choice       : {decisions['D3_stack']}")
    print(f"  D3 full-stack +?      : {decisions['D3_gate_increment_positive']}  ({decisions['D3_gate_increment']})")
    print(f"  D4 time-of-day        : {decisions['D4']}")
    print(f"  D5 vol band           : {decisions['D5']}")
    print(f"  D6 persistence rule   : {decisions['D6']}")
    print(f"  D7 M1 explanation     : {decisions['D7']}")
    print(f"  GATE (go/no-go)       : {'GO' if gate_pos else 'NO-GO'}")
    print("=" * 78)

    # persist decisions for Phase 2 (manual YAML; strings kept single-line)
    yml = os.path.join(RESULTS_DIR, "m1b15_decisions.yaml")
    with open(yml, "w", encoding="utf-8") as fh:
        fh.write("# DECISIONS_15MIN - generated by run_m1b15_diagnostics.py (IS only)\n")
        fh.write("# Cost model corrected 2026-10-10: base ('1.5t' label retained) = commission $0.50/side + slippage $0.20/side ($1.40 round trip); stress ('3.0t') = $0.50/side + $1.50/side slippage.\n")
        fh.write(f"freq: {FREQ}\n")
        fh.write(f"ppy: {ppy:.6f}\n")
        fh.write(f"gate_pass: {gate_pos}\n")
        fh.write(f'd0_side: "{decisions["D0"]}"\n')
        fh.write(f'd1_roles: {dict(decisions["D1"])}\n')
        fh.write(f'd2_primary: {decisions["D2_primary"]}\n')
        fh.write(f'd2_depth_mult: {decisions["D2_depth_mult"]}\n')
        fh.write(f'd3_keep: {decisions["D3_keep"]}\n')
        fh.write(f'd3_stack: "{decisions["D3_stack"]}"\n')
        fh.write(f'd3_gate_increment: {decisions["D3_gate_increment"]}\n')
        fh.write(f'd4: "{decisions["D4"]}"\n')
        fh.write(f'd5: "{decisions["D5"]}"\n')
        fh.write(f'd6: "{decisions["D6"]}"\n')
        fh.write(f'd7: "{decisions["D7"]}"\n')
    print(f"decisions -> {yml}")
    return 0 if gate_pos else 2


if __name__ == "__main__":
    raise SystemExit(main())
