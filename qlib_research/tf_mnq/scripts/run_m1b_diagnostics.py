"""M1b diagnostic suite (IN-SAMPLE ONLY, <= 2023-12-31).

Redesign inputs for the T1' strategy:
  context = close>EMA50 & slope3>0 (long) / mirror (short)
  event   = Donchian-20 breakout or pullback-resumption
  vetoes  = extension, VR, vol band, time-of-day
  exit    = 3xATR trailing stop (+ context exit in the final design)

D0..D7 each print a table and an explicit DECISION; a final DECISIONS block
summarises everything. Nothing after 2023-12-31 is loaded or evaluated; forward
labels are computed on the IS-sliced frame so they never cross the freeze.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tf_mnq import IS_END, IS_START  # noqa: E402
from tf_mnq.data import load_bars  # noqa: E402
from tf_mnq.engine import CostCfg, metrics, run_target_position  # noqa: E402
from tf_mnq.features import compute_features, ema  # noqa: E402

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
BARS_PER_YEAR = 23 * 252  # Globex convention from the brief (5796)
H_LABEL = 24
COST = CostCfg(slip_ticks=0.4, commission_per_side=0.5, tick=0.25, point_value=2.0)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def save(df: pd.DataFrame, name: str) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, name), index=False)


def print_frame(title: str, df: pd.DataFrame, floatfmt: str = "%.4f") -> None:
    print(title)
    if df.empty:
        print("  (empty)")
    else:
        with pd.option_context("display.width", 200, "display.max_columns", 50):
            print(df.to_string(index=False, float_format=lambda x: floatfmt % x))
    print()


def canonical_R(bars: pd.DataFrame, dir_series, atr: pd.Series, cost: CostCfg,
                horizon: int = 48, stop_mult: float = 2.0) -> pd.Series:
    """Directional fixed-horizon / fixed-stop R primitive.

    At bar ``t`` (direction known at close): enter at ``open[t+1]`` with adverse
    slip, stop at ``entry - stop_mult*ATR14[t]*dir``. Walk forward up to
    ``horizon`` bars; first stop touch exits at ``min(stop, open)`` for longs
    (``max`` for shorts, gap-through worse side), else exit at ``close[t+horizon]``.
    ``R = ((exit-entry)*dir - costs) / (stop_mult*ATR14[t])``.
    """
    idx = bars.index
    o = bars["open"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    low = bars["low"].to_numpy(float)
    c = bars["close"].to_numpy(float)
    dirn = dir_series.reindex(idx).fillna(0).to_numpy(float)
    atrn = atr.reindex(idx).to_numpy(float)
    n = len(bars)
    pv = float(cost.point_value)
    slip = float(cost.slip_ticks) * float(cost.tick)
    comm_pts = float(cost.commission_per_side) / pv

    R = np.full(n, np.nan)
    for t in range(n - 1):
        d = dirn[t]
        if d == 0 or not np.isfinite(atrn[t]) or atrn[t] <= 0 or not np.isfinite(d):
            continue
        entry = o[t + 1] + d * slip
        stop = entry - stop_mult * atrn[t] * d
        end = min(t + horizon, n - 1)
        exit_raw = None
        for k in range(t + 1, end + 1):
            if d > 0 and low[k] <= stop:
                exit_raw = min(stop, o[k])
                break
            if d < 0 and h[k] >= stop:
                exit_raw = max(stop, o[k])
                break
        if exit_raw is None:
            exit_raw = c[end]
        exit_fill = exit_raw - d * slip
        pnl_pts = (exit_fill - entry) * d
        R[t] = (pnl_pts - 2.0 * comm_pts) / (stop_mult * atrn[t])
    return pd.Series(R, index=idx)


def stat_block(R: pd.Series) -> dict:
    r = R.dropna().to_numpy(float)
    if r.size == 0:
        return {"n": 0, "mean_R": np.nan, "median_R": np.nan, "PF": np.nan, "hit": np.nan}
    wins = r[r > 0]
    losses = r[r < 0]
    gl = losses.sum()
    pf = float(wins.sum() / abs(gl)) if gl != 0 else float("inf")
    return {
        "n": int(r.size),
        "mean_R": float(r.mean()),
        "median_R": float(np.median(r)),
        "PF": pf,
        "hit": float(wins.size / r.size),
    }


def weekly_bootstrap_ci(R: pd.Series, n_boot: int = 400, seed: int = 7):
    r = R.dropna()
    if len(r) < 20:
        return (np.nan, np.nan)
    weeks = r.index.to_period("W")
    groups = [g.to_numpy(float) for _, g in r.groupby(weeks)]
    if len(groups) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        means[i] = np.concatenate([groups[j] for j in pick]).mean()
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _rank(a: np.ndarray) -> np.ndarray:
    order = a.argsort()
    r = np.empty(len(a), float)
    r[order] = np.arange(len(a))
    return r


def spearman(x, y) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    return float(np.corrcoef(_rank(x[m]), _rank(y[m]))[0, 1])


def classify_deciles(means: np.ndarray) -> str:
    means = np.asarray(means, float)
    if not np.all(np.isfinite(means)):
        return "drop"
    top, bot = means[-1], means[0]
    peak_idx = int(np.argmax(means))
    peak = means[peak_idx]
    rho = spearman(np.arange(len(means)), means)
    if peak_idx < len(means) - 1 and (peak - top) > max(1e-9, 0.15 * (peak - bot)):
        return "band"
    if top > bot and rho > 0.5:
        return "threshold"
    if top <= bot:
        return "drop"
    return "drop"


# --------------------------------------------------------------------------- #
# B2 breakout + 3xATR trailing stop engine (no lookahead)
# --------------------------------------------------------------------------- #
def run_breakout(bars: pd.DataFrame, atr: pd.Series, cost: CostCfg, donch: int = 20,
                 stop_mult: float = 3.0, allow_short: bool = False,
                 veto_long: pd.Series = None, veto_short: pd.Series = None) -> dict:
    idx = bars.index
    o = bars["open"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    low = bars["low"].to_numpy(float)
    c = bars["close"].to_numpy(float)
    atrn = atr.reindex(idx).to_numpy(float)
    n = len(bars)
    pv = float(cost.point_value)
    slip = float(cost.slip_ticks) * float(cost.tick)
    comm = float(cost.commission_per_side)

    don_hi = bars["high"].rolling(donch, min_periods=donch).max().shift(1).to_numpy(float)
    don_lo = bars["low"].rolling(donch, min_periods=donch).min().shift(1).to_numpy(float)
    vlong = veto_long.reindex(idx).fillna(True).to_numpy(bool) if veto_long is not None else np.zeros(n, bool)
    vshort = veto_short.reindex(idx).fillna(True).to_numpy(bool) if veto_short is not None else np.zeros(n, bool)

    realized = 0.0
    pos = 0
    entry_fill = entry_raw = 0.0
    entry_bar = 0
    water = 0.0
    pending = 0
    equity_usd = np.zeros(n)
    trades = []

    def close_trade(exit_bar, exit_raw, side):
        nonlocal realized, pos, trades
        exit_fill = exit_raw - side * slip
        pnl_pts = (exit_fill - entry_fill) * side
        pnl_usd = pnl_pts * pv - 2.0 * comm
        realized += pnl_pts * pv - comm
        trades.append({
            "entry_time": idx[entry_bar], "exit_time": idx[exit_bar], "side": int(side),
            "entry_px": entry_fill, "exit_px": exit_fill, "pnl_pts": float(pnl_pts),
            "pnl_usd_net": float(pnl_usd), "bars_held": int(exit_bar - entry_bar),
            "atr_entry": float(atrn[entry_bar]) if np.isfinite(atrn[entry_bar]) else np.nan,
        })
        pos = 0

    for k in range(n):
        if pending != 0 and pos == 0:
            side = pending
            pending = 0
            entry_fill = o[k] + side * slip
            entry_raw = o[k]
            entry_bar = k
            water = h[k] if side > 0 else low[k]
            realized -= comm
            pos = side
        if pos != 0 and k > 1 and np.isfinite(atrn[k - 1]):
            if pos > 0:
                stop = water - stop_mult * atrn[k - 1]
                if low[k] <= stop:
                    close_trade(k, min(stop, o[k]), pos)
                else:
                    water = max(water, h[k])
            else:
                stop = water + stop_mult * atrn[k - 1]
                if h[k] >= stop:
                    close_trade(k, max(stop, o[k]), pos)
                else:
                    water = min(water, low[k])
        unreal = pos * (c[k] - entry_fill) * pv if pos != 0 else 0.0
        equity_usd[k] = realized + unreal
        if pos == 0 and pending == 0 and k + 1 < n and np.isfinite(don_hi[k]):
            if c[k] > don_hi[k] and not vlong[k]:
                pending = 1
            elif allow_short and c[k] < don_lo[k] and not vshort[k]:
                pending = -1
    if pos != 0:
        close_trade(n - 1, c[n - 1], pos)
        equity_usd[n - 1] = realized

    capital = float(c[0]) * pv
    return {
        "equity": pd.Series(1.0 + equity_usd / capital, index=idx, name="equity"),
        "trades": trades,
        "capital": capital,
        "cost": cost,
    }


def run_strat(bars, res, ppy) -> dict:
    return metrics(res["equity"], res["trades"], bars.index, COST, ppy)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    bars = load_bars("60min", start=IS_START, end=IS_END)
    feat = compute_features(bars)
    close = bars["close"]
    high = bars["high"]
    low = bars["low"]
    atr = feat["atr"]
    ema50 = feat["ema50"]
    ema20 = ema(close, 20)
    slope3 = ema50.diff(3)
    dirn = np.sign(slope3)
    ext = (close - ema20) / atr
    slope_abs_atr = slope3.abs() / atr
    atr_pct = atr.rolling(500, min_periods=100).rank(pct=True)
    ppy = len(bars) / ((bars.index[-1] - bars.index[0]).days / 365.25)
    print(f"IS 60min bars={len(bars)}  [{bars.index[0]} .. {bars.index[-1]}]  ppy={ppy:.1f}\n")

    # canonical R used by D1 / D5
    R_dir = canonical_R(bars, dirn, atr, COST)

    decisions = {}

    # ------------------------------------------------------------------ D0
    fwd24 = close.shift(-H_LABEL) / close - 1.0
    d0 = []
    for label, mask in (("LONG (dir>0)", dirn > 0), ("SHORT (dir<0)", dirn < 0)):
        m = fwd24[mask].dropna()
        ann = float(m.mean() / H_LABEL * BARS_PER_YEAR)
        side_pnl = ann if "LONG" in label else -ann
        d0.append({"side": label, "n": int(m.size), "mean_fwd_ret": float(m.mean()),
                   "annualized_drift": ann, "side_contribution": side_pnl})
    d0 = pd.DataFrame(d0)
    print_frame("D0  conditional drift (H=24, annualized = mean/H * %d)" % BARS_PER_YEAR, d0)
    short_neg = float(d0.loc[d0["side"].str.contains("SHORT"), "side_contribution"].iloc[0]) <= 0
    decisions["D0"] = ("long-only (short drift negative on IS)" if short_neg
                       else "long+short allowed (short drift positive)")
    print(f"DECISION D0: {decisions['D0']}\n")
    save(d0, "d0_drift.csv")

    # ------------------------------------------------------------------ D1
    feats = {"ER20": feat["er"], "ADX14": feat["adx"], "VR12": feat["vr"],
             "ATR14_pct": atr_pct, "slopeAbsATR": slope_abs_atr, "extensionATR": ext}
    rows = []
    d1_dec = {}
    for fname, fs in feats.items():
        data = pd.DataFrame({"f": fs, "R": R_dir}).dropna()
        if len(data) < 100:
            rows.append({"feature": fname, "decile": -1, "n": len(data), "mean_R": np.nan,
                         "median_R": np.nan, "PF": np.nan, "hit": np.nan, "ci_lo": np.nan, "ci_hi": np.nan})
            d1_dec[fname] = "drop"
            continue
        data["decile"] = pd.qcut(data["f"], 10, labels=False, duplicates="drop")
        means = []
        for d, sub in data.groupby("decile"):
            st = stat_block(sub["R"])
            lo, hi = weekly_bootstrap_ci(sub["R"])
            means.append(st["mean_R"])
            rows.append({"feature": fname, "decile": int(d), "n": st["n"], "mean_R": st["mean_R"],
                         "median_R": st["median_R"], "PF": st["PF"], "hit": st["hit"],
                         "ci_lo": lo, "ci_hi": hi})
        d1_dec[fname] = classify_deciles(np.array(means))
    d1 = pd.DataFrame(rows)
    print_frame("D1  feature deciles vs canonical R (dir=sign(slope3))", d1)
    for k, v in d1_dec.items():
        print(f"  D1 {k:>12}: {v}")
    decisions["D1"] = d1_dec
    print("DECISION D1: thresholds from monotone features; bands for hump-shaped; drop flat/negative\n")
    save(d1, "d1_feature_deciles.csv")

    # ------------------------------------------------------------------ D2
    context_long = (close > ema50) & (slope3 > 0)
    don_hi20 = high.rolling(20, min_periods=20).max().shift(1)
    ev_a = context_long & (close > don_hi20)
    peak10 = high.rolling(10, min_periods=10).max()
    depth = (peak10 - low) / atr
    prior_pb = depth.rolling(5, min_periods=1).max()
    ev_b = context_long & (prior_pb >= 1.0) & (close > high.shift(1))
    d2_rows = []
    for name, ev in (("a_donchian20_breakout", ev_a), ("b_pullback_resumption", ev_b)):
        R = canonical_R(bars, (ev & context_long).astype(int), atr, COST)
        st = stat_block(R)
        d2_rows.append({"trigger": name, **st, "exp_x_sqrtn": st["mean_R"] * np.sqrt(st["n"]) if st["n"] else np.nan})
    d2 = pd.DataFrame(d2_rows)
    # depth buckets for event b
    evb_bars = bars[ev_b & context_long]
    if len(evb_bars):
        db = prior_pb.reindex(evb_bars.index).dropna()
        Rb = canonical_R(bars, (ev_b & context_long).astype(int), atr, COST).reindex(db.index)
        bins = pd.cut(db, [-np.inf, 1, 2, 3, np.inf], labels=["<1", "1-2", "2-3", ">3"])
        d2b_rows = []
        for lab, sub in Rb.groupby(bins):
            st = stat_block(sub)
            d2b_rows.append({"depth_ATR": str(lab), **st})
        d2_depth = pd.DataFrame(d2b_rows)
    else:
        d2_depth = pd.DataFrame()
    print_frame("D2  trigger families (long context)", d2)
    print_frame("D2b pullback-resumption by depth bucket", d2_depth)
    primary = d2.sort_values("exp_x_sqrtn", ascending=False)["trigger"].iloc[0] if len(d2) else "a_donchian20_breakout"
    decisions["D2_primary"] = primary
    decisions["D2_depth"] = "depth buckets license the pullback threshold"
    print(f"DECISION D2: primary trigger = {primary} (by expectancy*sqrt(n)); see depth buckets\n")
    save(d2, "d2_triggers.csv")
    save(d2_depth, "d2_depth.csv")

    # ------------------------------------------------------------------ D3
    ext_veto_long = ext > 2.5
    vr_veto = feat["vr"] >= 1.0
    vol_veto = atr_pct > 0.90
    tod_veto = pd.Series(bars.index.hour.isin([16, 17]), index=bars.index)
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

    def veto_short_mirror(s):
        # mirror extension for shorts; VR/vol/tod same
        return veto_l(s)

    ext_veto_short = (ext < -2.5)
    rows = []

    def add_row(name, res, allow_short=False, ungated_n=None):
        m = run_strat(bars, res, ppy)
        nratio = m["n_trades"] / ungated_n if ungated_n else 1.0
        rows.append({"variant": name, "n": m["n_trades"], "n_ratio": nratio, "MAR": m["MAR"],
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
    # symmetric variant
    b2_sym = run_breakout(bars, atr, COST, allow_short=True,
                          veto_long=full, veto_short=veto_l(ext_veto_short | vr_veto | vol_veto | tod_veto))
    add_row("B2_sym FULL (long+short)", b2_sym, ungated_n=None)
    d3 = pd.DataFrame(rows)
    print_frame("D3  B2 breakout + 3xATR trailing stop, veto deltas @1.5 ticks", d3)

    base_m = rows[0]
    single = []
    for r in rows[1:5]:
        delta_pf = r["PF"] - base_m["PF"]
        rel_mar = (r["MAR"] - base_m["MAR"]) / abs(base_m["MAR"]) if base_m["MAR"] else np.nan
        keep = r["n_ratio"] >= 0.60 and (delta_pf >= 0.05 or rel_mar >= 0.15)
        single.append((r["variant"], delta_pf, rel_mar, r["n_ratio"], keep))
    best_single = max([s for s in single if s[4]] or single, key=lambda x: x[2])
    full_row = [r for r in rows if r["variant"] == "B2g=FULL stack"][0]
    full_beats = full_row["MAR"] >= base_m["MAR"] and (
        full_row["PF"] >= best_single[1] + base_m["PF"] or full_row["MAR"] >= best_single[2] * abs(base_m["MAR"]) + base_m["MAR"]
    )
    decisions["D3_keep"] = [s[0] for s in single if s[4]] or ["none"]
    decisions["D3_stack"] = ("FULL stack" if full_beats else f"best single veto only ({best_single[0]})")
    inc_pf = full_row["PF"] - base_m["PF"]
    inc_mar = full_row["MAR"] - base_m["MAR"]
    decisions["D3_gate_increment_positive"] = bool(inc_pf > 0 or inc_mar > 0)
    print("DECISION D3: keep per rule ->", decisions["D3_keep"])
    print(f"  FULL stack beats best single? {full_beats} -> {decisions['D3_stack']}")
    print(f"  gate-as-veto increment B2->B2g: dPF={inc_pf:+.3f} dMAR={inc_mar:+.3f} "
          f"positive={decisions['D3_gate_increment_positive']}\n")
    save(d3, "d3_vetoes.csv")

    # ------------------------------------------------------------------ D4
    fwd12 = close.shift(-12) / close - 1.0
    hours = bars.index.hour
    d4_agg = pd.DataFrame({"hour": hours, "fwd_ret": fwd12}).dropna().groupby("hour").agg(
        n=("fwd_ret", "size"), mean_fwd_ret=("fwd_ret", "mean"), vol_fwd_ret=("fwd_ret", "std"))
    # B1 realized trade pnl by entry hour (existing M1 target)
    target_b1 = ((feat["gate"] == 1) & (feat["dir"] > 0)).astype(int)
    b1_res = run_target_position(bars, target_b1, COST, allow_short=False, atr=atr)
    tr = pd.DataFrame([{"hour": t["entry_time"].hour, "pnl": t["pnl_usd_net"]} for t in b1_res["trades"]])
    if len(tr):
        hb = tr.groupby("hour").agg(b1_n=("pnl", "size"), b1_mean_pnl=("pnl", "mean"))
        hb["b1_entry_share"] = hb["b1_n"] / hb["b1_n"].sum()
    else:
        hb = pd.DataFrame()
    d4 = d4_agg.join(hb, how="left").reset_index()
    print_frame("D4  forward-12 return / vol by ET hour + B1 realized PnL by entry hour", d4)
    bad_fwd = d4[(d4["mean_fwd_ret"] < 0) & (d4["b1_entry_share"].fillna(0) >= 0.05)]
    bad_pnl = d4[(d4["b1_mean_pnl"] < 0) & (d4["b1_entry_share"].fillna(0) >= 0.05)]
    if len(bad_fwd):
        hrs = sorted(int(x) for x in bad_fwd["hour"])
        decisions["D4"] = f"restrict entries away from ET hours {hrs} (neg fwd drift & >=5% of trades)"
    elif len(bad_pnl):
        hrs = sorted(int(x) for x in bad_pnl["hour"])
        decisions["D4"] = (f"no fwd-drift restriction (all hours +); realized B1 expectancy <0 at ET hours {hrs} "
                           f"(each >=5% of entries, but small n<=14) -> optional ET entry-window veto")
    else:
        decisions["D4"] = "no time-of-day entry restriction justified"
    print(f"DECISION D4: {decisions['D4']}\n")
    save(d4, "d4_hourly.csv")

    # ------------------------------------------------------------------ D5
    ev_primary = ev_a if primary.startswith("a") else ev_b
    prim_dir = (ev_primary & context_long).astype(int)
    Rprim = canonical_R(bars, prim_dir, atr, COST)
    data5 = pd.DataFrame({"atr_pct": atr_pct, "R": Rprim}).dropna()
    rows5 = []
    if len(data5) > 50:
        data5["decile"] = pd.qcut(data5["atr_pct"], 10, labels=False, duplicates="drop")
        for d, sub in data5.groupby("decile"):
            st = stat_block(sub["R"])
            rows5.append({"decile": int(d), **st})
    d5 = pd.DataFrame(rows5)
    print_frame("D5  ATR14-percentile deciles x expectancy (primary trigger)", d5)
    if len(d5) and len(d5) >= 3:
        peak_dec = int(d5.loc[d5["mean_R"].idxmax(), "decile"])
        top_mean = d5["mean_R"].iloc[-1]
        mid_mean = d5["mean_R"].max()
        if top_mean < mid_mean - 1e-9 and peak_dec < d5["decile"].max():
            decisions["D5"] = (f"vol band: top ATR decile underperforms peak ({top_mean:.3f} vs {mid_mean:.3f}) "
                               f"but is still >0 -> band as a risk/DD veto, not an expectancy veto")
        elif d5["mean_R"].iloc[-1] >= d5["mean_R"].iloc[0]:
            decisions["D5"] = "ATR monotone -> no vol band"
        else:
            decisions["D5"] = "ATR flat/negative -> drop vol from veto set"
    else:
        decisions["D5"] = "insufficient trigger sample"
    print(f"DECISION D5: {decisions['D5']}\n")
    save(d5, "d5_volband.csv")

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
        # run lengths
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
    print_frame("D6  persistence of regime features", d6)
    low_pers = [r["feature"] for _, r in d6.iterrows() if r["acf12"] < 0.5 or r["med_dur_ON"] < 4]
    decisions["D6"] = f"entry-time filters only (never state machine): {low_pers or ['none']}"
    print(f"DECISION D6: {decisions['D6']}\n")
    save(d6, "d6_persistence.csv")

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
    print_frame("D7  forward path-efficiency decomposition by gate @H=24", d7, floatfmt="%.6f")
    eff_inv = d7.loc[1, "mean_path_eff"] < d7.loc[0, "mean_path_eff"]
    drift_inv = d7.loc[1, "mean_abs_drift"] > d7.loc[0, "mean_abs_drift"]
    vol_inv = d7.loc[1, "mean_fwd_vol"] > d7.loc[0, "mean_fwd_vol"]
    if eff_inv and drift_inv and vol_inv:
        decisions["D7"] = "CONFIRMED vol artifact: gate-ON has higher fwd vol + higher |drift| but lower ratio"
    elif eff_inv:
        decisions["D7"] = "REFUTED/partial: inversion not explained by vol denominator"
    else:
        decisions["D7"] = "no inversion observed"
    print(f"DECISION D7: {decisions['D7']}\n")
    save(d7, "d7_efficiency_decomp.csv")

    # ------------------------------------------------------------------ FINAL
    b2_metrics = run_strat(bars, b2_long, ppy)
    b2g_metrics = run_strat(bars, b2_full, ppy)
    print("=" * 78)
    print("B2 vs B2g @1.5 ticks (IS only)")
    print("=" * 78)
    for name, m in (("B2 (ungated)", b2_metrics), ("B2g (full veto stack)", b2g_metrics)):
        print(f"  {name:<26} n={m['n_trades']:>4}  MAR={m['MAR']:.3f}  PF={m['profit_factor']:.3f}  "
              f"MaxDD={m['MaxDD']:.3f}  hit={m['hit_rate']:.3f}  expR={m['expectancy_R']:.3f}  CAGR={m['CAGR']:.4f}")
    print()

    print("=" * 78)
    print("DECISIONS")
    print("=" * 78)
    print(f"  D0 long/short default : {decisions['D0']}")
    print(f"  D1 feature roles      : {decisions['D1']}")
    print(f"  D2 primary trigger    : {decisions['D2_primary']}")
    print(f"  D3 vetoes kept        : {decisions['D3_keep']}")
    print(f"  D3 stack choice       : {decisions['D3_stack']}")
    print(f"  D3 gate-as-veto +?    : {decisions['D3_gate_increment_positive']}")
    print(f"  D4 time-of-day        : {decisions['D4']}")
    print(f"  D5 vol band           : {decisions['D5']}")
    print(f"  D6 persistence rule   : {decisions['D6']}")
    print(f"  D7 M1 explanation     : {decisions['D7']}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
