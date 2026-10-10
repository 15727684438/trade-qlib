"""M3 research stages on 15min decisions / 1min execution (IN-SAMPLE ONLY).

Stage 0  MFE/MAE forensics on the B2g realized trade log.
Stage A  exit shaping (trail / breakeven / time stop / partial / 5min structure).
Stage B  veto re-derivation from the Stage-A winner's realized buckets.
Stage C  trigger variants (breakout-retest, Donchian length).
Robustness R1-R4 + Stage E1 cost realism; freeze only if R1-R4 pass.

RULE enforced here: every ranking uses the REALIZED net-of-cost trade ledger from
the full engine. canonical_R is never used to rank (only the base config is known
a priori), so the D2 fixed-horizon failure cannot recur.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(_PKG_ROOT))

from tf_mnq import IS_END, IS_START  # noqa: E402
from tf_mnq.analytics import curve_quality, returns_metrics  # noqa: E402
from tf_mnq.breakout import run_breakout_ex, run_structure_stop_ff  # noqa: E402
from tf_mnq.data import load_bars  # noqa: E402
from tf_mnq.engine import CostCfg, metrics  # noqa: E402
from tf_mnq.features import compute_features  # noqa: E402
from tf_mnq.strategies import strategy_frame, veto_blocked  # noqa: E402

RESULTS_DIR = os.path.join(_PKG_ROOT, "results")
CONFIG_DIR = os.path.join(_PKG_ROOT, "config")
COST = CostCfg(slip_ticks=0.4, commission_per_side=0.5, tick=0.25, point_value=2.0)
COST3 = CostCfg(slip_ticks=3.0, commission_per_side=0.5, tick=0.25, point_value=2.0)
FREQ = "15min"
DONCH_CHAIN = [12, 20, 32]


def save(df, name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, name), index=False)


def print_frame(title, df, floatfmt="%.4f"):
    print(title)
    if df is None or len(df) == 0:
        print("  (empty)")
    else:
        with pd.option_context("display.width", 260, "display.max_columns", 80):
            print(df.to_string(index=False, float_format=lambda x: floatfmt % x))
    print()


# --------------------------------------------------------------------------- #
# globals filled in main()
# --------------------------------------------------------------------------- #
G = {}


def run_cfg(cfg, cost=None):
    cost = cost or COST
    extra = cfg.get("veto_extra")
    blocked = G["base_blocked"] if extra is None else (G["base_blocked"] | extra)
    bars, feat, atr, sf = G["bars"], G["feat"], G["atr"], G["sf"]
    if cfg.get("exit_engine", "atr") == "structure5":
        don = cfg.get("donch", 20)
        don_hi = bars["high"].rolling(don, min_periods=don).max().shift(1)
        sig = ((bars["close"] > don_hi) & (~blocked)).fillna(False)
        return run_structure_stop_ff(bars, G["bars5"], G["bars1"], feat, sig, cost,
                                     stop_mult=cfg.get("struct_stop_mult", 1.5),
                                     buffer_mult=cfg.get("buffer_mult", 0.25),
                                     pivot_n=cfg.get("pivot_n", 2))
    return run_breakout_ex(
        bars, atr, cost, donch=cfg.get("donch", 20), stop_mult=cfg.get("stop_mult", 3.0),
        veto_long=blocked, breakeven_atr=cfg.get("breakeven_atr"),
        time_stop_bars=cfg.get("time_stop_bars"), partial_atr=cfg.get("partial_atr"),
        partial_frac=0.5, entry_mode=cfg.get("entry_mode", "market"),
        retest_bars=cfg.get("retest_bars", 3),
    )


def mrow(res, tag, cost=None):
    cost = cost or COST
    bars, ppy = G["bars"], G["ppy"]
    m = metrics(res["equity"], res["trades"], bars.index, cost, ppy)
    cq = curve_quality(res["equity"], ppy)
    rm = returns_metrics(res["equity"], ppy, bars.index)
    row = {"tag": tag, "n": m["n_trades"], "PF": m["profit_factor"], "win": m["hit_rate"],
           "MAR": m["MAR"], "MaxDD": m["MaxDD"], "CAGR": m["CAGR"], "expR": m["expectancy_R"],
           "R2": cq["r2_log_equity_vs_time"], "pct_pos12m": cq["pct_positive_rolling_12m"],
           "Sharpe": rm["Sharpe"], "Sortino": rm["Sortino"], "Calmar": rm["Calmar"]}
    row["feasible"] = bool(row["MaxDD"] <= 0.12 and row["n"] >= 400
                           and row["R2"] >= 0.75 and row["pct_pos12m"] >= 0.70)
    return row


def save_trades(res, tag, stage):
    save(pd.DataFrame(res["trades"]), f"{stage}_{tag}_trades.csv")


def pf_of(pnl):
    pnl = np.asarray(pnl, float)
    w = pnl[pnl > 0].sum()
    l = -pnl[pnl < 0].sum()
    return float(w / l) if l > 0 else (np.inf if w > 0 else 0.0)


def sub_pf(trades, lo, hi):
    s = [t["pnl_usd_net"] for t in trades if lo <= t["entry_time"] <= hi]
    return pf_of(s), len(s)


# --------------------------------------------------------------------------- #
def stage0(base_res):
    print("=" * 78); print("STAGE 0 - MFE/MAE FORENSICS (B2g realized trades)"); print("=" * 78)
    bars1 = G["bars1"]
    t1 = bars1.index.to_numpy()
    h1 = bars1["high"].to_numpy(float)
    l1 = bars1["low"].to_numpy(float)
    step = np.timedelta64(15, "m")
    rows = []
    for tr in base_res["trades"]:
        s = int(np.searchsorted(t1, np.datetime64(tr["entry_time"])))
        e = int(np.searchsorted(t1, np.datetime64(tr["exit_time"]) + step))
        if e <= s:
            e = s + 1
        hi = h1[s:e]; lo = l1[s:e]
        entry = tr["entry_px"]; atr = tr["atr_entry"]
        mfe = float(np.max(hi) - entry)
        mae = float(entry - np.min(lo))
        run = np.maximum.accumulate(hi) - entry
        final_mfe = float(run[-1])
        nb = len(run)
        f25 = float(run[int(0.25 * (nb - 1))] / final_mfe) if final_mfe > 0 else np.nan
        f50 = float(run[int(0.50 * (nb - 1))] / final_mfe) if final_mfe > 0 else np.nan
        rows.append({
            "pnl_usd_net": tr["pnl_usd_net"], "winner": bool(tr["pnl_usd_net"] > 0),
            "atr_entry": atr, "mfe_pts": mfe, "mae_pts": mae,
            "mfe_atr": mfe / atr if atr > 0 else np.nan,
            "mae_atr": mae / atr if atr > 0 else np.nan,
            "dur_1min": nb, "bars_held": tr["bars_held"],
            "frac_mfe_25": f25, "frac_mfe_50": f50,
            "exit_reason": tr.get("exit_reason", ""),
        })
    s0 = pd.DataFrame(rows)
    save(s0, "s0_mfe_mae.csv")

    def q(x):
        x = np.asarray(x, float); x = x[np.isfinite(x)]
        if not x.size:
            return {}
        return {"p50": float(np.percentile(x, 50)), "p75": float(np.percentile(x, 75)),
                "p90": float(np.percentile(x, 90)), "p95": float(np.percentile(x, 95)),
                "mean": float(x.mean())}
    W = s0[s0["winner"]]; L = s0[~s0["winner"]]
    summ = []
    for grp, sub in (("WINNER", W), ("LOSER", L)):
        summ.append({"group": grp, "n": len(sub),
                     "mae_atr_p50": q(sub["mae_atr"]).get("p50"),
                     "mae_atr_p75": q(sub["mae_atr"]).get("p75"),
                     "mae_atr_p90": q(sub["mae_atr"]).get("p90"),
                     "mae_atr_p95": q(sub["mae_atr"]).get("p95"),
                     "mfe_atr_p50": q(sub["mfe_atr"]).get("p50"),
                     "mfe_atr_p90": q(sub["mfe_atr"]).get("p90"),
                     "frac25_p50": q(sub["frac_mfe_25"]).get("p50"),
                     "frac50_p50": q(sub["frac_mfe_50"]).get("p50"),
                     "dur_p50": q(sub["dur_1min"]).get("p50")})
    summary = pd.DataFrame(summ)
    print_frame("S0 distribution by WINNER/LOSER (ATR-normalised + MFE-path shape)", summary)
    save(summary, "s0_summary.csv")

    w_mae95 = q(W["mae_atr"]).get("p95")
    l_mae50 = q(L["mae_atr"]).get("p50")
    w_f50 = q(W["frac_mfe_50"]).get("p50")
    w_f25 = q(W["frac_mfe_25"]).get("p50")
    # stop-savings if losers were capped at k*ATR
    savings = []
    for k in (1.0, 1.5, 2.0, 2.5):
        capped = np.minimum(L["mae_atr"], k)  # actual loss capped at k (approx, pre-cost)
        savings.append({"stop_at_ATR": k, "loss_atr_saved_p50": l_mae50 - float(capped.mean())})
    print_frame("S0 loser stop-cap saving (mean MAE when capped)", pd.DataFrame(savings))
    dec = {
        "safe_initial_stop_mult": f"~{w_mae95:.2f} ATR (95th pct of winners' MAE)",
        "loser_MAE_median": f"{l_mae50:.2f} ATR",
        "mfe_path": f"winners reach {w_f25:.0%} of final MFE in first 25%, {w_f50:.0%} in first 50%",
        "partials": ("help (early MFE concentration)" if (w_f50 or 0) >= 0.6
                     else "amputate tails (MFE arrives late)"),
    }
    for k, v in dec.items():
        print(f"  S0 decision {k}: {v}")
    print()
    return s0, summary, dec


# --------------------------------------------------------------------------- #
def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CONFIG_DIR, exist_ok=True)

    bars = load_bars(FREQ, start=IS_START, end=IS_END)
    bars5 = load_bars("5min", start=IS_START, end=IS_END)
    bars1 = load_bars("1min", start=IS_START, end=IS_END)
    feat = compute_features(bars)
    sf = strategy_frame(bars, feat)
    atr = feat["atr"]
    base_blocked = veto_blocked(bars, feat, sf, components=["vol", "vr"])
    ppy = len(bars) / ((bars.index[-1] - bars.index[0]).days / 365.25)
    assert bars.index.max() <= pd.Timestamp(IS_END)
    G.update({"bars": bars, "bars5": bars5, "bars1": bars1, "feat": feat, "sf": sf,
              "atr": atr, "base_blocked": base_blocked, "ppy": ppy})
    print(f"IS 15min={len(bars)} 5min={len(bars5)} 1min={len(bars1)} "
          f"[{bars.index[0]} .. {bars.index[-1]}] ppy={ppy:.1f}\n")

    t_start = time.perf_counter()

    # ---------------- base B2g ----------------
    base_cfg = {"donch": 20, "stop_mult": 3.0, "entry_mode": "market"}
    base_res = run_cfg(base_cfg)
    base_row = mrow(base_res, "B2g_base")
    print_frame("BASE B2g @1.5 ticks", pd.DataFrame([base_row]))

    # ---------------- Stage 0 ----------------
    s0, s0sum, s0dec = stage0(base_res)

    # ---------------- Stage A ----------------
    print("=" * 78); print("STAGE A - EXIT SHAPING (rank by realized net PF)"); print("=" * 78)
    a_rows = []
    res_store = {}
    for m in (2.0, 2.5, 3.0):
        cfg = {**base_cfg, "stop_mult": m}
        res = run_cfg(cfg)
        res_store[f"trail{m}"] = res
        row = mrow(res, f"A1_trail{m}")
        a_rows.append(row)
        save_trades(res, f"trail{m}", "stageA")
    a1 = pd.DataFrame(a_rows)
    print_frame("A1 trail multiplier", a1)
    feas = a1[a1["feasible"]]
    a1_best = (feas if len(feas) else a1).sort_values("PF", ascending=False).iloc[0]
    winner_m = float(a1_best["tag"].replace("A1_trail", ""))
    incumbent = {**base_cfg, "stop_mult": winner_m}
    inc_res = res_store[f"trail{winner_m}"]
    inc_row = mrow(inc_res, f"incumbent_A1_trail{winner_m}")
    print(f"  A1 advance: trail={winner_m} PF={inc_row['PF']:.4f} MaxDD={inc_row['MaxDD']:.4f} "
          f"feasible={inc_row['feasible']}\n")

    # A2 breakeven
    cfg = {**incumbent, "breakeven_atr": 1.0}
    res = run_cfg(cfg)
    row = mrow(res, "A2_breakeven")
    save_trades(res, "breakeven", "stageA")
    keep_a2 = row["PF"] >= inc_row["PF"] + 0.03
    print_frame("A2 breakeven (+1.0 ATR)", pd.DataFrame([inc_row, row]))
    print(f"  A2 {'KEPT' if keep_a2 else 'DROPPED'} (PF {row['PF']:.4f} vs {inc_row['PF']:.4f}, need +0.03)\n")
    if keep_a2:
        incumbent, inc_res, inc_row = cfg, res, row

    # A3 time stop
    cfg = {**incumbent, "time_stop_bars": 16, "time_stop_min_pnl": 0.0}
    res = run_cfg(cfg)
    row = mrow(res, "A3_timestop16")
    save_trades(res, "timestop16", "stageA")
    keep_a3 = row["PF"] >= inc_row["PF"] + 0.03
    print_frame("A3 time stop (16 bars, PnL<0)", pd.DataFrame([inc_row, row]))
    print(f"  A3 {'KEPT' if keep_a3 else 'DROPPED'} (PF {row['PF']:.4f} vs {inc_row['PF']:.4f}, need +0.03)\n")
    if keep_a3:
        incumbent, inc_res, inc_row = cfg, res, row

    # A4 partial (curve variant only)
    cfg = {**incumbent, "partial_atr": 1.5}
    res = run_cfg(cfg)
    save_trades(res, "partial15", "stageA")
    a4 = mrow(res, "A4_partial_1.5ATR")
    dd_improve = (inc_row["MaxDD"] - a4["MaxDD"]) * 100.0
    keep_a4 = bool(a4["pct_pos12m"] >= 0.75 and dd_improve >= 2.0)
    print_frame("A4 partial 50% @+1.5ATR (curve variant only)", pd.DataFrame([inc_row, a4]))
    print(f"  A4 curve-variant {'AVAILABLE' if keep_a4 else 'not selected'} "
          f"(dMaxDD={dd_improve:+.2f}pts, %pos12m={a4['pct_pos12m']:.3f})\n")

    # S3 structure stop on 5min pivots
    cfg = {**incumbent, "exit_engine": "structure5", "struct_stop_mult": 1.5, "buffer_mult": 0.25, "pivot_n": 2}
    res = run_cfg(cfg)
    save_trades(res, "structure5", "stageA")
    s3 = mrow(res, "S3_structure5min")
    keep_s3 = s3["PF"] >= inc_row["PF"] + 0.03 and s3["n"] >= 400
    print_frame("S3 structure-low stop on 5min pivots", pd.DataFrame([inc_row, s3]))
    print(f"  S3 {'KEPT' if keep_s3 else 'RETIRED (structure stop for v1)'} "
          f"(PF {s3['PF']:.4f} vs {inc_row['PF']:.4f}, need +0.03)\n")
    if keep_s3:
        incumbent, inc_res, inc_row = cfg, res, s3

    stageA_summary = pd.concat([a1, pd.DataFrame([inc_row, a4, s3])], ignore_index=True)
    save(stageA_summary, "stageA_summary.csv")

    # ---------------- Stage B ----------------
    print("=" * 78); print("STAGE B - VETO RE-DERIVATION ON REALIZED TRADES"); print("=" * 78)
    trades = inc_res["trades"]
    fseries = {
        "VR": feat["vr"],
        "ATRpct": sf["atr_pct"],
        "ER20": feat["er"],
        "extension": sf["ext"],
        "pullback_depth": (bars["high"].rolling(10, min_periods=10).max() - bars["low"]) / atr,
        "hour": pd.Series(bars.index.hour, index=bars.index),
        "dow": pd.Series(bars.index.dayofweek, index=bars.index),
    }
    b_rows = []
    flagged_masks = []
    for fname, fs in fseries.items():
        vals = fs.to_numpy(float)[[t["signal_bar"] for t in trades]]
        pnl = np.array([t["pnl_usd_net"] for t in trades])
        if fname in ("hour", "dow"):
            groups = pd.Series(vals).groupby(vals)
            iterator = [(int(v), idx) for v, idx in groups.groups.items()]
            for v, gidx in iterator:
                idx = np.array(list(gidx))
                n = len(idx); pf = pf_of(pnl[idx]); win = float((pnl[idx] > 0).mean())
                b_rows.append({"feature": fname, "bucket": str(v), "n": n, "PF": pf,
                               "win": win, "mask_kind": "eq", "val": v})
        else:
            qq = pd.qcut(pd.Series(vals), 10, labels=False, duplicates="drop").to_numpy()
            cats = pd.qcut(pd.Series(vals), 10, duplicates="drop").cat.categories
            for d in sorted(pd.unique(qq[~pd.isna(qq)])):
                idx = np.where(qq == d)[0]
                if not len(idx):
                    continue
                n = len(idx); pf = pf_of(pnl[idx]); win = float((pnl[idx] > 0).mean())
                iv = cats[int(d)]
                lo = float(iv.left); hi = float(iv.right)
                b_rows.append({"feature": fname, "bucket": f"d{int(d)}", "n": n, "PF": pf,
                               "win": win, "mask_kind": "range", "lo": lo, "hi": hi})
    buckets = pd.DataFrame(b_rows)
    save(buckets, "stageB_buckets.csv")
    print_frame("B1 realized PF/win/n by entry-feature bucket", buckets, "%.4f")

    bad = buckets[(buckets["PF"] < 0.85) & (buckets["n"] >= 80)]
    save(bad[["feature", "bucket", "n", "PF", "win", "mask_kind", "lo", "hi", "val"]],
         "stageB_added_vetoes.csv")
    G["bad_buckets"] = bad
    print(f"  flagged bad buckets (PF<0.85 & n>=80): {len(bad)}")
    if len(bad):
        print(bad[["feature", "bucket", "n", "PF", "win"]].to_string(index=False))
    mask = pd.Series(False, index=bars.index)
    for _, r in bad.iterrows():
        fs = fseries[r["feature"]]
        if r["mask_kind"] == "eq":
            mask = mask | (fs == r["val"])
        else:
            mask = mask | ((fs > r["lo"]) & (fs <= r["hi"]))
    b2_cfg = {**incumbent, "veto_extra": mask}
    b2_res = run_cfg(b2_cfg)
    b2 = mrow(b2_res, "B2_added_vetoes")
    save_trades(b2_res, "added_vetoes", "stageB")
    keep_b2 = b2["PF"] >= inc_row["PF"] + 0.05 and b2["n"] >= 400
    print_frame("B2 add flagged-bucket vetoes", pd.DataFrame([inc_row, b2]))
    print(f"  B2 {'KEPT' if keep_b2 else 'DROPPED'} (PF {b2['PF']:.4f} vs {inc_row['PF']:.4f}, "
          f"n={b2['n']}, need +0.05 & n>=400)\n")
    if keep_b2:
        incumbent, inc_res, inc_row = b2_cfg, b2_res, b2

    # ---------------- Stage C ----------------
    print("=" * 78); print("STAGE C - TRIGGER VARIANTS"); print("=" * 78)
    c1_cfg = {**incumbent, "entry_mode": "retest", "retest_bars": 3}
    c1_res = run_cfg(c1_cfg)
    c1 = mrow(c1_res, "C1_retest")
    save_trades(c1_res, "retest", "stageC")
    fill_rate = (len(c1_res["trades"]) / c1_res["n_breakouts"]) if c1_res["n_breakouts"] else np.nan
    keep_c1 = c1["PF"] >= inc_row["PF"] + 0.05 and c1["n"] >= 400
    print_frame("C1 breakout-retest (limit at breakout level, <=3 bars)", pd.DataFrame([inc_row, c1]))
    print(f"  C1 fill rate={fill_rate:.3f} ({len(c1_res['trades'])}/{c1_res['n_breakouts']}); "
          f"{'KEPT' if keep_c1 else 'DROPPED'} (PF {c1['PF']:.4f} vs {inc_row['PF']:.4f}, need +0.05)\n")
    if keep_c1:
        incumbent, inc_res, inc_row = c1_cfg, c1_res, c1

    donch_rows = []
    for d in DONCH_CHAIN:
        cfg = {**incumbent, "donch": d}
        res = run_cfg(cfg)
        row = mrow(res, f"C_donch{d}")
        save_trades(res, f"donch{d}", "stageC")
        donch_rows.append(row)
    donch = pd.DataFrame(donch_rows)
    print_frame("C2/C3/C4 Donchian length", donch)
    print()

    final_cfg = incumbent
    final_res = inc_res
    final_row = inc_row
    print(f"FINAL candidate cfg: { {k: v for k, v in final_cfg.items() if k != 'veto_extra'} } "
          f"veto_extra={'yes' if final_cfg.get('veto_extra') is not None else 'no'}\n")

    # ---------------- Robustness ----------------
    print("=" * 78); print("ROBUSTNESS BATTERY (final candidate)"); print("=" * 78)
    pnl = np.array([t["pnl_usd_net"] for t in final_res["trades"]], float)
    etimes = pd.Series([t["entry_time"] for t in final_res["trades"]])
    # R1 stationary block bootstrap by calendar week
    weeks = etimes.dt.to_period("W")
    groups = [pnl[weeks == w] for w in weeks.unique()]
    rng = np.random.default_rng(12345)
    pfs = np.empty(2000)
    for i in range(2000):
        pick = rng.integers(0, len(groups), len(groups))
        samp = np.concatenate([groups[j] for j in pick])
        pfs[i] = pf_of(samp)
    pfs = pfs[np.isfinite(pfs)]
    r1_lower = float(np.percentile(pfs, 5))
    r1_pass = r1_lower > 1.05
    # R2 sub-periods
    bounds = [("2019H2-2020", pd.Timestamp("2019-05-06"), pd.Timestamp("2020-12-31")),
              ("2021", pd.Timestamp("2021-01-01"), pd.Timestamp("2021-12-31")),
              ("2022", pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
              ("2023", pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31"))]
    r2_rows = []
    for name, lo, hi in bounds:
        p, nn = sub_pf(final_res["trades"], lo, hi)
        r2_rows.append({"period": name, "n": nn, "PF": p})
    r2 = pd.DataFrame(r2_rows)
    r2_pos = int((r2["PF"] > 1.0).sum())
    r2_min = float(r2["PF"].min())
    r2_pass = bool(r2_pos >= 3 and r2_min >= 0.85)
    # R3 neighborhood (one-at-a-time +/- step)
    def neighbor_cfgs(cfg):
        out = []
        if cfg.get("exit_engine", "atr") == "atr":
            d = cfg.get("donch", 20)
            for v in DONCH_CHAIN:
                if v != d:
                    out.append((f"donch={v}", {**cfg, "donch": v}))
            for v in (cfg.get("stop_mult", 3.0) - 0.5, cfg.get("stop_mult", 3.0) + 0.5):
                out.append((f"stop_mult={v:.1f}", {**cfg, "stop_mult": v}))
            if cfg.get("breakeven_atr") is not None:
                for v in (cfg["breakeven_atr"] - 0.5, cfg["breakeven_atr"] + 0.5):
                    out.append((f"breakeven_atr={v:.1f}", {**cfg, "breakeven_atr": v}))
            if cfg.get("time_stop_bars") is not None:
                for v in (cfg["time_stop_bars"] - 4, cfg["time_stop_bars"] + 4):
                    out.append((f"time_stop_bars={v}", {**cfg, "time_stop_bars": v}))
            if cfg.get("entry_mode") == "retest":
                for v in (cfg.get("retest_bars", 3) - 1, cfg.get("retest_bars", 3) + 1):
                    out.append((f"retest_bars={v}", {**cfg, "retest_bars": v}))
        else:
            for v in (cfg.get("struct_stop_mult", 1.5) - 0.5, cfg.get("struct_stop_mult", 1.5) + 0.5):
                out.append((f"struct_stop_mult={v:.1f}", {**cfg, "struct_stop_mult": v}))
            for v in (cfg.get("buffer_mult", 0.25) - 0.25, cfg.get("buffer_mult", 0.25) + 0.25):
                out.append((f"buffer_mult={v:.2f}", {**cfg, "buffer_mult": v}))
            for v in (cfg.get("pivot_n", 2) - 1, cfg.get("pivot_n", 2) + 1):
                out.append((f"pivot_n={v}", {**cfg, "pivot_n": v}))
        return out
    r3_rows = []
    for label, ncfg in neighbor_cfgs(final_cfg):
        nres = run_cfg(ncfg)
        nrow = mrow(nres, label)
        r3_rows.append({"neighbor": label, "PF": nrow["PF"], "n": nrow["n"],
                        "MaxDD": nrow["MaxDD"]})
    r3 = pd.DataFrame(r3_rows)
    r3_by_param = {}
    for _, r in r3.iterrows():
        p = r["neighbor"].split("=")[0]
        r3_by_param.setdefault(p, []).append(r["PF"])
    r3_verdict = []
    for p, vals in r3_by_param.items():
        vals = [v for v in vals if np.isfinite(v)]
        med = float(np.median(vals)) if vals else np.nan
        mn = float(np.min(vals)) if vals else np.nan
        ok = (med >= 1.05) and (mn >= 0.6 * final_row["PF"])
        r3_verdict.append({"param": p, "median_neighbor_PF": med, "min_neighbor_PF": mn,
                           "verdict": "PASS" if ok else "FAIL"})
    r3v = pd.DataFrame(r3_verdict)
    r3_pass = bool(len(r3v) and (r3v["verdict"] == "PASS").all())
    # R4
    r4_pass = final_row["n"] >= 400

    print_frame("R1 stationary block bootstrap (2000x, weekly blocks)",
                pd.DataFrame([{"PF_point": final_row["PF"], "CI90_lower": r1_lower,
                               "PASS|lower>1.05": r1_pass}]))
    print_frame("R2 sub-period PF", r2)
    print(f"  R2 PASS={r2_pass}  ({r2_pos}/4 periods >1.0; min={r2_min:.3f})")
    print_frame("R3 +/-1-step neighborhood", r3)
    print_frame("R3 verdict per param", r3v)
    print(f"  R4 n>=400: {'PASS' if r4_pass else 'FAIL'} (n={final_row['n']})\n")

    # ---------------- Stage E1 ----------------
    e1_res = run_cfg(final_cfg, cost=COST3)
    e1 = mrow(e1_res, "final_3ticks", cost=COST3)
    print_frame("E1 final candidate @3 ticks", pd.DataFrame([final_row, e1]))
    thin = e1["PF"] < 1.05
    print(f"  E1 edge-too-thin-for-retail = {thin} (PF@3t={e1['PF']:.4f})\n")

    all_pass = bool(r1_pass and r2_pass and r3_pass and r4_pass)

    save(pd.DataFrame([final_row]), "m3_final_metrics.csv")
    save(pd.DataFrame([{**final_row, "ticks": 1.5}, {**e1, "ticks": 3.0}]),
         "m3_final_metrics_1p5_3.csv")
    robust = pd.concat([
        pd.DataFrame([{"check": "R1_bootstrap_CI90_lower", "value": r1_lower, "passed": r1_pass}]),
        r2.assign(check="R2_" + r2["period"]).rename(columns={"PF": "value"}).assign(
            passed=lambda d: (d["value"] > 1.0)),
        r3v.rename(columns={"median_neighbor_PF": "value"}).assign(
            check="R3_" + r3v["param"], passed=lambda d: d["verdict"] == "PASS"),
        pd.DataFrame([{"check": "R4_n>=400", "value": final_row["n"], "passed": r4_pass}]),
    ], ignore_index=True)
    save(robust, "m3_robustness.csv")

    # ---------------- freeze ----------------
    if all_pass:
        seal = os.path.join(CONFIG_DIR, "t1_sealed_15min.yaml")
        with open(seal, "w", encoding="utf-8") as fh:
            fh.write("# T1' sealed config v2 - 15min decisions, IN-SAMPLE ONLY\n")
            fh.write("# Cost model corrected 2026-10-10: base ('1.5t' label retained) = commission $0.50/side + slippage $0.20/side ($1.40 round trip); stress ('3.0t') = $0.50/side + $1.50/side slippage.\n")
            fh.write(f"sealed_is_start: '{IS_START}'\nsealed_is_end: '{IS_END}'\n")
            fh.write("freq: 15min\nentry_mode: %s\n" % final_cfg.get("entry_mode", "market"))
            fh.write("trigger: donchian_breakout\n")
            fh.write(f"donch: {final_cfg.get('donch', 20)}\n")
            fh.write("vetoes:\n  - 'VR12 < 1.0'\n  - 'ATR14_percentile(500) <= 90'\n")
            if final_cfg.get("veto_extra") is not None:
                for _, r in G.get("bad_buckets", pd.DataFrame()).iterrows():
                    if r["mask_kind"] == "eq":
                        fh.write(f"  - '{r['feature']} == {int(r['val'])}'\n")
                    else:
                        fh.write(f"  - '{r['feature']} in ({r['lo']}, {r['hi']}]'\n")
            fh.write("exit_engine: %s\n" % final_cfg.get("exit_engine", "atr"))
            fh.write(f"stop_mult: {final_cfg.get('stop_mult', 3.0)}\n")
            if final_cfg.get("breakeven_atr") is not None:
                fh.write(f"breakeven_atr: {final_cfg['breakeven_atr']}\n")
            if final_cfg.get("time_stop_bars") is not None:
                fh.write(f"time_stop_bars: {final_cfg['time_stop_bars']}\n")
            fh.write("is_metrics@1.5t:\n")
            for k in ("PF", "win", "MAR", "MaxDD", "CAGR", "R2", "pct_pos12m", "n"):
                fh.write(f"  {k}: {final_row[k]}\n")
            fh.write("stageE1_metrics@3t:\n")
            for k in ("PF", "win", "MAR", "MaxDD", "CAGR"):
                fh.write(f"  {k}: {e1[k]}\n")
            fh.write("robustness:\n")
            fh.write(f"  R1_lower: {r1_lower}\n  R1_pass: {r1_pass}\n")
            fh.write(f"  R2_pass: {r2_pass}\n  R3_pass: {r3_pass}\n  R4_pass: {r4_pass}\n")
            fh.write(f"  E1_thin: {thin}\n")
        print(f"FROZEN -> {seal}")
    else:
        print("NOT FROZEN (robustness failed) - existing seal left untouched")

    print("=" * 78)
    print("M3 ACCEPTANCE")
    print("=" * 78)
    pf = final_row["PF"]
    print(f"  final PF @1.5t = {pf:.4f}   win={final_row['win']:.4f} n={final_row['n']} "
          f"MAR={final_row['MAR']:.4f} MaxDD={final_row['MaxDD']:.4f} CAGR={final_row['CAGR']:.4f}")
    print(f"  R2(log eq)={final_row['R2']:.4f}  %pos-12m={final_row['pct_pos12m']:.4f}  "
          f"Sharpe={final_row['Sharpe']:.4f}")
    print(f"  PF>=1.35 (recommended): {'PASS' if pf >= 1.35 else 'FAIL'}")
    print(f"  PF>=1.50 (original)   : {'PASS' if pf >= 1.50 else 'FAIL'}")
    print(f"  PF@3t={e1['PF']:.4f}  thin={thin}  frozen={all_pass}")
    print(f"TOTAL RUNTIME {time.perf_counter() - t_start:.1f}s")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
