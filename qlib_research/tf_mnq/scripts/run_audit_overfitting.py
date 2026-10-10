"""HONEST overfitting audit of the frozen 15min config (t1_sealed_15min.yaml).

Frozen config = run_breakout_ex(donch=20, stop_mult=2.5, entry market,
veto_long = base{VR>=1 | ATRpct>90} OR narrow Stage-B bands) on 15min bars.

Tasks
  1  USD accounting layer ($50k, MNQ=$2/pt, 1 & 2 contracts, MaxDD_usd<=1500)
  2  One pre-registered OOS touch (2024-01-01 .. 2026-10-02), no tuning
  3  Narrow-band veto ablation (full / base / smoothed) IS+OOS
  4  Leave-one-year-out re-derivation of the vetoes
  5  Random-entry placebo Monte-Carlo (500 variants)
  6  Multiple-testing / deflated Sharpe

All new CSVs are written to tf_mnq/results/audit_*.csv (plus usd_metrics.csv).
This script never tunes on OOS: every OOS number applies an IS-frozen rule.
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

from scipy import stats as st  # noqa: E402

from tf_mnq.breakout import run_breakout_ex  # noqa: E402
from tf_mnq.data import load_bars  # noqa: E402
from tf_mnq.engine import CostCfg, metrics  # noqa: E402
from tf_mnq.features import compute_features  # noqa: E402
from tf_mnq.analytics import curve_quality, returns_metrics  # noqa: E402
from tf_mnq.strategies import strategy_frame, veto_blocked  # noqa: E402

RESULTS_DIR = os.path.join(_PKG_ROOT, "results")
IS_START, IS_END = "2019-05-06", "2023-12-31"
OOS_START, OOS_END = "2024-01-01", "2026-10-02"
FREQ = "15min"
COST = CostCfg(slip_ticks=1.5, commission_per_side=0.75, tick=0.25, point_value=2.0)
PV = 2.0          # $/point/contract
ACCOUNT = 50000.0
DD_CAP = 1500.0    # hard user constraint
DONCH, STOP = 20, 2.5
BASE_COMPS = ["vol", "vr"]
N_PLACEBO = 500
N_TRIALS = 70      # ~48 grid + ~22 Stage variants (stated, approximate)


def save(df: pd.DataFrame, name: str) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, name), index=False)


def pf_of(pnl) -> float:
    pnl = np.asarray(pnl, float)
    w = pnl[pnl > 0].sum()
    l = -pnl[pnl < 0].sum()
    return float(w / l) if l > 0 else (np.inf if w > 0 else 0.0)


def print_frame(title, df, floatfmt="%.4f"):
    print(title)
    if df is None or len(df) == 0:
        print("  (empty)")
    else:
        with pd.option_context("display.width", 260, "display.max_columns", 80):
            print(df.to_string(index=False, float_format=lambda x: floatfmt % x))
    print()


# --------------------------------------------------------------------------- #
# feature / veto construction
# --------------------------------------------------------------------------- #
def make_fseries(bars, feat, sf) -> dict:
    return {
        "VR": feat["vr"],
        "ATRpct": sf["atr_pct"],
        "ER20": feat["er"],
        "extension": sf["ext"],
        "pullback_depth": (bars["high"].rolling(10, min_periods=10).max() - bars["low"]) / feat["atr"],
        "hour": pd.Series(bars.index.hour, index=bars.index),
        "dow": pd.Series(bars.index.dayofweek, index=bars.index),
    }


def mask_from_bad(bad: pd.DataFrame, fseries: dict, index) -> pd.Series:
    mask = pd.Series(False, index=index)
    for _, r in bad.iterrows():
        fs = fseries[r["feature"]]
        if r["mask_kind"] == "eq":
            mask = mask | (fs == r["val"])
        else:
            mask = mask | ((fs > r["lo"]) & (fs <= r["hi"]))
    return mask.fillna(False)


def frozen_bad() -> pd.DataFrame:
    """The 4 Stage-B bad buckets that became the sealed narrow-band vetoes."""
    return pd.read_csv(os.path.join(RESULTS_DIR, "stageB_added_vetoes.csv"))


def smoothed_mask(fseries: dict, index) -> pd.Series:
    """Coarse, non-narrow replacements of the 4 Stage-B bands (fixed a priori)."""
    atrp = fseries["ATRpct"]
    ext = fseries["extension"]
    pb = fseries["pullback_depth"]
    hour = fseries["hour"]
    m = (atrp <= 27.48) | (ext > 2.268) | (pb <= 1.456) | hour.isin([2, 3, 4])
    return m.fillna(False)


# --------------------------------------------------------------------------- #
# USD accounting layer
# --------------------------------------------------------------------------- #
def usd_metrics(eq_norm, trades, idx, ppy, contracts, capital, label, kind) -> dict:
    """equity_usd = 50000 + (normalised equity - 1)*capital*contracts."""
    eq = np.asarray(eq_norm, float)
    cap = float(capital)
    eq_usd = ACCOUNT + (eq - 1.0) * cap * contracts
    s_usd = pd.Series(eq_usd, index=idx)

    years = (idx[-1] - idx[0]).days / 365.25
    end = float(eq_usd[-1])
    cagr = (end / ACCOUNT) ** (1.0 / years) - 1.0 if (end > 0 and years > 0) else np.nan

    peak = np.maximum.accumulate(eq_usd)
    maxdd_usd = float((peak - eq_usd).max())
    maxdd_pct_50k = maxdd_usd / ACCOUNT

    # points-based MaxDD (unit cross-check; independent of contract count)
    pts = (eq - 1.0) * cap / PV
    peak_p = np.maximum.accumulate(pts)
    maxdd_pts = float((peak_p - pts).max())

    ret = np.diff(eq_usd) / eq_usd[:-1] if len(eq_usd) > 1 else np.array([])
    std = ret.std(ddof=1) if ret.size > 1 else np.nan
    sharpe = ret.mean() / std * np.sqrt(ppy) if (np.isfinite(std) and std > 0) else np.nan
    dn = ret[ret < 0]
    dstd = dn.std(ddof=1) if dn.size > 1 else np.nan
    sortino = ret.mean() / dstd * np.sqrt(ppy) if (np.isfinite(dstd) and dstd > 0) else np.nan

    W = int(round(ppy))
    if len(s_usd) > W:
        worst12_usd = float((s_usd - s_usd.shift(W)).min())
        worst12_ret = float((s_usd / s_usd.shift(W) - 1.0).min())
    else:
        worst12_usd = worst12_ret = np.nan

    tr = pd.DataFrame(trades) if trades else pd.DataFrame(columns=["pnl_pts", "pnl_usd_net"])
    if len(tr):
        pnl_usd = tr["pnl_pts"].to_numpy(float) * PV * contracts      # spec formula
        pnl_usd_net = tr["pnl_usd_net"].to_numpy(float) * contracts   # incl. commission
        n_tr = len(tr)
        wins = pnl_usd_net[pnl_usd_net > 0]
        losses = pnl_usd_net[pnl_usd_net < 0]
        pf = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else np.inf
        winr = float((pnl_usd_net > 0).mean())
        exp_usd = float(pnl_usd_net.mean())
        total_usd = float(pnl_usd_net.sum())
    else:
        n_tr, pf, winr, exp_usd, total_usd = 0, np.nan, np.nan, np.nan, 0.0

    return {
        "label": label, "kind": kind, "contracts": contracts,
        "n_trades": n_tr, "PF": pf, "win": winr,
        "equity_end_usd": end, "total_pnl_usd": total_usd, "expectancy_usd": exp_usd,
        "CAGR_usd": cagr, "Sharpe_usd": sharpe, "Sortino_usd": sortino,
        "MaxDD_usd": maxdd_usd, "MaxDD_pct_50k": maxdd_pct_50k,
        "MaxDD_points": maxdd_pts,
        "MaxDD_usd_from_points": maxdd_pts * PV * contracts,
        "worst_12m_usd": worst12_usd, "worst_12m_ret": worst12_ret,
        "dd_pass_le_1500": bool(maxdd_usd <= DD_CAP),
    }


def monthly_usd(eq_norm, idx, capital, contracts) -> pd.Series:
    eq_usd = ACCOUNT + (np.asarray(eq_norm, float) - 1.0) * capital * contracts
    s = pd.Series(eq_usd, index=idx)
    return s.resample("ME").last().pct_change().dropna()


# --------------------------------------------------------------------------- #
# normalised metrics (same definitions as run_m3_stages)
# --------------------------------------------------------------------------- #
def norm_metrics(res, ppy) -> dict:
    eq = res["equity"]
    idx = eq.index
    m = metrics(eq, res["trades"], idx, COST, ppy)
    cq = curve_quality(eq, ppy)
    rm = returns_metrics(eq, ppy, idx)
    return {
        "PF": m["profit_factor"], "win": m["hit_rate"], "n": m["n_trades"],
        "MAR": m["MAR"], "MaxDD_norm": m["MaxDD"], "CAGR": m["CAGR"],
        "Sharpe": rm["Sharpe"], "Sortino": rm["Sortino"],
        "R2": cq["r2_log_equity_vs_time"],
        "pct_pos12m": cq["pct_positive_rolling_12m"],
        "worst_12m_ret": cq["worst_12m_return"],
        "TIM": m["time_in_market"], "expectancy_R": m["expectancy_R"],
        "expectancy_pts": float(np.mean([t["pnl_pts"] for t in res["trades"]])) if res["trades"] else np.nan,
        "expectancy_usd": float(np.mean([t["pnl_usd_net"] for t in res["trades"]])) if res["trades"] else np.nan,
        "total_return": rm["total_return"], "capital": res["capital"],
        "n_breakouts": res.get("n_breakouts", np.nan),
    }


def run_cfg(bars, feat, atr, blocked, stop_mult=STOP, donch=DONCH):
    return run_breakout_ex(bars, atr, COST, donch=donch, stop_mult=stop_mult,
                           veto_long=blocked, entry_mode="market")


# --------------------------------------------------------------------------- #
def main() -> int:
    t_start = time.perf_counter()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---------------- load full series once; features are continuous --------
    allb = load_bars(FREQ)
    feat_all = compute_features(allb)
    sf_all = strategy_frame(allb, feat_all)
    print(f"FULL {FREQ}: {len(allb)} bars [{allb.index.min()} .. {allb.index.max()}]")

    def sl(lo, hi):
        b = allb.loc[(allb.index >= pd.Timestamp(lo)) & (allb.index <= pd.Timestamp(hi))].copy()
        return b, feat_all.reindex(b.index), sf_all.reindex(b.index)

    # ================= IS ==================================================
    bars, feat, sf = sl(IS_START, IS_END)
    atr = feat["atr"]
    fseries = make_fseries(bars, feat, sf)
    ppy = len(bars) / ((bars.index[-1] - bars.index[0]).days / 365.25)
    base_blocked = veto_blocked(bars, feat, sf, components=BASE_COMPS)
    bad = frozen_bad()
    narrow_mask = mask_from_bad(bad, fseries, bars.index)
    full_blocked = (base_blocked | narrow_mask)
    smooth_blocked = (base_blocked | smoothed_mask(fseries, bars.index))

    res_full = run_cfg(bars, feat, atr, full_blocked)
    nf = norm_metrics(res_full, ppy)
    cap_is = nf["capital"]
    print(f"IS bars={len(bars)} ppy={ppy:.1f} capital=${cap_is:.0f} "
          f"n_signals(full)={int((((bars['close'] > bars['high'].rolling(DONCH, min_periods=DONCH).max().shift(1)) & np.isfinite(atr)) & (~full_blocked)).sum())} "
          f"n_breakouts={int(nf['n_breakouts'])} n_trades={nf['n']}")
    print(f"IS full frozen: PF={nf['PF']:.4f} win={nf['win']:.4f} n={nf['n']} "
          f"MAR={nf['MAR']:.4f} MaxDD_norm={nf['MaxDD_norm']:.4f} Sharpe={nf['Sharpe']:.4f} "
          f"R2={nf['R2']:.4f} %pos12m={nf['pct_pos12m']:.4f}")

    # ================= TASK 1: USD accounting ==============================
    print("=" * 78); print("TASK 1 - USD ACCOUNTING LAYER (frozen config, IS @1.5t)"); print("=" * 78)
    usd_rows = []
    for c in (1, 2):
        r = usd_metrics(res_full["equity"], res_full["trades"], bars.index, ppy, c,
                        cap_is, "IS_frozen", "IS")
        usd_rows.append(r)
    usd_is = pd.DataFrame(usd_rows)
    show1 = ["label", "contracts", "n_trades", "PF", "win", "total_pnl_usd", "expectancy_usd",
             "CAGR_usd", "Sharpe_usd", "Sortino_usd", "MaxDD_usd", "MaxDD_points",
             "MaxDD_usd_from_points", "worst_12m_usd", "dd_pass_le_1500"]
    print_frame("IS USD metrics", usd_is[show1], "%.4f")
    # unit cross-check
    chk = usd_is.iloc[0]
    ok = abs(chk["MaxDD_usd"] - chk["MaxDD_usd_from_points"]) < 1.0
    print(f"  unit check MaxDD_points*2*contracts == MaxDD_usd : {'OK' if ok else 'MISMATCH'} "
          f"({chk['MaxDD_usd']:.2f} vs {chk['MaxDD_usd_from_points']:.2f})")
    print(f"  MaxDD_usd @1c = ${chk['MaxDD_usd']:.2f}  -> <= $1500? "
          f"{'PASS' if chk['dd_pass_le_1500'] else 'FAIL'}")
    print(f"  MaxDD_usd @2c = ${usd_is.iloc[1]['MaxDD_usd']:.2f}  -> <= $1500? "
          f"{'PASS' if usd_is.iloc[1]['dd_pass_le_1500'] else 'FAIL'}\n")

    m_is_1 = monthly_usd(res_full["equity"], bars.index, cap_is, 1)
    m_is_2 = monthly_usd(res_full["equity"], bars.index, cap_is, 2)
    mdf = pd.DataFrame({"IS_1c": m_is_1, "IS_2c": m_is_2})
    mdf.index.name = "month"

    # ================= OOS (single sanctioned touch) ======================
    print("=" * 78); print("TASK 2 - OOS VALIDATION (2024-01-01..2026-10-02, frozen, no tuning)"); print("=" * 78)
    obars, ofeat, osf = sl(OOS_START, OOS_END)
    oatr = ofeat["atr"]
    ofseries = make_fseries(obars, ofeat, osf)
    oppy = len(obars) / ((obars.index[-1] - obars.index[0]).days / 365.25)
    obase = veto_blocked(obars, ofeat, osf, components=BASE_COMPS)
    ofull = (obase | mask_from_bad(bad, ofseries, obars.index))
    osmooth = (obase | smoothed_mask(ofseries, obars.index))
    res_oos = run_cfg(obars, ofeat, oatr, ofull)
    no = norm_metrics(res_oos, oppy)
    cap_oos = no["capital"]
    print(f"OOS bars={len(obars)} ppy={oppy:.1f} capital=${cap_oos:.0f} "
          f"n_breakouts={int(no['n_breakouts'])} n_trades={no['n']}")
    print(f"OOS full frozen: PF={no['PF']:.4f} win={no['win']:.4f} n={no['n']} "
          f"MAR={no['MAR']:.4f} MaxDD_norm={no['MaxDD_norm']:.4f} CAGR={no['CAGR']:.4f} "
          f"Sharpe={no['Sharpe']:.4f} R2={no['R2']:.4f} %pos12m={no['pct_pos12m']:.4f} "
          f"TIM={no['TIM']:.4f}")
    oos_usd_rows = []
    for c in (1, 2):
        oos_usd_rows.append(usd_metrics(res_oos["equity"], res_oos["trades"], obars.index,
                                        oppy, c, cap_oos, "OOS_frozen", "OOS"))
    usd_oos = pd.DataFrame(oos_usd_rows)
    print_frame("OOS USD metrics", usd_oos[show1], "%.4f")

    # sub-period PFs
    sub_rows = []
    for y in (2024, 2025, 2026):
        pnl = [t["pnl_usd_net"] for t in res_oos["trades"]
               if pd.Timestamp(t["entry_time"]).year == y]
        sub_rows.append({"period": y, "n": len(pnl), "PF": pf_of(pnl)})
    sub = pd.DataFrame(sub_rows)
    print_frame("OOS sub-period PF", sub)

    # pre-registered pass bar (chosen size = 1 contract, the min-DD size)
    c1 = usd_oos.iloc[0]
    bar_pf = bool(no["PF"] >= 1.15)
    bar_exp = bool(no["expectancy_usd"] > 0)
    bar_dd = bool(c1["dd_pass_le_1500"])
    oos_pass = bool(bar_pf and bar_exp and bar_dd)
    print(f"  PASS BAR: PF>=1.15 {'PASS' if bar_pf else 'FAIL'} (PF={no['PF']:.4f}); "
          f"expectancy>0 {'PASS' if bar_exp else 'FAIL'} (${no['expectancy_usd']:.2f}/trade); "
          f"MaxDD_usd<=1500 @1c {'PASS' if bar_dd else 'FAIL'} (${c1['MaxDD_usd']:.2f})")
    print(f"  OOS OVERALL: {'PASS' if oos_pass else 'FAIL'}\n")

    m_oos_1 = monthly_usd(res_oos["equity"], obars.index, cap_oos, 1)
    m_oos_2 = monthly_usd(res_oos["equity"], obars.index, cap_oos, 2)
    mdf = mdf.join(pd.DataFrame({"OOS_1c": m_oos_1, "OOS_2c": m_oos_2}), how="outer")
    mdf.to_csv(os.path.join(RESULTS_DIR, "audit_usd_monthly.csv"))

    # ================= TASK 3: ablation ===================================
    print("=" * 78); print("TASK 3 - NARROW-BAND VETO ABLATION (IS + OOS)"); print("=" * 78)
    abl = []
    for tag, fb, ob in (("full_frozen", full_blocked, ofull),
                        ("no_narrow(base)", base_blocked, obase),
                        ("smoothed_coarse", smooth_blocked, osmooth)):
        ri = run_cfg(bars, feat, atr, fb)
        ro = run_cfg(obars, ofeat, oatr, ob)
        mi, mo = norm_metrics(ri, ppy), norm_metrics(ro, oppy)
        ui = usd_metrics(ri["equity"], ri["trades"], bars.index, ppy, 1, mi["capital"], tag, "IS")
        uo = usd_metrics(ro["equity"], ro["trades"], obars.index, oppy, 1, mo["capital"], tag, "OOS")
        abl.append({"config": tag,
                    "IS_PF": mi["PF"], "IS_win": mi["win"], "IS_n": mi["n"], "IS_MAR": mi["MAR"],
                    "OOS_PF": mo["PF"], "OOS_win": mo["win"], "OOS_n": mo["n"], "OOS_MAR": mo["MAR"],
                    "IS_n_breakouts": int(mi["n_breakouts"]), "OOS_n_breakouts": int(mo["n_breakouts"]),
                    "IS_MaxDD_usd_1c": ui["MaxDD_usd"], "OOS_MaxDD_usd_1c": uo["MaxDD_usd"],
                    "IS_dd_pass": ui["dd_pass_le_1500"], "OOS_dd_pass": uo["dd_pass_le_1500"]})
    abl = pd.DataFrame(abl)
    print_frame("Ablation", abl, "%.4f")
    d_is = abl.loc[0, "IS_PF"] - abl.loc[1, "IS_PF"]
    d_oos = abl.loc[0, "OOS_PF"] - abl.loc[1, "OOS_PF"]
    retained = d_oos / d_is if abs(d_is) > 1e-9 else np.nan
    print(f"  narrow-band PF contribution: IS {d_is:+.4f}   OOS {d_oos:+.4f}   "
          f"(retained {retained*100:.0f}% of the IS effect)\n")

    # ================= TASK 4: leave-one-year-out =========================
    print("=" * 78); print("TASK 4 - LEAVE-ONE-YEAR-OUT VETO RE-DERIVATION (IS)"); print("=" * 78)
    base_res_is = run_cfg(bars, feat, atr, base_blocked)  # no narrow band
    loyo = []
    base_year_pf = {}
    for y in (2020, 2021, 2022, 2023):
        pnl_base = [t["pnl_usd_net"] for t in base_res_is["trades"]
                    if pd.Timestamp(t["entry_time"]).year == y]
        pfy0 = pf_of(pnl_base)
        base_year_pf[y] = pfy0
        other = [t for t in base_res_is["trades"]
                 if pd.Timestamp(t["entry_time"]).year != y]
        if other:
            vals = {k: fs.to_numpy(float)[[t["signal_bar"] for t in other]]
                    for k, fs in fseries.items()}
            pnl_other = np.array([t["pnl_usd_net"] for t in other], float)
            rows = []
            for fname, vv in vals.items():
                if fname in ("hour", "dow"):
                    for u in np.unique(vv):
                        idxm = np.where(vv == u)[0]
                        rows.append({"feature": fname, "mask_kind": "eq", "val": float(u),
                                     "lo": np.nan, "hi": np.nan, "n": len(idxm),
                                     "PF": pf_of(pnl_other[idxm])})
                else:
                    qq, cats = pd.qcut(pd.Series(vv), 10, labels=False, duplicates="drop", retbins=True)
                    qq = np.asarray(qq)
                    finite = qq[~pd.isna(qq)]
                    edges = cats
                    for d in sorted(pd.unique(finite)):
                        d = int(d)
                        idxm = np.where(qq == d)[0]
                        lo = float(edges[d]); hi = float(edges[d + 1])
                        rows.append({"feature": fname, "mask_kind": "range", "val": np.nan,
                                     "lo": lo, "hi": hi, "n": len(idxm),
                                     "PF": pf_of(pnl_other[idxm])})
            allbuckets = pd.DataFrame(rows)
            badb = allbuckets[(allbuckets["PF"] < 0.85) & (allbuckets["n"] >= 80)]
        else:
            badb = pd.DataFrame(columns=["feature", "mask_kind", "val", "lo", "hi", "n", "PF"])
        if len(badb):
            ymask = mask_from_bad(badb, fseries, bars.index)
            yres = run_cfg(bars, feat, atr, base_blocked | ymask)
            pnl_y = [t["pnl_usd_net"] for t in yres["trades"]
                     if pd.Timestamp(t["entry_time"]).year == y]
            pfy1 = pf_of(pnl_y)
        else:
            pfy1 = np.nan
        loyo.append({"year": y, "n_bad_buckets_reidentified": len(badb),
                     "year_PF_without_veto": pfy0, "year_PF_with_rederived_veto": pfy1,
                     "edge": (pfy1 - pfy0) if np.isfinite(pfy1) else np.nan,
                     "bad_features": ",".join(badb["feature"].unique()) if len(badb) else ""})
    loyo = pd.DataFrame(loyo)
    print_frame("LOYO", loyo, "%.4f")
    if loyo["edge"].notna().any():
        print(f"  LOYO edge signs: {loyo['edge'].round(3).tolist()}")
    print("  NOTE: full-frozen year-PF reference available via audit_loyo (base = no narrow band).\n")

    # ================= TASK 5: random-entry placebo =======================
    print("=" * 78); print(f"TASK 5 - RANDOM-ENTRY PLACEBO ({N_PLACEBO} variants)"); print("=" * 78)
    don_hi = bars["high"].rolling(DONCH, min_periods=DONCH).max().shift(1)
    cand = (bars["close"] > don_hi) & np.isfinite(atr)
    cand_idx = np.flatnonzero(cand.to_numpy())
    n_allow = int(nf["n_breakouts"])
    real_pf = nf["PF"]
    real_ret = nf["total_return"]
    real_ntr = int(nf["n"])
    real_held = np.array([t["bars_held"] for t in res_full["trades"]], float)
    rng = np.random.default_rng(20261009)
    idx_arr = bars.index

    def placebo_run(draw):
        pick = rng.choice(cand_idx, size=min(draw, cand_idx.size), replace=False)
        blk = np.ones(len(bars), dtype=bool)
        blk[pick] = False
        r = run_breakout_ex(bars, atr, COST, donch=DONCH, stop_mult=STOP,
                            veto_long=pd.Series(blk, index=idx_arr), entry_mode="market")
        return r

    # calibrate the draw count so the placebo executes ~the same number of trades
    scale, draw = 1.0, n_allow
    for _ in range(4):
        execs = [len(placebo_run(int(round(scale * n_allow)))["trades"]) for _ in range(40)]
        med = float(np.median(execs))
        if med <= 0:
            break
        scale = min(8.0, scale * real_ntr / med)
        draw = min(int(round(scale * n_allow)), cand_idx.size)
        if abs(med - real_ntr) / real_ntr < 0.05:
            break
    print(f"  calibrated placebo draw={draw} (scale={scale:.2f}) to match real n_trades={real_ntr}")

    pfs = np.empty(N_PLACEBO)
    rets = np.empty(N_PLACEBO)
    ntrades = np.empty(N_PLACEBO, dtype=int)
    held_med = np.full(N_PLACEBO, np.nan)
    t0 = time.perf_counter()
    for i in range(N_PLACEBO):
        r = placebo_run(draw)
        p = np.array([t["pnl_usd_net"] for t in r["trades"]], float) if r["trades"] else np.array([0.0])
        pfs[i] = pf_of(p)
        rets[i] = float(r["equity"].iloc[-1] - 1.0)
        ntrades[i] = len(r["trades"])
        if r["trades"]:
            held_med[i] = float(np.median([t["bars_held"] for t in r["trades"]]))
        if (i + 1) % 100 == 0:
            print(f"  placebo {i+1}/{N_PLACEBO}  ({time.perf_counter()-t0:.0f}s)")
    finite = np.isfinite(pfs)
    pf_pct = float((pfs[finite] < real_pf).mean())
    ret_pct = float((rets < real_ret).mean())
    plac = pd.DataFrame({"PF": pfs, "total_return": rets, "n_trades": ntrades,
                         "median_bars_held": held_med})
    plac.to_csv(os.path.join(RESULTS_DIR, "audit_placebo.csv"), index=False)
    summ = pd.DataFrame([{
        "real_PF": real_pf, "real_total_return": real_ret,
        "real_n_trades": real_ntr,
        "real_median_bars_held": float(np.median(real_held)),
        "placebo_draw": int(draw),
        "placebo_PF_mean": float(pfs[finite].mean()),
        "placebo_PF_p50": float(np.median(pfs[finite])),
        "placebo_PF_p95": float(np.percentile(pfs[finite], 95)),
        "placebo_PF_p99": float(np.percentile(pfs[finite], 99)),
        "placebo_PF_max": float(pfs[finite].max()),
        "real_PF_percentile": pf_pct, "real_in_top5pct": bool(pf_pct >= 0.95),
        "real_return_percentile": ret_pct,
        "placebo_ntrades_p50": float(np.median(ntrades)),
        "placebo_median_bars_held_p50": float(np.nanmedian(held_med)),
        "n_placebo": int(N_PLACEBO)}])
    print_frame("Placebo summary", summ, "%.4f")
    print(f"  real PF={real_pf:.4f} at percentile {pf_pct*100:.1f}% of placebo distribution "
          f"-> {'DISTINGUISHABLE (top5%)' if pf_pct >= 0.95 else 'NOT distinguishable from luck'}\n")

    # ================= TASK 6: multiple testing ==========================
    print("=" * 78); print("TASK 6 - MULTIPLE-TESTING / DEFLATED SHARPE"); print("=" * 78)
    eqv = res_full["equity"].to_numpy(float)
    ret = np.diff(eqv) / eqv[:-1]
    T = len(ret)
    sr_pb = ret.mean() / ret.std(ddof=1)
    g3 = float(st.skew(ret))
    g4 = float(st.kurtosis(ret, fisher=False))  # Pearson (normal=3)

    # same-family trial dispersion (base vetoes; donch x stop x atr-cap)
    trial_sr_ann = []
    for d in (16, 20, 24, 32):
        for sm in (1.5, 2.0, 2.5, 3.0):
            for cap in (85.0, 90.0, 95.0):
                blkm = veto_blocked(bars, feat, sf, components=BASE_COMPS, atr_pct_cap=cap)
                rr = run_breakout_ex(bars, atr, COST, donch=d, stop_mult=sm,
                                     veto_long=blkm, entry_mode="market")
                e = rr["equity"].to_numpy(float)
                rt = np.diff(e) / e[:-1]
                if rt.std(ddof=1) > 0:
                    trial_sr_ann.append(rt.mean() / rt.std(ddof=1) * np.sqrt(ppy))
    trial_sr_ann = np.array(trial_sr_ann)
    trial_sr_pb = trial_sr_ann / np.sqrt(ppy)
    V = float(np.var(trial_sr_pb, ddof=1))
    V_pos = float(np.var(trial_sr_pb[trial_sr_pb > 0], ddof=1)) if (trial_sr_pb > 0).sum() > 1 else V
    gamma = 0.5772156649015329

    def dsr_at(V_, N_):
        emax_ = np.sqrt(V_) * ((1 - gamma) * st.norm.ppf(1 - 1.0 / N_)
                               + gamma * st.norm.ppf(1 - 1.0 / (N_ * np.e)))
        zz = (sr_pb - emax_) * np.sqrt(T - 1) / denom
        return emax_ * np.sqrt(ppy), float(st.norm.cdf(zz))

    denom = np.sqrt(max(1e-12, 1 - g3 * sr_pb + ((g4 - 1) / 4) * sr_pb ** 2))
    sr0_ann, dsr = dsr_at(V, N_TRIALS)
    _, dsr_pos = dsr_at(V_pos, N_TRIALS)
    _, dsr_n48 = dsr_at(V, 48)
    p_dsr = float(1 - dsr)

    # single-test block bootstrap p(PF<=1) on frozen IS trades
    pnl = np.array([t["pnl_usd_net"] for t in res_full["trades"]], float)
    et = pd.Series([pd.Timestamp(t["entry_time"]) for t in res_full["trades"]])
    wk = et.dt.to_period("W")
    groups = [pnl[wk == w] for w in wk.unique()]
    boot = np.empty(2000)
    rb = np.random.default_rng(7)
    for i in range(2000):
        pick = rb.integers(0, len(groups), len(groups))
        boot[i] = pf_of(np.concatenate([groups[j] for j in pick]))
    boot = boot[np.isfinite(boot)]
    p_boot = float((boot <= 1.0).mean())
    ci5 = float(np.percentile(boot, 5))
    p_sidak = float(1 - (1 - p_boot) ** N_TRIALS)
    p_bonf = float(min(1.0, p_boot * N_TRIALS))

    mt = pd.DataFrame([
        {"stat": "IS_Sharpe_ann", "value": sr_pb * np.sqrt(ppy)},
        {"stat": "skew", "value": g3}, {"stat": "kurtosis_pearson", "value": g4},
        {"stat": "T_returns", "value": T},
        {"stat": "n_trials_N", "value": N_TRIALS},
        {"stat": "family_trial_sharpe_ann_std", "value": float(trial_sr_ann.std(ddof=1))},
        {"stat": "family_trial_sharpe_ann_min", "value": float(trial_sr_ann.min())},
        {"stat": "family_trial_sharpe_ann_max", "value": float(trial_sr_ann.max())},
        {"stat": "expected_max_Sharpe_null_ann", "value": sr0_ann},
        {"stat": "DSR", "value": dsr}, {"stat": "p_deflated", "value": p_dsr},
        {"stat": "DSR_positive_trials_only", "value": dsr_pos},
        {"stat": "DSR_N48", "value": dsr_n48},
        {"stat": "PF_point", "value": real_pf},
        {"stat": "bootstrap_p_PF_le_1", "value": p_boot},
        {"stat": "bootstrap_PF_CI5", "value": ci5},
        {"stat": "sidak_p", "value": p_sidak}, {"stat": "bonferroni_p", "value": p_bonf},
    ])
    print_frame("Multiple-testing", mt, "%.6f")
    print(f"  Expected max Sharpe under null over {N_TRIALS} trials = {sr0_ann:.3f} (ann); "
          f"observed = {sr_pb*np.sqrt(ppy):.3f}")
    print(f"  Deflated Sharpe = {dsr:.4f} (p={p_dsr:.4f}); single-test bootstrap p(PF<=1)={p_boot:.4f}\n")

    # ================= save all ==========================================
    usd_all = pd.concat([usd_is, usd_oos], ignore_index=True)
    save(usd_all, "audit_usd_metrics.csv")
    save(usd_all, "usd_metrics.csv")
    oos_row = {**{f"OOS_{k}": v for k, v in no.items()},
               **{f"OOS_USD_1c_{k}": v for k, v in usd_oos.iloc[0].items()},
               **{f"OOS_USD_2c_{k}": v for k, v in usd_oos.iloc[1].items()},
               "OOS_PF_2024": sub.loc[sub.period == 2024, "PF"].iloc[0],
               "OOS_PF_2025": sub.loc[sub.period == 2025, "PF"].iloc[0],
               "OOS_PF_2026YTD": sub.loc[sub.period == 2026, "PF"].iloc[0],
               "OOS_PASS": oos_pass}
    save(pd.DataFrame([oos_row]), "audit_oos_metrics.csv")
    save(abl, "audit_ablation.csv")
    save(loyo, "audit_loyo.csv")
    save(summ, "audit_placebo_summary.csv")
    save(mt, "audit_multiple_testing.csv")
    save(pd.DataFrame(res_full["trades"]), "audit_trades_is_full.csv")
    save(pd.DataFrame(res_oos["trades"]), "audit_trades_oos_full.csv")

    print("=" * 78); print("SAVED"); print("=" * 78)
    print("  audit_usd_metrics.csv / usd_metrics.csv")
    print("  audit_usd_monthly.csv  audit_oos_metrics.csv  audit_ablation.csv")
    print("  audit_loyo.csv  audit_placebo.csv  audit_placebo_summary.csv")
    print("  audit_multiple_testing.csv  audit_trades_is_full.csv  audit_trades_oos_full.csv")
    print(f"TOTAL RUNTIME {time.perf_counter()-t_start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
