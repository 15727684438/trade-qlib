"""STRATEGY V2 — EMA20/144 regime + structural-low trail, IS-only pre-registered grid.

Spec is pre-registered; runs are fixed (no extra runs). OOS must NOT be touched.
See scripts/run_risk_grid.py for the risk-grid conventions reused here.

Run:  python scripts/run_ema_struct_grid.py
Out:  results/ema_grid.csv, ema_bootstrap.csv, ema_placebo.csv,
      ema_trades_<tag>.csv, STRATEGY_V2.md
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
from tf_mnq.engine import CostCfg  # noqa: E402
from tf_mnq.features import compute_features  # noqa: E402
from tf_mnq.strategies import strategy_frame, veto_blocked  # noqa: E402
from tf_mnq.strategies_v2 import build_signals_v2, regime_series, run_ema_struct  # noqa: E402

RESULTS_DIR = os.path.join(_PKG_ROOT, "results")
IS_START, IS_END = "2019-05-06", "2023-12-31"
COST = CostCfg(slip_ticks=0.4, commission_per_side=0.5, tick=0.25, point_value=2.0)
COST3 = CostCfg(slip_ticks=3.0, commission_per_side=0.5, tick=0.25, point_value=2.0)
PV, ACCOUNT = 2.0, 50000.0
DD_GATE, DD_HARD = 1200.0, 1500.0
PROFIT_GATE = 5000.0
N_BOOT, N_PLACEBO, N_TRIALS = 2000, 500, 100
# frozen stop anchor
ANCH = dict(pivot_n=2, buf=0.25, floor_mult=2.0, act_mult=0.0,
            regime_confirm=0, breakeven_atr=None)
BASE_GATES = dict(atrp_cap=90.0, er_min=None, rth=False)


def save(df, name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, name), index=False)


def pf_of(pnl):
    pnl = np.asarray(pnl, float)
    w = pnl[pnl > 0].sum()
    l = -pnl[pnl < 0].sum()
    return float(w / l) if l > 0 else (np.inf if w > 0 else 0.0)


def bootstrap_pf(trades, n_boot=N_BOOT, seed=999):
    if not trades:
        return np.nan, np.nan
    pnl = np.array([t["pnl_usd_net"] for t in trades], float)
    wk = pd.Series([pd.Timestamp(t["entry_time"]) for t in trades]).dt.to_period("W")
    groups = [pnl[(wk == w).to_numpy()] for w in wk.unique()]
    rng = np.random.default_rng(seed)
    pfs = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        pfs[i] = pf_of(np.concatenate([groups[j] for j in pick]))
    pfs = pfs[np.isfinite(pfs)]
    return float((pfs <= 1.0).mean()), float(np.percentile(pfs, 5))


def metrics_row(res, cost, tag, ppy):
    idx = res["equity"].index
    eq = res["equity"].to_numpy(float)
    cap = float(res["capital"])
    eq_usd = ACCOUNT + (eq - 1.0) * cap
    tr = res["trades"]
    pnl = np.array([t["pnl_usd_net"] for t in tr], float) if tr else np.array([])
    peak = np.maximum.accumulate(eq_usd)
    maxdd_usd = float((peak - eq_usd).max())
    years = (idx[-1] - idx[0]).days / 365.25
    end = float(eq_usd[-1])
    cagr = (end / ACCOUNT) ** (1.0 / years) - 1.0 if end > 0 and years > 0 else np.nan
    ret = np.diff(eq_usd) / eq_usd[:-1] if len(eq_usd) > 1 else np.array([])
    sd = ret.std(ddof=1) if ret.size > 1 else np.nan
    sharpe = ret.mean() / sd * np.sqrt(ppy) if (np.isfinite(sd) and sd > 0) else np.nan
    daily = pd.Series(eq_usd, index=idx).resample("D").last().dropna()
    worst_day = float(daily.diff().min()) if len(daily) > 1 else np.nan
    return {
        "tag": tag, "cost": "1.5t" if cost is COST else "3.0t",
        "n": int(len(tr)), "n_breakouts": int(res.get("n_breakouts", 0)),
        "PF": pf_of(pnl) if len(pnl) else np.nan,
        "win": float((pnl > 0).mean()) if len(pnl) else np.nan,
        "expectancy_usd": float(pnl.mean()) if len(pnl) else np.nan,
        "worst_trade_usd": float(pnl.min()) if len(pnl) else np.nan,
        "net_profit_usd": float(pnl.sum()) if len(pnl) else 0.0,
        "CAGR_usd": cagr, "Sharpe_usd": sharpe,
        "MaxDD_usd": maxdd_usd, "MaxDD_pt": maxdd_usd / PV,
        "MAR": cagr / (maxdd_usd / ACCOUNT) if maxdd_usd > 0 else np.nan,
        "worst_day_usd": worst_day,
        "halted_days": int(res.get("halted_days", 0)),
        "trades_foregone": int(res.get("n_breaker_blocked", 0)),
    }


# --------------------------------------------------------------------------- #
def build_specs():
    def sp(tag, family, kind, trigger=None, cost=COST, params=None, gates=None,
           breaker=None, regime=False):
        return {"tag": tag, "family": family, "kind": kind, "trigger": trigger,
                "cost": cost, "params": dict(params or {}), "gates": dict(gates or BASE_GATES),
                "breaker": breaker, "regime": regime}
    g_base = dict(BASE_GATES)
    S = [
        sp("B0_1.5t", "ref", "breakout", cost=COST),
        sp("B0_3t", "ref", "breakout", cost=COST3),
        sp("Bridge", "ref", "breakout", regime=True),
        sp("A_T2", "anchor", "struct", "T2", params=ANCH),
        sp("T1", "trigger", "struct", "T1", params=ANCH),
        sp("T3", "trigger", "struct", "T3", params=ANCH),
        sp("SP_buf0.5", "stop", "struct", "T2", params={**ANCH, "buf": 0.50}),
        sp("SP_pivotn3", "stop", "struct", "T2", params={**ANCH, "pivot_n": 3}),
        sp("SP_floor1.5", "stop", "struct", "T2", params={**ANCH, "floor_mult": 1.5}),
        sp("SP_act1.0", "stop", "struct", "T2", params={**ANCH, "act_mult": 1.0}),
        sp("G_atrp80", "gate", "struct", "T2", params=ANCH, gates={**g_base, "atrp_cap": 80.0}),
        sp("G_er25", "gate", "struct", "T2", params=ANCH, gates={**g_base, "er_min": 0.25}),
        sp("G_rth", "gate", "struct", "T2", params=ANCH, gates={**g_base, "rth": True}),
        sp("E_confirm2", "exit", "struct", "T2", params={**ANCH, "regime_confirm": 2}),
        sp("W_be1.0", "win", "struct", "T2", params={**ANCH, "breakeven_atr": 1.0}),
        sp("DD_daily100", "dd", "struct", "T2", params=ANCH, breaker={"daily": -100.0}),
        sp("DD_daily150", "dd", "struct", "T2", params=ANCH, breaker={"daily": -150.0}),
        sp("DD_halt300", "dd", "struct", "T2", params=ANCH, breaker={"dd": 300.0}),
        sp("DD_halt300_rth", "dd", "struct", "T2", params=ANCH, gates={**g_base, "rth": True},
           breaker={"dd": 300.0}),
    ]
    return S


def main() -> int:
    t0 = time.perf_counter()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    allb = load_bars("15min")
    feat_all = compute_features(allb)
    sf_all = strategy_frame(allb, feat_all)
    bars = allb.loc[(allb.index >= pd.Timestamp(IS_START)) & (allb.index <= pd.Timestamp(IS_END))].copy()
    feat = feat_all.reindex(bars.index)
    sf = sf_all.reindex(bars.index)
    atr = feat["atr"]
    ppy = len(bars) / ((bars.index[-1] - bars.index[0]).days / 365.25)
    bars1 = load_bars("1min", start=IS_START, end=IS_END)
    print(f"IS 15min={len(bars)} 1min={len(bars1)} ppy={ppy:.1f}")

    regime, e20, e144 = regime_series(bars)
    print(f"regime on {int(regime.sum())}/{len(bars)} bars ({regime.mean()*100:.1f}%)")

    def blocked_breakout(gates):
        return veto_blocked(bars, feat, sf, components=["vr", "vol", "ext"], ext_mult=2.5,
                            atr_pct_cap=float(gates["atrp_cap"]))

    def run_spec(s, cost):
        if s["kind"] == "breakout":
            blk = blocked_breakout(s["gates"])
            if s["regime"]:
                blk = blk | (~regime)
            return run_breakout_ex(bars, atr, cost, donch=20, stop_mult=2.5,
                                   veto_long=blk, entry_mode="market")
        sig, _e20, _e144 = build_signals_v2(bars, feat, sf, s["trigger"], **s["gates"])
        return run_ema_struct(bars, bars1, feat, sig, cost, exit_ema=e144,
                              breaker=s["breaker"], **s["params"])

    # ---------------- run fixed grid ----------------
    specs = build_specs()
    rows, res_store = [], {}
    t1 = time.perf_counter()
    for s in specs:
        res = run_spec(s, s["cost"])
        row = metrics_row(res, s["cost"], s["tag"], ppy)
        row["family"] = s["family"]
        bp, ci5 = bootstrap_pf(res["trades"])
        row["boot_p"], row["boot_ci5_PF"] = bp, ci5
        rows.append(row)
        res_store[s["tag"]] = (res, s)
        save(pd.DataFrame(res["trades"]), f"ema_trades_{s['tag']}.csv")
        print(f"  {s['tag']:16s} cost={row['cost']} n={row['n']:5d} PF={row['PF']:.3f} "
              f"win={row['win']:.3f} net=${row['net_profit_usd']:.0f} "
              f"MaxDD=${row['MaxDD_usd']:.0f} Sharpe={row['Sharpe_usd']:.2f} "
              f"({time.perf_counter()-t1:.0f}s)")
    grid = pd.DataFrame(rows)

    # ---------------- combos (only 2, pre-registered) ----------------
    def prelim(s_tag):
        r = grid[(grid["tag"] == s_tag) & (grid["cost"] == "1.5t")].iloc[0]
        return (r["MaxDD_usd"] <= DD_GATE and r["net_profit_usd"] >= PROFIT_GATE
                and r["boot_p"] <= 0.10 and r["PF"] >= 1.10
                and 0.42 <= r["win"] <= 0.55 and r["Sharpe_usd"] >= 1.0
                and r["expectancy_usd"] >= 4.0 and r["n"] >= 250
                and r["worst_trade_usd"] >= -400.0)

    def rank_of(tag):
        r = grid[(grid["tag"] == tag) & (grid["cost"] == "1.5t")].iloc[0]
        return (bool(prelim(tag)), float(r["net_profit_usd"]), float(r["Sharpe_usd"]),
                float(r["win"]), -float(r["MaxDD_usd"]))

    trig_tags = ["A_T2", "T1", "T3"]
    stop_tags = ["SP_buf0.5", "SP_pivotn3", "SP_floor1.5", "SP_act1.0"]
    best_trig = max(trig_tags, key=rank_of)
    best_stop = max(stop_tags, key=rank_of)
    print(f"prelim picks: best_trigger={best_trig} best_stop_probe={best_stop}")

    def combo(tag, rth):
        base_s = res_store[best_stop][1]
        trig = res_store[best_trig][1]["trigger"]
        gates = {**BASE_GATES, "rth": rth}
        return {"tag": tag, "family": "combo", "kind": "struct", "trigger": trig,
                "cost": COST, "params": dict(base_s["params"]), "gates": gates,
                "breaker": {"dd": 300.0}, "regime": False}

    x_specs = [combo("X_best_dd300", False), combo("X_best_dd300_rth", True)]
    for s in x_specs:
        res = run_spec(s, s["cost"])
        row = metrics_row(res, s["cost"], s["tag"], ppy)
        row["family"] = s["family"]
        bp, ci5 = bootstrap_pf(res["trades"])
        row["boot_p"], row["boot_ci5_PF"] = bp, ci5
        rows.append(row)
        res_store[s["tag"]] = (res, s)
        save(pd.DataFrame(res["trades"]), f"ema_trades_{s['tag']}.csv")
        print(f"  {s['tag']:16s} cost={row['cost']} n={row['n']:5d} PF={row['PF']:.3f} "
              f"win={row['win']:.3f} net=${row['net_profit_usd']:.0f} MaxDD=${row['MaxDD_usd']:.0f}")
    grid = pd.DataFrame(rows)

    # ---------------- acceptance ----------------
    def g_of(row):
        if row is None:
            return (False, False, False, False)
        g1 = bool(row["MaxDD_usd"] <= DD_GATE)
        g2 = bool(row["net_profit_usd"] >= PROFIT_GATE)
        g3 = bool(0.42 <= row["win"] <= 0.55)
        g4 = bool(row["Sharpe_usd"] >= 1.0 and row["expectancy_usd"] >= 4.0
                  and row["n"] >= 250 and row["worst_trade_usd"] >= -400.0)
        return g1, g2, g3, g4

    DOM = {"buf": [0.25, 0.50], "floor_mult": [1.5, 2.0]}

    results = []
    for tag in grid[grid["cost"] == "1.5t"]["tag"]:
        r15 = grid[(grid["tag"] == tag) & (grid["cost"] == "1.5t")]
        r30 = grid[(grid["tag"] == tag) & (grid["cost"] == "3.0t")]
        r15 = r15.iloc[0] if len(r15) else None
        r30 = r30.iloc[0] if len(r30) else None
        g1a, g2a, g3a, g4a = g_of(r15)
        g1b, g2b, g3b, g4b = g_of(r30)
        s = res_store[tag][1]
        # gate 5: bootstrap + neighborhood on (buf, floor)
        boot_ok = bool((r15 is not None and r15["boot_p"] <= 0.10)
                       and (r30 is not None and r30["boot_p"] <= 0.10))
        nbr_dd, nbr_ok = [], True
        if s["kind"] == "struct":
            for pname, dom in DOM.items():
                cur = s["params"].get(pname)
                if cur is None or cur not in dom:
                    continue
                j = dom.index(cur)
                for dj in (-1, 1):
                    kk = j + dj
                    if not (0 <= kk < len(dom)):
                        continue
                    nov = dict(res_store[tag][1])
                    nov["params"] = {**s["params"], pname: dom[kk]}
                    rn = run_spec(nov, COST)
                    rn3 = run_spec(nov, COST3)
                    m = metrics_row(rn, COST, f"{tag}~{pname}", ppy)
                    m3 = metrics_row(rn3, COST3, f"{tag}~{pname}", ppy)
                    nbr_dd.append(max(m["MaxDD_usd"], m3["MaxDD_usd"]))
        if nbr_dd:
            nbr_ok = bool(max(nbr_dd) <= DD_HARD)
        passed = bool(g1a and g2a and g3a and g4a and g1b and g2b and g3b and g4b
                      and boot_ok and nbr_ok)
        results.append({
            "tag": tag, "family": s["family"],
            "g1_1p5": g1a, "g2_1p5": g2a, "g3_1p5": g3a, "g4_1p5": g4a,
            "ran_3t": r30 is not None, "g1_3t": g1b, "g2_3t": g2b, "g3_3t": g3b, "g4_3t": g4b,
            "boot_ok": boot_ok, "g5_nbr": nbr_ok, "pass": passed,
            "nbr_max_DD_usd": (max(nbr_dd) if nbr_dd else np.nan),
        })
    res_tab = pd.DataFrame(results)
    grid = grid.merge(res_tab, on=["tag", "family"], how="left")
    save(grid, "ema_grid.csv")

    passers = res_tab[res_tab["pass"]]
    print("=" * 78)
    print(f"IS PASSERS: {len(passers)}")

    # ---------------- selection (lexicographic) ----------------
    chosen = None
    if len(passers):
        cand = grid[(grid["cost"] == "1.5t") & (grid["tag"].isin(passers["tag"]))].copy()
        best_profit = cand["net_profit_usd"].max()
        near = cand[cand["net_profit_usd"] >= 0.9 * best_profit]
        near = near.sort_values(["Sharpe_usd", "win", "MaxDD_usd"],
                                ascending=[False, False, True])
        chosen = near.iloc[0]["tag"]
    print(f"CHOSEN: {chosen}")

    # ---------------- DSR sanity (N=100) + Sidak ----------------
    def is_sharpe(res):
        e = res["equity"].to_numpy(float)
        r = np.diff(e) / e[:-1]
        return r.mean() / r.std(ddof=1) * np.sqrt(ppy), r

    sr_grid = []
    for tag in grid[grid["cost"] == "1.5t"]["tag"]:
        sr, _ = is_sharpe(res_store[tag][0])
        sr_grid.append(sr)
    sr_grid = np.array(sr_grid, float)
    V = float(np.nanvar(sr_grid / np.sqrt(ppy), ddof=1))
    gamma = 0.5772156649015329

    def dsr_of(sr_ann, r):
        srp = sr_ann / np.sqrt(ppy)
        g3 = float(st.skew(r)); g4 = float(st.kurtosis(r, fisher=False)); T = len(r)
        emax = np.sqrt(V) * ((1 - gamma) * st.norm.ppf(1 - 1.0 / N_TRIALS)
                             + gamma * st.norm.ppf(1 - 1.0 / (N_TRIALS * np.e)))
        denom = np.sqrt(max(1e-12, 1 - g3 * srp + ((g4 - 1) / 4) * srp ** 2))
        return float(st.norm.cdf((srp - emax) * np.sqrt(T - 1) / denom)), emax * np.sqrt(ppy)

    ref_tag = chosen if chosen else "A_T2"
    sr_ref, ret_ref = is_sharpe(res_store[ref_tag][0])
    dsr_ref, sr0 = dsr_of(sr_ref, ret_ref)
    # Sidak for the ref's bootstrap p
    p_ref = float(grid[(grid["tag"] == ref_tag) & (grid["cost"] == "1.5t")].iloc[0]["boot_p"])
    sidak = float(1 - (1 - p_ref) ** N_TRIALS)
    print(f"DSR({ref_tag})={dsr_ref:.4f} p={1-dsr_ref:.4f} (E[maxSR]={sr0:.3f}, obs={sr_ref:.3f}) "
          f"Sidak p={sidak:.4f}")

    # ---------------- placebo (n-matched, same stop + chosen breaker) ----
    ref_res, ref_s = res_store[ref_tag]
    ref_row = grid[(grid["tag"] == ref_tag) & (grid["cost"] == "1.5t")].iloc[0]
    real_pf, real_ntr = float(ref_row["PF"]), int(ref_row["n"])
    universe = np.flatnonzero((regime & np.isfinite(atr)).to_numpy())
    rng = np.random.default_rng(31337)

    def placebo_run(draw):
        pick = rng.choice(universe, size=min(draw, universe.size), replace=False)
        mask = np.zeros(len(bars), bool)
        mask[pick] = True
        sig = pd.Series(mask, index=bars.index)
        return run_ema_struct(bars, bars1, feat, sig, COST, exit_ema=e144,
                              breaker=ref_s["breaker"], **ref_s["params"])

    scale, draw = 1.0, real_ntr
    for _ in range(3):
        med = float(np.median([len(placebo_run(int(round(scale * real_ntr)))["trades"]) for _ in range(10)]))
        if med <= 0:
            break
        scale = min(3.0, scale * real_ntr / med)
        draw = min(int(round(scale * real_ntr)), universe.size)
        if abs(med - real_ntr) / max(real_ntr, 1) < 0.05:
            break
    print(f"placebo draw={draw} (scale={scale:.2f}) target n={real_ntr}")
    t2 = time.perf_counter()
    plac_pf = np.empty(N_PLACEBO)
    plac_ntr = np.empty(N_PLACEBO, dtype=int)
    for i in range(N_PLACEBO):
        r = placebo_run(draw)
        pnl = np.array([t["pnl_usd_net"] for t in r["trades"]], float) if r["trades"] else np.array([0.0])
        plac_pf[i] = pf_of(pnl)
        plac_ntr[i] = len(r["trades"])
    pf_f = plac_pf[np.isfinite(plac_pf)]
    pf_pct = float((pf_f < real_pf).mean())
    save(pd.DataFrame({"PF": plac_pf, "n_trades": plac_ntr}), "ema_placebo.csv")
    print(f"placebo {N_PLACEBO} in {time.perf_counter()-t2:.0f}s: real PF={real_pf:.3f} "
          f"pct={pf_pct*100:.1f}% p95={np.percentile(pf_f,95):.3f} max={pf_f.max():.3f} "
          f"median_n={np.median(plac_ntr):.0f}")

    # ---------------- per-year net $ (winner) ----------------
    def year_tab(res):
        out = {}
        for y in (2019, 2020, 2021, 2022, 2023):
            p = [t["pnl_usd_net"] for t in res["trades"]
                 if pd.Timestamp(t["entry_time"]).year == y]
            out[y] = float(np.sum(p))
        return out
    yc = year_tab(ref_res)
    pos_years = sum(1 for v in yc.values() if v > 0)
    print(f"per-year net $ ({ref_tag}): {[(k, round(v)) for k, v in yc.items()]} -> {pos_years}/5 positive")

    # ---------------- bootstrap CSV ----------------
    b15 = grid[grid["cost"] == "1.5t"][["tag", "boot_p", "boot_ci5_PF"]]
    b3 = grid[grid["cost"] == "3.0t"][["tag", "boot_p", "boot_ci5_PF"]].rename(
        columns={"boot_p": "boot_p_3t", "boot_ci5_PF": "boot_ci5_PF_3t"})
    boot_tab = pd.concat([b15.set_index("tag"), b3.set_index("tag")], axis=1).reset_index()
    save(boot_tab, "ema_bootstrap.csv")

    # ---------------- markdown ----------------
    def frow(tag, cost):
        r = grid[(grid["tag"] == tag) & (grid["cost"] == cost)]
        if not len(r):
            return None
        r = r.iloc[0]
        return (f"| {tag} | {cost} | {int(r['n'])} | {r['PF']:.3f} | {r['win']:.3f} | "
                f"${r['expectancy_usd']:.2f} | ${r['net_profit_usd']:.0f} | {r['Sharpe_usd']:.2f} | "
                f"${r['MaxDD_usd']:.0f} | ${r['worst_trade_usd']:.0f} | {int(r['halted_days'])} | "
                f"{int(r['trades_foregone'])} |")

    L = []
    L.append("# STRATEGY V2 — EMA20/144 regime + structural trail (IS-only verdict)\n")
    L.append("- Core regime EMA20>EMA144 & close>EMA144 (fixed); structural stop n=2/buf0.25/floor2.0/act0; "
             "base gates {VR<1, ATRpct<=90, ext<=2.5}; 1 contract, $50k, 1.5t (stress 3t).\n")
    L.append("- Cost model corrected 2026-10-10: base ('1.5t') = $0.50/side comm + $0.20/side slip; "
             "stress ('3.0t') = $0.50/side + $1.50/side slip.\n")
    L.append(f"- Pre-registered gates (both 1.5t & 3t): MaxDD_usd@1c<=${DD_GATE:.0f}; net profit>=${PROFIT_GATE:.0f}; "
             f"win in [0.42,0.55]; Sharpe>=1.0; expectancy>=$4; n>=250; worst trade>=-$400; "
             f"boot p<=0.10; neighborhood MaxDD<=${DD_HARD:.0f}.\n")
    L.append(f"- **Verdict: {'1+ PASSER(S)' if len(passers) else 'ZERO PASSERS'}**\n")
    if len(passers):
        L.append("## Passers\n")
        L.append("| tag | 1.5t net$ | 1.5t Sharpe | 1.5t MaxDD$ | 3t net$ | 3t MaxDD$ | nbr maxDD$ |")
        L.append("|---|---|---|---|---|---|---|")
        for _, p in passers.iterrows():
            r15 = grid[(grid["tag"] == p["tag"]) & (grid["cost"] == "1.5t")].iloc[0]
            r30 = grid[(grid["tag"] == p["tag"]) & (grid["cost"] == "3.0t")]
            r30 = r30.iloc[0] if len(r30) else None
            net3 = f"${r30['net_profit_usd']:.0f}" if r30 is not None else "—"
            dd3 = f"${r30['MaxDD_usd']:.0f}" if r30 is not None else "—"
            L.append(f"| {p['tag']} | ${r15['net_profit_usd']:.0f} | {r15['Sharpe_usd']:.2f} | "
                     f"${r15['MaxDD_usd']:.0f} | {net3} | {dd3} | ${p['nbr_max_DD_usd']:.0f} |")
        L.append("")
    L.append("## Grid (IS)\n")
    L.append("| tag | cost | n | PF | win | exp$ | net$ | Sharpe$ | MaxDD$ | worst trade$ | halted d | foregone | g1 | g2 | g3 | g4 | pass |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in grid.sort_values(["tag", "cost"]).iterrows():
        L.append(
            f"| {r['tag']} | {r['cost']} | {int(r['n'])} | {r['PF']:.3f} | {r['win']:.3f} | "
            f"${r['expectancy_usd']:.2f} | ${r['net_profit_usd']:.0f} | {r['Sharpe_usd']:.2f} | "
            f"${r['MaxDD_usd']:.0f} | ${r['worst_trade_usd']:.0f} | {int(r['halted_days'])} | "
            f"{int(r['trades_foregone'])} | {bool(r.get('g1_1p5'))} | {bool(r.get('g2_1p5'))} | "
            f"{bool(r.get('g3_1p5'))} | {bool(r.get('g4_1p5'))} | {bool(r.get('pass'))} |")
    L.append("")
    L.append("## Diagnostics\n")
    L.append(f"- DSR({ref_tag}) = {dsr_ref:.4f} (p={1-dsr_ref:.4f}); E[max SR] N={N_TRIALS} = {sr0:.3f}, "
             f"observed = {sr_ref:.3f}; Sidak p = {sidak:.4f}.")
    L.append(f"- Placebo ({N_PLACEBO} random-entry, n-matched, same structural stop + breaker): "
             f"real PF={real_pf:.3f} at {pf_pct*100:.1f}th pct (p95={np.percentile(pf_f,95):.3f}, "
             f"max={pf_f.max():.3f}); median placebo n={np.median(plac_ntr):.0f} vs real {real_ntr}.")
    L.append(f"- Per-year net $ ({ref_tag}): {[(k, round(v)) for k, v in yc.items()]} -> {pos_years}/5 positive.")
    L.append("- OOS NOT touched.")
    L.append("")
    with open(os.path.join(RESULTS_DIR, "STRATEGY_V2.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))

    print("=" * 78)
    print("SAVED: ema_grid.csv, ema_bootstrap.csv, ema_placebo.csv, ema_trades_*.csv, STRATEGY_V2.md")
    print(f"TOTAL RUNTIME {time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
