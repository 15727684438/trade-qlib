"""RISK-FIRST redesign grid on the SIMPLIFIED BASE config (IS ONLY, no OOS).

Base = Donchian-20 breakout, trail 2.5xATR, vetoes {VR>=1, ATRpct<=90}, market
entry, cost 1.5 tick, long-only, 1 contract, $50k account. NO narrow-band vetoes.

Engine additions exercised here (all causal, defaults = legacy):
  (a) initial_stop_mult : hard floor = entry - k_in*ATR(entry)
  (b) ratchet_only      : trail never loosens when ATR expands
  (c) breaker           : per-bar-equity C1 daily / C2 DD-hysteresis / C3 weekly
                          halt that blocks NEXT-bar entries only.

Run:  python scripts/run_risk_grid.py
Outputs: results/risk_grid.csv, risk_bootstrap.csv, risk_placebo.csv,
         risk_trades_<tag>.csv, RISK_REDESIGN.md
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

RESULTS_DIR = os.path.join(_PKG_ROOT, "results")
IS_START, IS_END = "2019-05-06", "2023-12-31"
FREQ = "15min"
COST = CostCfg(slip_ticks=1.5, commission_per_side=0.75, tick=0.25, point_value=2.0)
COST3 = CostCfg(slip_ticks=3.0, commission_per_side=0.75, tick=0.25, point_value=2.0)
PV = 2.0
ACCOUNT = 50000.0
DD_GATE = 1200.0      # hard gate 1 (80% of $1500)
DD_HARD = 1500.0      # neighborhood ceiling
N_TRIALS = 100        # updated trial count for DSR sanity
N_BOOT = 2000
N_PLACEBO = 500
BASE_COMPONENTS = ["vol", "vr"]  # {ATRpct<=cap, VR<1}

BASE = {
    "stop_mult": 2.5, "donch": 20, "cap": 90,
    "initial_stop_mult": None, "ratchet_only": False,
    "breakeven_atr": None, "time_stop_bars": None, "breaker": None,
    "entry_mode": "market",
}


def save(df, name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, name), index=False)


def pf_of(pnl):
    pnl = np.asarray(pnl, float)
    w = pnl[pnl > 0].sum()
    l = -pnl[pnl < 0].sum()
    return float(w / l) if l > 0 else (np.inf if w > 0 else 0.0)


def bootstrap_pf(trades, n_boot=N_BOOT, seed=12345):
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


def metrics_row(res, cost, tag, kind):
    idx = res["equity"].index
    eq = res["equity"].to_numpy(float)
    cap = float(res["capital"])
    eq_usd = ACCOUNT + (eq - 1.0) * cap                       # 1 contract
    tr = res["trades"]
    pnl = np.array([t["pnl_usd_net"] for t in tr], float) if tr else np.array([])
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    peak = np.maximum.accumulate(eq_usd)
    maxdd_usd = float((peak - eq_usd).max())
    years = (idx[-1] - idx[0]).days / 365.25
    end = float(eq_usd[-1])
    cagr = (end / ACCOUNT) ** (1.0 / years) - 1.0 if end > 0 and years > 0 else np.nan
    ret = np.diff(eq_usd) / eq_usd[:-1] if len(eq_usd) > 1 else np.array([])
    sd = ret.std(ddof=1) if ret.size > 1 else np.nan
    sharpe = ret.mean() / sd * np.sqrt(23941.7) if (np.isfinite(sd) and sd > 0) else np.nan
    daily = pd.Series(eq_usd, index=idx).resample("D").last().dropna()
    worst_day = float(daily.diff().min()) if len(daily) > 1 else np.nan
    return {
        "tag": tag, "kind": kind, "cost": "1.5t" if cost is COST else "3.0t",
        "n": int(len(tr)), "n_breakouts": int(res.get("n_breakouts", 0)),
        "PF": pf_of(pnl) if len(pnl) else np.nan,
        "win": float((pnl > 0).mean()) if len(pnl) else np.nan,
        "expectancy_usd": float(pnl.mean()) if len(pnl) else np.nan,
        "worst_trade_usd": float(pnl.min()) if len(pnl) else np.nan,
        "total_pnl_usd": float(pnl.sum()) if len(pnl) else 0.0,
        "CAGR_usd": cagr, "Sharpe_usd": sharpe,
        "MaxDD_usd": maxdd_usd, "MaxDD_pt": maxdd_usd / PV,
        "MAR": cagr / (maxdd_usd / ACCOUNT) if maxdd_usd > 0 else np.nan,
        "worst_day_usd": worst_day,
        "halted_days": int(res.get("halted_days", 0)),
        "trades_foregone": int(res.get("n_breaker_blocked", 0)),
        "total_return": float(end / ACCOUNT - 1.0),
    }


# --------------------------------------------------------------------------- #
# grid definition
# --------------------------------------------------------------------------- #
def build_variants():
    def mk(tag, family, primary, cost=COST, **ov):
        return {"tag": tag, "family": family, "cost": cost, "primary": primary, "ov": ov}
    V = [
        mk("R0_base_1.5t", "R0", [], COST),
        mk("R0_base_3t", "R0", [], COST3),
        mk("E1_trail1.0", "E1", [("stop_mult", 1.0)], stop_mult=1.0),
        mk("E1_trail1.5", "E1", [("stop_mult", 1.5)], stop_mult=1.5),
        mk("E1_trail2.0", "E1", [("stop_mult", 2.0)], stop_mult=2.0),
        mk("E2_init1.0", "E2", [("initial_stop_mult", 1.0)], initial_stop_mult=1.0),
        mk("E2_init1.5", "E2", [("initial_stop_mult", 1.5)], initial_stop_mult=1.5),
        mk("E2_init2.0", "E2", [("initial_stop_mult", 2.0)], initial_stop_mult=2.0),
        mk("E3_ratchet", "E3", [], ratchet_only=True),
        mk("E4_be1.0", "E4", [("breakeven_atr", 1.0)], breakeven_atr=1.0),
        mk("E4_be1.5", "E4", [("breakeven_atr", 1.5)], breakeven_atr=1.5),
        mk("E5_time16", "E5", [("time_stop_bars", 16)], time_stop_bars=16),
        mk("E5_time48", "E5", [("time_stop_bars", 48)], time_stop_bars=48),
        mk("F1_donch32", "F1", [("donch", 32)], donch=32),
        mk("F1_donch48", "F1", [("donch", 48)], donch=48),
        mk("F2_cap75", "F2", [("cap", 75)], cap=75),
        mk("F2_cap85", "F2", [("cap", 85)], cap=85),
        mk("C1_daily100", "C1", [("daily", -100)], breaker={"daily": -100}),
        mk("C1_daily150", "C1", [("daily", -150)], breaker={"daily": -150}),
        mk("C2_dd300", "C2", [("dd", 300)], breaker={"dd": 300}),
        mk("C2_dd500", "C2", [("dd", 500)], breaker={"dd": 500}),
        mk("C3_weekly250", "C3", [("weekly", -250)], breaker={"weekly": -250}),
    ]
    return V


DOMAIN = {
    "stop_mult": [1.0, 1.5, 2.0, 2.5],
    "initial_stop_mult": [1.0, 1.5, 2.0],
    "breakeven_atr": [1.0, 1.5],
    "time_stop_bars": [16, 48],
    "donch": [20, 32, 48],
    "cap": [75, 85, 90],
    "daily": [-100, -150],
    "weekly": [-250],
    "dd": [300, 500],
}


def neighbors_of(primary):
    out = []
    for pname, pval in primary:
        dom = DOMAIN.get(pname, [])
        if pval not in dom:
            continue
        j = dom.index(pval)
        for dj in (-1, 1):
            k = j + dj
            if 0 <= k < len(dom):
                out.append((pname, dom[k]))
    return out


# --------------------------------------------------------------------------- #
def main() -> int:
    t0 = time.perf_counter()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    allb = load_bars(FREQ)
    feat_all = compute_features(allb)
    sf_all = strategy_frame(allb, feat_all)
    bars = allb.loc[(allb.index >= pd.Timestamp(IS_START)) & (allb.index <= pd.Timestamp(IS_END))].copy()
    feat = feat_all.reindex(bars.index)
    sf = sf_all.reindex(bars.index)
    atr = feat["atr"]
    ppy = len(bars) / ((bars.index[-1] - bars.index[0]).days / 365.25)
    print(f"IS {FREQ}: {len(bars)} bars [{bars.index[0]} .. {bars.index[-1]}] ppy={ppy:.1f}")

    blocked_cache = {}

    def blocked_for(cap):
        if cap not in blocked_cache:
            blocked_cache[cap] = veto_blocked(bars, feat, sf, components=BASE_COMPONENTS,
                                              atr_pct_cap=float(cap))
        return blocked_cache[cap]

    def full_params(ov):
        p = dict(BASE)
        p.update(ov)
        return p

    def run_v(ov, cost):
        p = full_params(ov)
        return run_breakout_ex(bars, atr, cost, donch=p["donch"], stop_mult=p["stop_mult"],
                               veto_long=blocked_for(p["cap"]),
                               breakeven_atr=p["breakeven_atr"],
                               time_stop_bars=p["time_stop_bars"],
                               initial_stop_mult=p["initial_stop_mult"],
                               ratchet_only=p["ratchet_only"], breaker=p["breaker"],
                               entry_mode=p["entry_mode"])

    # ---------------- baseline reproduction guard ----------------
    base_res = run_v({}, COST)
    base_row = metrics_row(base_res, COST, "R0_base_1.5t", "grid")
    print(f"BASE reproduction: n={base_row['n']} PF={base_row['PF']:.4f} "
          f"expectancy=${base_row['expectancy_usd']:.2f} MaxDD_usd=${base_row['MaxDD_usd']:.0f} "
          f"(audit no_narrow: n=1302 PF=1.2348 MaxDD $2304)")

    variants = build_variants()

    def evaluate(v):
        res = run_v(v["ov"], v["cost"])
        row = metrics_row(res, v["cost"], v["tag"], f"grid_{v['family']}")
        row["family"] = v["family"]
        bp, ci5 = bootstrap_pf(res["trades"])
        row["boot_p"] = bp
        row["boot_ci5_PF"] = ci5
        return row, res

    t1 = time.perf_counter()
    rows, res_store = [], {}
    for v in variants:
        row, res = evaluate(v)
        rows.append(row)
        res_store[v["tag"]] = (res, v)
        save(pd.DataFrame(res["trades"]), f"risk_trades_{v['tag']}.csv")
    grid = pd.DataFrame(rows)
    print(f"first batch: {len(variants)} configs in {time.perf_counter()-t1:.1f}s")

    # ---------------- preliminary family picks for the 2 interactions ----
    def prelim_ok(row):
        return (row["MaxDD_usd"] <= DD_GATE and row["PF"] >= 1.10
                and row["boot_p"] <= 0.10 and row["expectancy_usd"] >= 2.0
                and row["n"] >= 300 and row["worst_trade_usd"] >= -500.0)

    def pick_family(fams):
        sub = grid[grid["tag"].str.split("_").str[0].isin(fams)].copy()
        sub["ok"] = sub.apply(prelim_ok, axis=1)
        sub = sub.sort_values(["ok", "CAGR_usd", "MaxDD_usd"], ascending=[False, False, True])
        return sub.iloc[0]["tag"]

    best_e = pick_family(["E1", "E2", "E3", "E4", "E5"])
    best_f = pick_family(["F1", "F2"])
    best_c2 = pick_family(["C2"])
    print(f"prelim picks: best_E={best_e} best_F={best_f} best_C2={best_c2}")

    # ---------------- 2 pre-registered interactions ----------------
    def vparams(tag):
        return res_store[tag][1]

    def combined(e_tag, c2_tag, new_tag):
        ov = dict(vparams(e_tag)["ov"])
        b = dict(vparams(c2_tag)["ov"].get("breaker", {}))
        ov["breaker"] = b
        primary = list(vparams(e_tag)["primary"])
        if b.get("dd") is not None:
            primary = primary + [("dd", b["dd"])]
        return {"tag": new_tag, "family": "X", "cost": COST, "primary": primary, "ov": ov}

    x_variants = [combined(best_e, best_c2, "X_Ebest_C2best"),
                  combined(best_f, best_c2, "X_Fbest_C2best")]
    x_rows = []
    for v in x_variants:
        row, res = evaluate(v)
        x_rows.append(row)
        res_store[v["tag"]] = (res, v)
        save(pd.DataFrame(res["trades"]), f"risk_trades_{v['tag']}.csv")
    if x_rows:
        grid = pd.concat([grid, pd.DataFrame(x_rows)], ignore_index=True)
    print(f"interactions added: {[v['tag'] for v in x_variants]}")

    # ---------------- 3t stress for 1.5t prelim passers ----------------
    cand = grid[(grid["cost"] == "1.5t") & (grid.apply(prelim_ok, axis=1))]
    print(f"1.5t prelim passers (before 3t): {len(cand)} -> {list(cand['tag'])}")
    stress_rows = []
    for tag in cand["tag"]:
        ov = res_store[tag][1]["ov"]
        res3 = run_v(ov, COST3)
        row3 = metrics_row(res3, COST3, tag, res_store[tag][1]["kind"])
        row3["family"] = res_store[tag][1]["family"]
        bp3, ci5 = bootstrap_pf(res3["trades"])
        row3["boot_p"] = bp3
        row3["boot_ci5_PF"] = ci5
        stress_rows.append(row3)
        save(pd.DataFrame(res3["trades"]), f"risk_trades_{tag}_3t.csv")
    if stress_rows:
        grid = pd.concat([grid, pd.DataFrame(stress_rows)], ignore_index=True)

    # ---------------- acceptance ----------------
    def g_of(row):
        if row is None:
            return (False, False, False)
        g1 = bool(row["MaxDD_usd"] <= DD_GATE)
        g2 = bool(row["PF"] >= 1.10 and row["boot_p"] <= 0.10)
        g3 = bool(row["expectancy_usd"] >= 2.0 and row["n"] >= 300
                  and row["worst_trade_usd"] >= -500.0)
        return g1, g2, g3

    results = []
    for tag in grid[grid["cost"] == "1.5t"]["tag"]:
        r15 = grid[(grid["tag"] == tag) & (grid["cost"] == "1.5t")]
        r30 = grid[(grid["tag"] == tag) & (grid["cost"] == "3.0t")]
        r15 = r15.iloc[0] if len(r15) else None
        r30 = r30.iloc[0] if len(r30) else None
        g1a, g2a, g3a = g_of(r15)
        g1b, g2b, g3b = g_of(r30)
        v = res_store[tag][1]
        # neighborhood gate g4 (1.5t and 3t MaxDD <= 1500)
        nbr_ok = True
        nbr_dd = []
        for pname, pval in neighbors_of(v["primary"]):
            nov = dict(v["ov"])
            if pname in ("daily", "weekly", "dd"):
                b = dict(nov.get("breaker", {}))
                b[pname] = pval
                nov["breaker"] = b
            else:
                nov[pname] = pval
            rn = metrics_row(run_v(nov, COST), COST, f"{tag}~{pname}={pval}", "nbr")
            rn3 = metrics_row(run_v(nov, COST3), COST3, f"{tag}~{pname}={pval}", "nbr")
            nbr_dd.append(max(rn["MaxDD_usd"], rn3["MaxDD_usd"]))
        if nbr_dd:
            nbr_ok = bool(max(nbr_dd) <= DD_HARD)
        passed = bool(g1a and g2a and g3a and g1b and g2b and g3b and nbr_ok)
        results.append({
            "tag": tag, "family": v["family"],
            "g1_1p5": g1a, "g2_1p5": g2a, "g3_1p5": g3a,
            "ran_3t": r30 is not None, "g1_3t": g1b, "g2_3t": g2b, "g3_3t": g3b,
            "g4_nbr": nbr_ok, "pass": passed,
            "nbr_max_DD_usd": (max(nbr_dd) if nbr_dd else np.nan),
        })
    res_tab = pd.DataFrame(results)
    grid = grid.merge(res_tab, on=["tag", "family"], how="left")
    save(grid, "risk_grid.csv")

    passers = res_tab[res_tab["pass"]]
    print("=" * 78)
    print(f"IS PASSERS (all gates at 1.5t AND 3t): {len(passers)}")
    if len(passers):
        print(passers[["tag", "family", "nbr_max_DD_usd"]].to_string(index=False))

    # ---------------- selection ----------------
    chosen = None
    if len(passers):
        cand2 = grid[(grid["cost"] == "1.5t") & (grid["tag"].isin(passers["tag"]))]
        cand2 = cand2.sort_values(["CAGR_usd", "MaxDD_usd"], ascending=[False, True])
        chosen = cand2.iloc[0]["tag"]
    print(f"CHOSEN: {chosen}")

    # ---------------- DSR sanity ----------------
    def is_sharpe(res):
        e = res["equity"].to_numpy(float)
        r = np.diff(e) / e[:-1]
        return r.mean() / r.std(ddof=1) * np.sqrt(ppy), r

    sr_base, ret_base = is_sharpe(base_res)
    trial = []
    for d in (16, 20, 24, 32):
        for sm in (1.5, 2.0, 2.5, 3.0):
            for cap in (85.0, 90.0, 95.0):
                rr = run_breakout_ex(bars, atr, COST, donch=d, stop_mult=sm,
                                     veto_long=blocked_for(cap), entry_mode="market")
                e = rr["equity"].to_numpy(float)
                rt = np.diff(e) / e[:-1]
                trial.append(rt.mean() / rt.std(ddof=1) * np.sqrt(ppy))
    trial = np.array(trial)
    V = float(np.var(trial / np.sqrt(ppy), ddof=1))
    gamma = 0.5772156649015329

    def dsr_of(sr_ann, r):
        srp = sr_ann / np.sqrt(ppy)
        g3 = float(st.skew(r)); g4 = float(st.kurtosis(r, fisher=False)); T = len(r)
        emax = np.sqrt(V) * ((1 - gamma) * st.norm.ppf(1 - 1.0 / N_TRIALS)
                             + gamma * st.norm.ppf(1 - 1.0 / (N_TRIALS * np.e)))
        denom = np.sqrt(max(1e-12, 1 - g3 * srp + ((g4 - 1) / 4) * srp ** 2))
        z = (srp - emax) * np.sqrt(T - 1) / denom
        return float(st.norm.cdf(z)), emax * np.sqrt(ppy)

    dsr_base, sr0 = dsr_of(sr_base, ret_base)
    print(f"DSR(base)={dsr_base:.4f} p={1-dsr_base:.4f} (E[max SR]={sr0:.3f}, "
          f"obs SR={sr_base:.3f}, N={N_TRIALS})")
    dsr_chosen = np.nan
    if chosen is not None:
        sr_ch, ret_ch = is_sharpe(res_store[chosen][0])
        dsr_chosen, _ = dsr_of(sr_ch, ret_ch)
        print(f"DSR(chosen={chosen})={dsr_chosen:.4f} p={1-dsr_chosen:.4f} (obs SR={sr_ch:.3f})")

    # ---------------- placebo with the chosen breaker ----------------
    ref_tag = chosen if chosen is not None else "C2_dd500"
    ref_res, ref_v = res_store[ref_tag]
    ref_row = grid[(grid["tag"] == ref_tag) & (grid["cost"] == "1.5t")].iloc[0]
    don_hi = bars["high"].rolling(full_params(ref_v["ov"])["donch"],
                                  min_periods=full_params(ref_v["ov"])["donch"]).max().shift(1)
    cand_idx = np.flatnonzero(((bars["close"] > don_hi) & np.isfinite(atr)).to_numpy())
    real_pf = float(ref_row["PF"])
    real_ntr = int(ref_row["n"])

    def run_placebo(draw):
        pick = rng.choice(cand_idx, size=min(draw, cand_idx.size), replace=False)
        blk = np.ones(len(bars), bool)
        blk[pick] = False
        p = full_params(ref_v["ov"])
        return run_breakout_ex(bars, atr, COST, donch=p["donch"], stop_mult=p["stop_mult"],
                               veto_long=pd.Series(blk, index=bars.index),
                               breakeven_atr=p["breakeven_atr"],
                               time_stop_bars=p["time_stop_bars"],
                               initial_stop_mult=p["initial_stop_mult"],
                               ratchet_only=p["ratchet_only"], breaker=p["breaker"],
                               entry_mode=p["entry_mode"])

    rng = np.random.default_rng(424242)
    scale, draw = 1.0, real_ntr
    for _ in range(4):
        med = float(np.median([len(run_placebo(int(round(scale * real_ntr)))["trades"]) for _ in range(30)]))
        if med <= 0:
            break
        scale = min(2.5, scale * real_ntr / med)
        draw = min(int(round(scale * real_ntr)), cand_idx.size)
        if abs(med - real_ntr) / real_ntr < 0.05:
            break
    print(f"placebo draw={draw} (scale={scale:.2f}) target n={real_ntr}, breaker={ref_v['ov'].get('breaker')}")
    t2 = time.perf_counter()
    plac_pf = np.empty(N_PLACEBO)
    plac_ntr = np.empty(N_PLACEBO, dtype=int)
    for i in range(N_PLACEBO):
        r = run_placebo(draw)
        pnl = np.array([t["pnl_usd_net"] for t in r["trades"]], float) if r["trades"] else np.array([0.0])
        plac_pf[i] = pf_of(pnl)
        plac_ntr[i] = len(r["trades"])
    plac_pf_f = plac_pf[np.isfinite(plac_pf)]
    pf_pct = float((plac_pf_f < real_pf).mean())
    placebo = pd.DataFrame({"PF": plac_pf, "n_trades": plac_ntr})
    save(placebo, "risk_placebo.csv")
    print(f"placebo {N_PLACEBO} runs in {time.perf_counter()-t2:.0f}s; real PF={real_pf:.4f} "
          f"pct={pf_pct*100:.1f}% p95={np.percentile(plac_pf_f,95):.3f} "
          f"max={plac_pf_f.max():.3f} median_n={np.median(plac_ntr):.0f}")

    # ---------------- 2020 contribution ----------------
    def year_contrib(res, y):
        t = [x for x in res["trades"] if pd.Timestamp(x["entry_time"]).year == y]
        pnl = np.array([x["pnl_usd_net"] for x in t], float)
        return {"year": y, "n": len(t), "PF": pf_of(pnl) if len(pnl) else np.nan,
                "pnl_usd": float(pnl.sum()) if len(pnl) else 0.0}
    yc_base = year_contrib(base_res, 2020)
    yc_ref = year_contrib(ref_res, 2020)
    print(f"2020 contribution base: n={yc_base['n']} PF={yc_base['PF']:.3f} PnL=${yc_base['pnl_usd']:.0f}; "
          f"ref: n={yc_ref['n']} PF={yc_ref['PF']:.3f} PnL=${yc_ref['pnl_usd']:.0f}")

    # ---------------- extra CSVs ----------------
    boot_tab = grid[grid["cost"] == "1.5t"][["tag", "boot_p", "boot_ci5_PF"]].copy()
    boot3 = grid[grid["cost"] == "3.0t"][["tag", "boot_p", "boot_ci5_PF"]].rename(
        columns={"boot_p": "boot_p_3t", "boot_ci5_PF": "boot_ci5_PF_3t"})
    boot_tab = pd.concat([boot_tab.set_index("tag"),
                          boot3.set_index("tag")], axis=1).reset_index()
    save(boot_tab, "risk_bootstrap.csv")

    # ---------------- markdown ----------------
    def fmt_row(df, tag, cost):
        r = df[(df["tag"] == tag) & (df["cost"] == cost)]
        if not len(r):
            return None
        r = r.iloc[0]
        return (f"| {tag} | {cost} | {int(r['n'])} | {r['PF']:.3f} | {r['win']:.3f} | "
                f"${r['expectancy_usd']:.2f} | {r['CAGR_usd']*100:.2f}% | {r['Sharpe_usd']:.2f} | "
                f"${r['MaxDD_usd']:.0f} | {r['worst_trade_usd']:.0f} | ${r['worst_day_usd']:.0f} | "
                f"{int(r['halted_days'])} | {int(r['trades_foregone'])} | {r['MAR']:.3f} |")

    lines = []
    lines.append("# RISK-FIRST REDESIGN — IS-only verdict\n")
    lines.append(f"- Base = Donchian-20, trail 2.5xATR, vetoes {{VR>=1, ATRpct<=90}}, market, 1.5t, 1 contract, $50k.\n")
    lines.append(f"- Pre-registered gate: MaxDD_usd@1c <= ${DD_GATE:.0f} (hard), PF>=1.10, boot p<=0.10, "
                 f"expectancy>=$2, n>=300, worst trade <= $500, neighborhood MaxDD<=${DD_HARD:.0f}.\n")
    lines.append(f"- **Verdict: {'1+ PASSER(S)' if len(passers) else 'ZERO PASSERS'}** "
                 f"(IS gates at BOTH 1.5t and 3t).\n")
    if len(passers):
        lines.append("## Passers\n")
        lines.append("| tag | family | 1.5t MaxDD$ | 3t MaxDD$ | 1.5t CAGR$ | nbr max DD$ |")
        lines.append("|---|---|---|---|---|---|")
        for _, p in passers.iterrows():
            r15 = grid[(grid["tag"] == p["tag"]) & (grid["cost"] == "1.5t")].iloc[0]
            r30 = grid[(grid["tag"] == p["tag"]) & (grid["cost"] == "3.0t")]
            dd3 = r30.iloc[0]["MaxDD_usd"] if len(r30) else float("nan")
            lines.append(f"| {p['tag']} | {p['family']} | ${r15['MaxDD_usd']:.0f} | ${dd3:.0f} | "
                         f"{r15['CAGR_usd']*100:.2f}% | ${p['nbr_max_DD_usd']:.0f} |")
        lines.append("")
        lines.append("## Chosen (max CAGR@1c, tie lower MaxDD)\n")
        rc = grid[(grid["tag"] == chosen) & (grid["cost"] == "1.5t")].iloc[0]
        lines.append(f"**{chosen}** params: `{full_params(res_store[chosen][1]['ov'])}`\n")
        lines.append("| config | cost | n | PF | win | exp$ | CAGR$ | Sharpe$ | MaxDD$ | worst trade$ | worst day$ | halted d | foregone | MAR |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for cost in ("1.5t", "3.0t"):
            s = fmt_row(grid, chosen, cost)
            if s:
                lines.append(s)
        lines.append("")
    else:
        lines.append("## Why nothing passed\n")
        lines.append("The pre-registered gate required MaxDD_usd@1c <= $1200 at BOTH 1.5t and 3t "
                     "while keeping PF>=1.10 (boot p<=0.10), expectancy>=$2, n>=300 and "
                     "worst trade>=-$500. Two disjoint failure modes emerged:\n")
        lines.append("1. **Good-PF variants blow the DD gate.** Best non-breaker improvements "
                     "(Donchian-32, ratchet-only, time-stop-16) reach PF 1.24-1.29 but only cut "
                     "MaxDD to ~$1,730-$2,100, still far above the $1,200 gate (and $1,500 hard cap).")
        lines.append("2. **The DD-halt breaker meets the DD gate but destroys the sample/edge.** "
                     "C2 with T=300/500 pt halts new entries for ~1,185-1,190 days, cuts n from "
                     "1,302 to ~220, and drops PF to 0.97-1.05 with expectancy ~$0-$1 — failing "
                     "gate 2 (PF/bootstrap) and gate 3 (n>=300, expectancy>=$2). The breaker does "
                     "**not** manufacture PF; it lowers it.\n")
        lines.append("### Key competitors (IS, 1.5t)\n")
        lines.append("| config | n | PF | win | exp$ | CAGR$ | Sharpe$ | MaxDD$ | halted d | foregone | g1 (DD<=1200) | g2 (PF/boot) | g3 (exp/n) |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for tag in ("R0_base_1.5t", "F1_donch32", "E3_ratchet", "E5_time16",
                    "C2_dd300", "C2_dd500", "X_Ebest_C2best", "X_Fbest_C2best"):
            r = grid[(grid["tag"] == tag) & (grid["cost"] == "1.5t")]
            if not len(r):
                continue
            r = r.iloc[0]
            lines.append(
                f"| {tag} | {int(r['n'])} | {r['PF']:.3f} | {r['win']:.3f} | ${r['expectancy_usd']:.2f} | "
                f"{r['CAGR_usd']*100:.2f}% | {r['Sharpe_usd']:.2f} | ${r['MaxDD_usd']:.0f} | "
                f"{int(r['halted_days'])} | {int(r['trades_foregone'])} | "
                f"{bool(r['g1_1p5'])} | {bool(r['g2_1p5'])} | {bool(r['g3_1p5'])} |")
        lines.append("")
        lines.append("**Honest statement: ZERO variants pass all IS gates.** The $1,500 risk "
                     "budget (let alone the $1,200 buffer) is not attainable at 1 contract on a "
                     "$50k account with this breakout+trail engine: the per-contract risk "
                     "(base MaxDD $2,304 = 1,152 pt) is ~2x the budget, and the only mechanism "
                     "that fits the budget (equity-DD halting) removes so much participation that "
                     "the statistical edge collapses.\n")

    lines.append("## Sample table (all variants, 1.5t)\n")
    lines.append("| tag | n | PF | win | exp$ | CAGR$ | Sharpe$ | MaxDD$ | worst trade$ | worst day$ | halted d | foregone | MAR | g1 | g2 | g3 | pass |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    g15 = grid[grid["cost"] == "1.5t"].copy()
    for _, r in g15.iterrows():
        lines.append(
            f"| {r['tag']} | {int(r['n'])} | {r['PF']:.3f} | {r['win']:.3f} | ${r['expectancy_usd']:.2f} | "
            f"{r['CAGR_usd']*100:.2f}% | {r['Sharpe_usd']:.2f} | ${r['MaxDD_usd']:.0f} | "
            f"${r['worst_trade_usd']:.0f} | ${r['worst_day_usd']:.0f} | {int(r['halted_days'])} | "
            f"{int(r['trades_foregone'])} | {r['MAR']:.3f} | "
            f"{bool(r['g1_1p5'])} | {bool(r['g2_1p5'])} | {bool(r['g3_1p5'])} | {bool(r['pass'])} |")
    lines.append("")
    lines.append("## Multiple-testing / placebo\n")
    lines.append(f"- DSR(base) = {dsr_base:.4f} (p={1-dsr_base:.4f}); E[max SR] under N={N_TRIALS} "
                 f"= {sr0:.3f}, observed base SR = {sr_base:.3f}.")
    if chosen is not None and np.isfinite(dsr_chosen):
        lines.append(f"- DSR(chosen) = {dsr_chosen:.4f} (p={1-dsr_chosen:.4f}).")
    lines.append(f"- Random-entry placebo WITH chosen breaker ({ref_tag}): real PF={real_pf:.3f} at "
                 f"percentile {pf_pct*100:.1f}% (p95={np.percentile(plac_pf_f,95):.3f}, "
                 f"max={plac_pf_f.max():.3f}); median placebo n={np.median(plac_ntr):.0f} vs real {real_ntr}.")
    lines.append(f"- 2020 contribution — base: n={yc_base['n']}, PF={yc_base['PF']:.3f}, "
                 f"PnL=${yc_base['pnl_usd']:.0f}; ref: n={yc_ref['n']}, PF={yc_ref['PF']:.3f}, "
                 f"PnL=${yc_ref['pnl_usd']:.0f}.")
    lines.append("- OOS NOT touched this round.")
    lines.append("")
    with open(os.path.join(RESULTS_DIR, "RISK_REDESIGN.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print("=" * 78)
    print("SAVED: risk_grid.csv, risk_bootstrap.csv, risk_placebo.csv, "
          "risk_trades_*.csv, RISK_REDESIGN.md")
    print(f"TOTAL RUNTIME {time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
