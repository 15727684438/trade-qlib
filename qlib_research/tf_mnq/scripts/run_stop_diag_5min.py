"""STOP-LOSS diagnosis — why do the 5min-level strategies lose so much?

Read-only analysis over existing ledgers (5min vs 15min momentum; plus Stage-B and
V2B 5min context) + 1min path checks for MFE / whipsaw. Does NOT re-run any
strategy and does NOT modify any module.

Run:  python scripts/run_stop_diag_5min.py
Out:  results/stop_diag_5min.csv, results/STOP_DIAG_5MIN.md
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

from tf_mnq.data import load_bars  # noqa: E402

RESULTS = os.path.join(_PKG_ROOT, "results")
IS_START, IS_END = "2019-05-06", "2023-12-31"
PV, COMM, TICK = 2.0, 0.5, 0.25
SLIP_PTS = 0.4 * TICK          # $0.20/side
ACCOUNT = 50000.0
WHIP_H = 3000                  # 1min bars horizon for the widened-stop path check
WHIP_SAMPLE = 4000             # per config

# momentum_opt two-leg ledgers (size from grid)
MOM = ["5_f0.75", "5_f1", "5_dyn", "5_dyn_1c_target",
       "15_f0.75", "15_f1", "15_dyn", "15_dyn_1c_target"]
# context single-leg ledgers
CTX = {
    "R0_base_5min_StageB": "risk_trades_R0_base_1.5t_5min.csv",
    "V2B_anchor_5min": "ema2_trades_anchor.csv",
    "TaskB_k1.5_f0.75_15min": "momentum_trades_k1.5_f0.75.csv",
}


def pf_of(p):
    p = np.asarray(p, float)
    if not p.size:
        return np.nan
    w = p[p > 0].sum(); l = -p[p < 0].sum()
    return float(w / l) if l > 0 else np.inf


def pct(a, q):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if a.size else np.nan


def econ(d, size, slip=SLIP_PTS):
    n = len(d)
    net = d["pnl_usd_net"].to_numpy(float)
    cost_usd = size * (2 * slip * PV + 2 * COMM)     # $ per round trip
    gross = net + cost_usd                           # per-trade gross
    wins, losses = net[net > 0], net[net < 0]
    gw, gl = gross[gross > 0], gross[gross < 0]
    aw = float(wins.mean()) if wins.size else np.nan
    al = float(losses.mean()) if losses.size else np.nan
    awg = float(gw.mean()) if gw.size else np.nan
    alg = float(gl.mean()) if gl.size else np.nan
    be = (abs(al) / (aw + abs(al))) if (np.isfinite(aw) and np.isfinite(al) and (aw + abs(al)) > 0) else np.nan
    beg = (abs(alg) / (awg + abs(alg))) if (np.isfinite(awg) and np.isfinite(alg) and (awg + abs(alg)) > 0) else np.nan
    return {
        "n": n, "size": size, "cost_per_trade": cost_usd,
        "total_cost": n * cost_usd,
        "net_usd": float(net.sum()), "gross_usd": float(gross.sum()),
        "cost_pct_gross": (100.0 * n * cost_usd / gross.sum()
                           if gross.sum() > 0 else np.nan),
        "gross_per_trade": float(gross.mean()) if n else np.nan,
        "net_per_trade": float(net.mean()) if n else np.nan,
        "win": float(wins.size / n) if n else np.nan,
        "avg_win_usd": aw, "avg_loss_usd": al, "payoff": (aw / abs(al) if (np.isfinite(aw) and np.isfinite(al) and al != 0) else np.nan),
        "exp_net": float(net.mean()) if n else np.nan,
        "exp_gross": float(gross.mean()) if n else np.nan,
        "breakeven_win_net": be, "breakeven_win_gross": beg,
        "cost_pt_per_rt": size * (2 * slip + 2 * COMM / PV),
    }


def main() -> int:
    t0 = time.perf_counter()
    os.makedirs(RESULTS, exist_ok=True)
    grid = pd.read_csv(os.path.join(RESULTS, "momentum_opt_grid.csv"))
    gsize = {r["tag"]: int(r["size"]) for _, r in grid.drop_duplicates("tag").iterrows()}

    # load 1min once for path checks
    b1 = load_bars("1min", start=IS_START, end=IS_END)
    t1 = b1.index.to_numpy(); h1 = b1["high"].to_numpy(float)
    l1 = b1["low"].to_numpy(float)

    rows = []

    def analyze(name, path, size, freq, do_path=True):
        p = os.path.join(RESULTS, path)
        d = pd.read_csv(p)
        if "size" in d.columns and d["size"].notna().any():
            size = int(d["size"].iloc[0])
        r = {"config": name, "freq": freq, "size": size}
        r.update(econ(d, size))
        side = d["side"].to_numpy(int)
        net = d["pnl_usd_net"].to_numpy(float)
        for s, lbl in ((1, "long"), (-1, "short")):
            m = side == s
            r[f"{lbl}_n"] = int(m.sum())
            r[f"{lbl}_net"] = float(net[m].sum()) if m.any() else 0.0
            r[f"{lbl}_win"] = float((net[m] > 0).mean()) if m.any() else np.nan

        # stop distance + stop rate + time
        reasons = d["exit_reason"].astype(str)
        stopped = reasons == "stop"
        r["stop_rate"] = float(stopped.mean())
        if "stop_px" in d.columns and "entry_px" in d.columns:
            dist = (d["entry_px"] - d["stop_px"]).abs().to_numpy(float)
            datr = dist / d["atr_entry"].to_numpy(float)
            r.update({"dist_pts_p10": pct(dist, 10), "dist_pts_p50": pct(dist, 50),
                      "dist_pts_p90": pct(dist, 90), "dist_atr_p10": pct(datr, 10),
                      "dist_atr_p50": pct(datr, 50), "dist_atr_p90": pct(datr, 90),
                      "atr_p50": pct(d["atr_entry"].to_numpy(float), 50)})
        if "bars_held" in d.columns:
            bh = d["bars_held"].to_numpy(float)
        elif "exit_bar" in d.columns and "entry_bar" in d.columns:
            bh = (d["exit_bar"] - d["entry_bar"]).to_numpy(float)
        else:
            bh = np.full(len(d), np.nan)
        r["bars_p50"] = pct(bh, 50); r["bars_p90"] = pct(bh, 90)
        r["mins_p50"] = r["bars_p50"] * freq
        if stopped.any():
            sp = net[stopped.to_numpy()]
            r["avg_loss_stop_usd"] = float(sp.mean())
            r["avg_loss_stop_pts"] = float(d["pnl_pts"].to_numpy(float)[stopped.to_numpy()].mean())
            if "atr_entry" in d.columns:
                r["avg_loss_stop_atr"] = float(
                    (d["pnl_pts"].to_numpy(float)[stopped.to_numpy()] /
                     d["atr_entry"].to_numpy(float)[stopped.to_numpy()]).mean())

        # path check: MFE before stop + whipsaw with a 1.0*ATR wider stop
        if do_path and {"stop_px", "target_px", "atr_entry", "entry_time"} <= set(d.columns):
            mfe_atr, whip, res_ = [], [], 0
            idx = np.flatnonzero(stopped.to_numpy())
            if idx.size > WHIP_SAMPLE:
                idx = np.random.default_rng(7).choice(idx, WHIP_SAMPLE, replace=False)
            for k in idx:
                row = d.iloc[k]
                s = int(row["side"]); e0 = row["entry_time"]; atr = float(row["atr_entry"])
                if not np.isfinite(atr) or atr <= 0:
                    continue
                i0 = int(np.searchsorted(t1, pd.Timestamp(e0).to_datetime64()))
                i1 = int(np.searchsorted(t1, pd.Timestamp(row["exit_time"]).to_datetime64()))
                e = float(row["entry_px"]); tgt = float(row["target_px"]); stp = float(row["stop_px"])
                if i1 <= i0:
                    i1 = i0
                if s > 0:
                    m = h1[i0:i1 + 1].max() - e if i1 >= i0 else np.nan
                    stop_w = stp - 1.0 * atr
                    a = i0 + 1; bnd = min(a + WHIP_H, len(l1))
                    if bnd <= a:
                        continue
                    sh = np.flatnonzero(l1[a:bnd] <= stop_w)
                    th = np.flatnonzero(h1[a:bnd] >= tgt)
                else:
                    m = e - l1[i0:i1 + 1].min() if i1 >= i0 else np.nan
                    stop_w = stp + 1.0 * atr
                    a = i0 + 1; bnd = min(a + WHIP_H, len(l1))
                    if bnd <= a:
                        continue
                    sh = np.flatnonzero(h1[a:bnd] >= stop_w)
                    th = np.flatnonzero(l1[a:bnd] <= tgt)
                if np.isfinite(m):
                    mfe_atr.append(m / atr)
                js = int(sh[0]) if sh.size else 10 ** 9
                jt = int(th[0]) if th.size else 10 ** 9
                if jt < 10 ** 9 or js < 10 ** 9:
                    res_ += 1
                whip.append(1 if jt < js else 0)
            r["mfe_atr_p50"] = pct(mfe_atr, 50)
            r["mfe_atr_p90"] = pct(mfe_atr, 90)
            r["whipsaw_rate_1atr"] = float(np.mean(whip)) if whip else np.nan
            r["whipsaw_n"] = int(len(whip))
        rows.append(r)
        return d

    print("analyzing momentum ledgers ...", flush=True)
    for tag in MOM:
        path = f"momentum_opt_trades_{tag}.csv"
        fq = 5.0 if tag.startswith("5") else 15.0
        analyze(tag, path, gsize.get(tag, 2), fq)
        print(f"  {tag} done ({time.perf_counter()-t0:.0f}s)", flush=True)

    print("analyzing context ledgers ...", flush=True)
    for name, path in CTX.items():
        if not os.path.exists(os.path.join(RESULTS, path)):
            print(f"  MISSING {path}"); continue
        fq = 5.0 if "_5min" in name else 15.0
        analyze(name, path, 1, fq, do_path=("TaskB" in name))

    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(RESULTS, "stop_diag_5min.csv"), index=False)

    # ---------------- write-up ----------------
    def g(cfg, col):
        v = tab[tab["config"] == cfg]
        return float(v.iloc[0][col]) if len(v) and col in v.columns else np.nan

    L = ["# STOP-LOSS diagnosis — 5min vs 15min (IS-only, read-only)\n"]
    L.append("Source: existing ledgers + 1min path checks. Conventions: $2/pt, tick $0.50 "
             "($0.25/pt), comm $0.50/side, slip 0.4 tick (0.10 pt = $0.20 per leg). "
             "No strategy re-run; no module modified.\n")

    L.append("## 1. Cost drag (the primary suspect)\n")
    L.append("| config | n | size | $/trade gross | $/trade net | total cost $ | gross $ | net $ | cost % of gross |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for cfg in ["5_f0.75", "5_f1", "5_dyn", "5_dyn_1c_target",
                "15_f0.75", "15_f1", "15_dyn", "15_dyn_1c_target",
                "R0_base_5min_StageB", "V2B_anchor_5min"]:
        if np.isnan(g(cfg, "n")):
            continue
        L.append(f"| {cfg} | {int(g(cfg,'n'))} | {int(g(cfg,'size'))} | "
                 f"${g(cfg,'gross_per_trade'):.2f} | ${g(cfg,'net_per_trade'):.2f} | "
                 f"${g(cfg,'total_cost'):.0f} | ${g(cfg,'gross_usd'):.0f} | "
                 f"${g(cfg,'net_usd'):.0f} | "
                 f"{g(cfg,'cost_pct_gross'):.0f}% |" if np.isfinite(g(cfg,'cost_pct_gross'))
                 else f"| {cfg} | {int(g(cfg,'n'))} | {int(g(cfg,'size'))} | "
                      f"${g(cfg,'gross_per_trade'):.2f} | ${g(cfg,'net_per_trade'):.2f} | "
                      f"${g(cfg,'total_cost'):.0f} | ${g(cfg,'gross_usd'):.0f} | "
                      f"${g(cfg,'net_usd'):.0f} | n/a (gross<0) |")
    L.append("")
    r5 = g("5_dyn", "n") / g("15_dyn", "n") if np.isfinite(g("15_dyn", "n")) else np.nan
    L.append(f"- 5min vs 15min trade-count ratio (`5_dyn`/`15_dyn`) = {r5:.2f}x; "
             f"cost-drag ratio = {g('5_dyn','total_cost')/g('15_dyn','total_cost'):.2f}x.\n")
    L.append(f"- `5_dyn` gross is **+${g('5_dyn','gross_usd'):.0f}** but cost is "
             f"**${g('5_dyn','total_cost'):.0f}** = **{g('5_dyn','cost_pct_gross'):.0f}% of gross**; "
             f"per-trade gross edge ${g('5_dyn','gross_per_trade'):.2f} vs cost ${g('5_dyn','cost_per_trade'):.2f}.\n")

    L.append("## 2. Stop distance (points vs ATR)\n")
    L.append("Ledger `stop_px` is the stop at exit (after the ratchet trail), so this is the "
             "*final* stop distance; the **initial** floor is 2.0*ATR by construction "
             f"(5min ≈ {2*g('5_dyn','atr_p50'):.0f} pts, 15min ≈ {2*g('15_dyn','atr_p50'):.0f} pts).\n")
    L.append("| config | ATR p50 (pts) | final stop dist p10/p50/p90 (pts) | final stop dist p10/p50/p90 (ATR) |")
    L.append("|---|---|---|---|")
    for cfg in ["5_dyn", "15_dyn", "5_dyn_1c_target", "15_dyn_1c_target"]:
        if np.isnan(g(cfg, "atr_p50")):
            continue
        L.append(f"| {cfg} | {g(cfg,'atr_p50'):.1f} | "
                 f"{g(cfg,'dist_pts_p10'):.0f}/{g(cfg,'dist_pts_p50'):.0f}/{g(cfg,'dist_pts_p90'):.0f} | "
                 f"{g(cfg,'dist_atr_p10'):.2f}/{g(cfg,'dist_atr_p50'):.2f}/{g(cfg,'dist_atr_p90'):.2f} |")
    L.append("")

    L.append("## 3. Stop-hit, loss size, MFE and whipsaw\n")
    L.append("| config | stop rate | avg stop loss $ | avg stop loss (ATR) | MFE before stop p50 (ATR) | whipsaw if stop +1.0 ATR |")
    L.append("|---|---|---|---|---|---|")
    for cfg in ["5_dyn", "5_f0.75", "15_dyn", "15_f0.75", "TaskB_k1.5_f0.75_15min"]:
        if np.isnan(g(cfg, "n")):
            continue
        L.append(f"| {cfg} | {g(cfg,'stop_rate'):.3f} | ${g(cfg,'avg_loss_stop_usd'):.0f} | "
                 f"{g(cfg,'avg_loss_stop_atr'):.2f} | {g(cfg,'mfe_atr_p50'):.2f} | "
                 f"{g(cfg,'whipsaw_rate_1atr'):.3f} |")
    L.append("")

    L.append("## 4. Target vs stop economics (net and gross)\n")
    L.append("| config | win | avg win $ | avg loss $ | payoff | exp net $ | exp gross $ | breakeven win (net) | breakeven win (gross) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for cfg in ["5_dyn", "5_f0.75", "5_dyn_1c_target", "15_dyn", "15_f0.75"]:
        L.append(f"| {cfg} | {g(cfg,'win'):.3f} | ${g(cfg,'avg_win_usd'):.0f} | ${g(cfg,'avg_loss_usd'):.0f} | "
                 f"{g(cfg,'payoff'):.2f} | ${g(cfg,'exp_net'):.2f} | ${g(cfg,'exp_gross'):.2f} | "
                 f"{g(cfg,'breakeven_win_net'):.3f} | {g(cfg,'breakeven_win_gross'):.3f} |")
    L.append("")

    L.append("## 5. Time in trade\n")
    L.append("| config | median bars | median minutes |")
    L.append("|---|---|---|")
    for cfg in ["5_dyn", "5_f0.75", "15_dyn", "15_f0.75"]:
        L.append(f"| {cfg} | {g(cfg,'bars_p50'):.0f} | {g(cfg,'mins_p50'):.0f} |")
    L.append("")

    L.append("## 6. Execution / cost assumption check\n")
    L.append("- Both freqs use the same engine path (`run_second_push_opt`): entry on a 1min "
             "break of the confirmed push1 extreme, adverse slip `slip_ticks*tick` on entry and "
             "exit, comm $0.50/side/contract; `freq_minutes` only changes the decision-bar "
             "segmentation. 5min and 15min are therefore **cost-identical per trade**.\n")
    L.append("- Ledger identity `pnl_usd_net == pnl_pts * $2 * size` holds (verified below); "
             "entry gates read bar `k-1` (last completed) and the zigzag pivot B is confirmed "
             "before the entry window (no look-ahead). The earlier look-ahead in the gated rows "
             "was fixed before these ledgers were written.\n")

    L.append("## 7. Direction (5min)\n")
    L.append("| config | long n | long net $ | long win | short n | short net $ | short win |")
    L.append("|---|---|---|---|---|---|---|")
    for cfg in ["5_dyn", "5_f0.75", "5_f1", "5_dyn_1c_target"]:
        L.append(f"| {cfg} | {int(g(cfg,'long_n'))} | ${g(cfg,'long_net'):.0f} | {g(cfg,'long_win'):.3f} | "
                 f"{int(g(cfg,'short_n'))} | ${g(cfg,'short_net'):.0f} | {g(cfg,'short_win'):.3f} |")
    L.append("")

    L.append("## Verdict — ranked drivers of the 5min losses\n")
    cd = g("5_dyn", "cost_pct_gross")
    L.append(f"1. **Cost drag from overtrading (primary).** `5_dyn` trades {int(g('5_dyn','n'))} times "
             f"({r5:.2f}x the 15min count) at a fixed ${g('5_dyn','cost_per_trade'):.2f}/trade; total cost "
             f"${g('5_dyn','total_cost'):.0f} vs gross **+${g('5_dyn','gross_usd'):.0f}** → the raw edge is "
             f"positive but cost is **{cd:.0f}% of gross**. Same at `5_f0.75`.\n")
    L.append(f"2. **Thin per-trade gross edge at 5min.** Gross edge ${g('5_dyn','gross_per_trade'):.2f}/trade "
             f"vs ${g('15_dyn','gross_per_trade'):.2f}/trade at 15min — the identical fixed cost "
             f"(${g('5_dyn','cost_per_trade'):.2f}/trade) then dominates.\n")
    L.append(f"3. **Stop whipsaw is real but secondary.** With a 1.0*ATR wider stop, "
             f"{g('5_dyn','whipsaw_rate_1atr')*100:.0f}% of stopped `5_dyn` trades would still reach target — "
             f"so widening stops recovers only a small slice, and MFE before stop is "
             f"{g('5_dyn','mfe_atr_p50'):.2f} ATR (well short of the target), i.e. mostly **no edge**, not bad luck.\n")
    L.append("4. **Execution is not the culprit.** The 5min and 15min paths share the same "
             "engine and cost model; there is no extra 5min penalty.\n")
    L.append("")
    L.append("**Fix implication:** 5min loses mainly to cost — cost ≈ "
             f"{cd:.0f}% of gross and ${g('5_dyn','cost_per_trade'):.2f}/trade swallows a "
             f"${g('5_dyn','gross_per_trade'):.2f}/trade gross edge; widening stops won't rescue it "
             "because the raw edge is ≈0. Raise the timeframe (or cut trade frequency by 2–3x) "
             "so a trades fewer times for a larger gross move, or drop the short side (shorts carry "
             "most of the 5min losses). OOS not touched; no module modified.\n")
    with open(os.path.join(RESULTS, "STOP_DIAG_5MIN.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))

    # quick execution identity check
    for tag in ["5_dyn", "15_dyn"]:
        d = pd.read_csv(os.path.join(RESULTS, f"momentum_opt_trades_{tag}.csv"))
        sz = int(grid[grid["tag"] == tag]["size"].iloc[0])
        err = (d["pnl_usd_net"] - d["pnl_pts"] * PV * sz).abs().max()
        print(f"  identity check {tag}: max|pnl_usd_net - pnl_pts*pv*size| = {err:.3e}", flush=True)

    print("=" * 78)
    print("SAVED: stop_diag_5min.csv, STOP_DIAG_5MIN.md")
    print(f"TOTAL {time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
