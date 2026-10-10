import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, pandas as pd
from tf_mnq.data import load_bars
from tf_mnq.features import compute_features
from tf_mnq.strategies import strategy_frame, veto_blocked
from tf_mnq.engine import CostCfg
from tf_mnq.breakout import run_breakout_ex

IS_START, IS_END = "2019-05-06", "2023-12-31"
OOS_START, OOS_END = "2024-01-01", "2026-10-02"
COST = CostCfg(slip_ticks=1.5, commission_per_side=0.75, tick=0.25, point_value=2.0)
FREQ = "15min"

allb = load_bars(FREQ)
print("FULL range", allb.index.min(), allb.index.max(), len(allb))
print("cap first close", allb["close"].iloc[0]*2, "last", allb["close"].iloc[-1]*2)

feat_all = compute_features(allb)
sf_all = strategy_frame(allb, feat_all)

def slice_all(lo, hi):
    b = allb.loc[(allb.index>=lo)&(allb.index<=hi)]
    f = feat_all.reindex(b.index)
    s = sf_all.reindex(b.index)
    return b, f, s

bars, feat, sf = slice_all(IS_START, IS_END)
print("IS bars", len(bars), bars.index.min(), bars.index.max())
ppy = len(bars)/((bars.index[-1]-bars.index[0]).days/365.25)
print("ppy", ppy)

# reconstruct frozen mask from stageB csv
sb = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"results","stageB_added_vetoes.csv"))
fseries = {
    "VR": feat["vr"], "ATRpct": sf["atr_pct"], "ER20": feat["er"],
    "extension": sf["ext"],
    "pullback_depth": (bars["high"].rolling(10, min_periods=10).max()-bars["low"])/feat["atr"],
    "hour": pd.Series(bars.index.hour, index=bars.index),
    "dow": pd.Series(bars.index.dayofweek, index=bars.index),
}
mask = pd.Series(False, index=bars.index)
for _, r in sb.iterrows():
    fs = fseries[r["feature"]]
    if r["mask_kind"] == "eq":
        mask = mask | (fs == r["val"])
    else:
        mask = mask | ((fs > r["lo"]) & (fs <= r["hi"]))
base = veto_blocked(bars, feat, sf, components=["vol","vr"])
blocked = base | mask
print("blocked count", int(blocked.sum()), "base", int(base.sum()))

t0=time.perf_counter()
res = run_breakout_ex(bars, feat["atr"], COST, donch=20, stop_mult=2.5, veto_long=blocked)
print("engine runtime", time.perf_counter()-t0)
tr = res["trades"]
pnl = np.array([t["pnl_usd_net"] for t in tr])
w=pnl[pnl>0].sum(); l=-pnl[pnl<0].sum()
print("n", len(tr), "PF", w/l, "win", (pnl>0).mean(), "capital", res["capital"])
eq = res["equity"].to_numpy()
peak=np.maximum.accumulate(eq); dd=eq/peak-1; print("MaxDD_norm", -dd.min())
print("eq last", eq[-1])
