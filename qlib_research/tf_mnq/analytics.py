"""Full performance analytics for IN-SAMPLE strategy evaluation.

Consumes a normalised equity curve (base ~1.0) on the 15min decision index plus
a trade ledger and produces the returns/risk, trade, z-score and curve-quality
blocks requested for the 15min run. No OOS data is ever touched here: callers
pass already-frozen IS frames.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def _eq(equity) -> pd.Series:
    if isinstance(equity, pd.Series):
        return equity.astype(float)
    return pd.Series(np.asarray(equity, dtype=float))


def drawdown(equity) -> np.ndarray:
    eq = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    with np.errstate(divide="ignore", invalid="ignore"):
        return eq / peak - 1.0


def max_dd_duration_days(equity, index) -> float:
    """Longest continuous underwater stretch, in calendar days."""
    eq = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    underwater = eq < peak - 1e-15
    idx = pd.DatetimeIndex(index)
    best = 0.0
    start = None
    for i, u in enumerate(underwater):
        if u and start is None:
            start = i
        elif (not u) and start is not None:
            best = max(best, (idx[i] - idx[start]).days)
            start = None
    if start is not None:
        best = max(best, (idx[-1] - idx[start]).days)
    return float(best)


def returns_metrics(equity, ppy: float, index=None) -> dict:
    eq = np.asarray(equity, dtype=float)
    n = len(eq)
    start, end = float(eq[0]), float(eq[-1])
    total_return = end / start - 1.0 if start else np.nan
    years = n / ppy if ppy else np.nan
    if start > 0 and end > 0 and years and years > 0:
        cagr = (end / start) ** (1.0 / years) - 1.0
    else:
        cagr = np.nan
    ret = np.diff(eq) / eq[:-1] if n > 1 else np.array([])
    std = ret.std(ddof=1) if ret.size > 1 else np.nan
    ann_vol = std * np.sqrt(ppy) if np.isfinite(std) else np.nan
    sharpe = (ret.mean() / std * np.sqrt(ppy)) if (np.isfinite(std) and std > 0) else np.nan
    dneg = ret[ret < 0]
    dstd = dneg.std(ddof=1) if dneg.size > 1 else np.nan
    sortino = (ret.mean() / dstd * np.sqrt(ppy)) if (np.isfinite(dstd) and dstd > 0) else np.nan
    dd = drawdown(eq)
    max_dd = float(-dd.min()) if n else np.nan
    calmar = cagr / max_dd if (np.isfinite(cagr) and max_dd > 0) else np.nan
    mar = calmar
    ulcer = float(np.sqrt(np.mean(dd ** 2))) if n else np.nan
    time_underwater = float(np.mean(dd < -1e-15)) if n else np.nan
    recovery_factor = total_return / max_dd if max_dd > 0 else np.nan
    return {
        "total_return": total_return,
        "CAGR": cagr,
        "ann_vol": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "MAR": mar,
        "Ulcer": ulcer,
        "MaxDD": max_dd,
        "max_dd_duration_days": (max_dd_duration_days(eq, index) if index is not None else np.nan),
        "time_underwater": time_underwater,
        "recovery_factor": recovery_factor,
    }


def trades_metrics(trades, n_bars: int, ppy: float) -> dict:
    n_tr = len(trades)
    if not n_tr:
        return {"n_trades": 0, "win_rate": np.nan, "profit_factor": np.nan,
                "expectancy_pts": np.nan, "expectancy_usd": np.nan, "expectancy_R": np.nan,
                "avg_win_usd": np.nan, "avg_loss_usd": np.nan, "avg_win_pts": np.nan,
                "avg_loss_pts": np.nan, "time_in_market": 0.0, "turnover_trades_per_year": 0.0}
    pnl_usd = np.array([t["pnl_usd_net"] for t in trades], dtype=float)
    pnl_pts = np.array([t["pnl_pts"] for t in trades], dtype=float)
    wins, losses = pnl_usd[pnl_usd > 0], pnl_usd[pnl_usd < 0]
    pf = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else np.inf
    atr_e = np.array([t.get("atr_entry", np.nan) for t in trades], dtype=float)
    loss_pts = np.abs(pnl_pts[pnl_pts < 0])
    avg_loss_pts = float(loss_pts.mean()) if loss_pts.size else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(np.isfinite(atr_e) & (atr_e > 0), atr_e, avg_loss_pts)
        valid = np.isfinite(r) & (r > 0)
        expectancy_r = float(np.mean(pnl_pts[valid] / r[valid])) if valid.any() else np.nan
    bars_held = np.array([t.get("bars_held", 0) for t in trades], dtype=float)
    years = n_bars / ppy if ppy else np.nan
    return {
        "n_trades": int(n_tr),
        "win_rate": float(wins.size / n_tr),
        "profit_factor": pf,
        "expectancy_pts": float(pnl_pts.mean()),
        "expectancy_usd": float(pnl_usd.mean()),
        "expectancy_R": expectancy_r,
        "avg_win_usd": float(wins.mean()) if wins.size else np.nan,
        "avg_loss_usd": float(losses.mean()) if losses.size else np.nan,
        "avg_win_pts": float(pnl_pts[pnl_pts > 0].mean()) if (pnl_pts > 0).any() else np.nan,
        "avg_loss_pts": float(pnl_pts[pnl_pts < 0].mean()) if (pnl_pts < 0).any() else np.nan,
        "time_in_market": float(bars_held.sum() / n_bars) if n_bars else np.nan,
        "turnover_trades_per_year": float(n_tr / years) if years and years > 0 else np.nan,
    }


def beta_alpha_ir(equity, bench_equity, ppy: float) -> dict:
    s = _eq(equity)
    b = _eq(bench_equity)
    rs = s.pct_change()
    rb = b.pct_change()
    df = pd.concat([rs.rename("s"), rb.rename("b")], axis=1).dropna()
    if len(df) < 10 or df["b"].var() == 0:
        return {"beta_vs_B0": np.nan, "alpha_ann_vs_B0": np.nan, "info_ratio_vs_B0": np.nan}
    beta = float(df["s"].cov(df["b"]) / df["b"].var())
    alpha_ann = float((df["s"].mean() - beta * df["b"].mean()) * ppy)
    ex = df["s"] - df["b"]
    ir = float(ex.mean() / ex.std(ddof=1) * np.sqrt(ppy)) if ex.std(ddof=1) > 0 else np.nan
    return {"beta_vs_B0": beta, "alpha_ann_vs_B0": alpha_ann, "info_ratio_vs_B0": ir}


# --------------------------------------------------------------------------- #
# z-scores
# --------------------------------------------------------------------------- #
def monthly_returns(equity) -> pd.Series:
    eq = _eq(equity)
    m = eq.resample("ME").last()
    return m.pct_change().dropna()


def monthly_z(equity) -> pd.Series:
    mr = monthly_returns(equity)
    if mr.std(ddof=1) == 0 or len(mr) < 2:
        return pd.Series(0.0, index=mr.index)
    return (mr - mr.mean()) / mr.std(ddof=1)


def rolling_return_z(equity, window: int) -> pd.Series:
    window = int(round(window))
    eq = _eq(equity)
    roll = eq / eq.shift(window) - 1.0
    exp_mean = roll.expanding(min_periods=window).mean()
    exp_std = roll.expanding(min_periods=window).std()
    return (roll - exp_mean) / exp_std


def drawdown_z(equity) -> pd.Series:
    eq = _eq(equity)
    dd = pd.Series(drawdown(eq), index=eq.index)
    std = dd.std(ddof=1)
    return (dd - dd.mean()) / std if std and std > 0 else pd.Series(0.0, index=eq.index)


# --------------------------------------------------------------------------- #
# curve quality ("steadily rising")
# --------------------------------------------------------------------------- #
def curve_quality(equity, ppy: float) -> dict:
    eq = _eq(equity)
    n = len(eq)
    arr = eq.to_numpy(dtype=float)
    t = np.arange(n, dtype=float)
    logeq = np.log(np.clip(arr, 1e-12, None))
    if n > 2 and np.ptp(t) > 0:
        slope, intercept = np.polyfit(t, logeq, 1)
        yhat = slope * t + intercept
        ss_res = float(((logeq - yhat) ** 2).sum())
        ss_tot = float(((logeq - logeq.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    else:
        r2 = np.nan
    mr = monthly_returns(eq)
    window_bars = int(round(ppy))
    roll = (eq / eq.shift(window_bars) - 1.0).dropna()
    ac = eq.pct_change().autocorr(1)
    return {
        "r2_log_equity_vs_time": r2,
        "pct_positive_months": float((mr > 0).mean()) if len(mr) else np.nan,
        "best_month": float(mr.max()) if len(mr) else np.nan,
        "worst_month": float(mr.min()) if len(mr) else np.nan,
        "pct_positive_rolling_12m": float((roll > 0).mean()) if len(roll) else np.nan,
        "worst_12m_return": float(roll.min()) if len(roll) else np.nan,
        "equity_curve_autocorr": float(ac) if np.isfinite(ac) else np.nan,
    }


def smoothness_ranks(calmar, sortino, r2) -> np.ndarray:
    """Mean percentile rank of Calmar/Sortino/log-equity-R² across the grid."""
    from scipy.stats import rankdata

    arrs = []
    for a in (calmar, sortino, r2):
        a = np.asarray(a, dtype=float)
        filled = np.where(np.isfinite(a), a, -np.inf)
        arrs.append(rankdata(filled) / len(filled))
    return np.nanmean(np.vstack(arrs), axis=0)


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def plot_equity_dd_monthly(equity, out_path: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eq = _eq(equity)
    dd = pd.Series(drawdown(eq), index=eq.index)
    mr = monthly_returns(eq)
    hm = pd.DataFrame({"y": mr.index.year, "m": mr.index.month, "r": mr.values})
    pivot = hm.pivot_table(index="y", columns="m", values="r", aggfunc="last")
    vmax = float(np.nanmax(np.abs(pivot.to_numpy()))) if pivot.size else 0.01
    vmax = vmax if vmax > 0 else 0.01

    fig = plt.figure(figsize=(12, 11))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 1.1, 1.5], hspace=0.35)
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(eq.index, eq.to_numpy(), color="#1f77b4", lw=1.1)
    ax0.set_title(f"{title} - equity (IS, <=2023-12-31)")
    ax0.grid(alpha=0.25)

    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.fill_between(dd.index, dd.to_numpy() * 100.0, 0.0, color="#d62728", alpha=0.45)
    ax1.set_title("drawdown %")
    ax1.grid(alpha=0.25)

    ax2 = fig.add_subplot(gs[2])
    im = ax2.imshow(pivot.to_numpy(), aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax2.set_yticks(range(len(pivot.index)))
    ax2.set_yticklabels([str(y) for y in pivot.index])
    ax2.set_xticks(range(len(pivot.columns)))
    ax2.set_xticklabels([str(m) for m in pivot.columns])
    ax2.set_title("monthly returns heatmap")
    fig.colorbar(im, ax=ax2, format="%.1f%%")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.to_numpy()[i, j]
            if np.isfinite(v):
                ax2.text(j, i, f"{v*100:.1f}", ha="center", va="center", fontsize=6)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
