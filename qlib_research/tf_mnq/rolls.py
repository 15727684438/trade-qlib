"""Continuous-contract roll diagnostic for ``@MNQ#C``-style day bars.

If the series is *raw-rolled* (naive splice of front contracts) the overnight
gap ``open_t / close_{t-1} - 1`` shows large discontinuities clustered on the
quarterly roll dates. A *back-adjusted* series removes those jumps. This module
quantifies which is the case instead of guessing.

Run over IN-SAMPLE day bars only (<= 2023-12-31).
"""

from __future__ import annotations

import calendar

import numpy as np
import pandas as pd


def overnight_gaps(day_bars: pd.DataFrame) -> pd.Series:
    """Session-to-session gap ``open_t / close_{t-1} - 1``."""
    return day_bars["open"] / day_bars["close"].shift(1) - 1.0


def second_thursday(year: int, month: int) -> pd.Timestamp:
    """Date of the 2nd Thursday of ``month`` (typical CME roll anchor)."""
    thursdays = [
        week[calendar.THURSDAY]
        for week in calendar.monthcalendar(year, month)
        if week[calendar.THURSDAY] != 0
    ]
    return pd.Timestamp(year, month, int(thursdays[1]))


def quarterly_roll_dates(index: pd.DatetimeIndex) -> list:
    """2nd-Thursday roll anchors in Mar/Jun/Sep/Dec that fall inside ``index``."""
    out = []
    for year in sorted({d.year for d in index}):
        for month in (3, 6, 9, 12):
            d = second_thursday(year, month)
            if index[0] <= d <= index[-1]:
                out.append(d)
    return out


def diagnose(day_bars: pd.DataFrame, top_k: int = 15) -> dict:
    """Return a dict of roll-diagnostic numbers (also printable via ``report``)."""
    gaps = overnight_gaps(day_bars)
    idx = day_bars.index
    absg = gaps.abs().dropna()

    stats = {
        "n_sessions": int(len(day_bars)),
        "gap_std": float(gaps.std()),
        "mean_abs_gap": float(absg.mean()),
        "median_abs_gap": float(absg.median()),
    }

    top = absg.sort_values(ascending=False).head(top_k)
    top_rows = [(idx.get_loc(ts), ts, float(g), float(gaps.loc[ts])) for ts, g in top.items()]

    roll_dates = quarterly_roll_dates(idx)
    roll_rows = []
    for d in roll_dates:
        pos = int(idx.searchsorted(d))
        if pos >= len(idx) or idx[pos] != d:
            # roll anchor is not itself a session (holiday); use nearest session
            pos = int(np.argmin(np.abs((idx - d).to_numpy())))
        for off in (-1, 0, 1):
            j = pos + off
            if 0 <= j < len(idx):
                roll_rows.append((idx[j], float(gaps.iloc[j])))

    roll_vals = np.array([g for _, g in roll_rows if np.isfinite(g)], dtype=float)
    roll_abs = np.abs(roll_vals)
    ratio = float(roll_abs.mean() / stats["mean_abs_gap"]) if stats["mean_abs_gap"] > 0 else np.nan

    roll_set = {ts for ts, _ in roll_rows}
    top_in_roll = sum(1 for _, ts, _, _ in top_rows if ts in roll_set)

    stats.update(
        {
            "n_roll_dates": len(roll_dates),
            "roll_abs_mean": float(roll_abs.mean()) if roll_abs.size else np.nan,
            "roll_abs_max": float(roll_abs.max()) if roll_abs.size else np.nan,
            "roll_to_baseline_ratio": ratio,
            "top_in_roll": int(top_in_roll),
            "top_k": top_k,
        }
    )
    return {
        "stats": stats,
        "top": top_rows,
        "roll_rows": roll_rows,
        "roll_dates": roll_dates,
    }


def verdict(diag: dict) -> str:
    """Return a data-driven RAW-ROLLED vs BACK-ADJUSTED verdict string."""
    s = diag["stats"]
    ratio = s["roll_to_baseline_ratio"]
    dominant = s["top_in_roll"] >= max(3, s["top_k"] // 2)
    raw = bool(np.isfinite(ratio) and (ratio > 1.5 or dominant))
    if raw:
        return (
            f"RAW-ROLLED: roll-window |gap| x baseline = {ratio:.2f}; "
            f"{s['top_in_roll']}/{s['top_k']} largest gaps fall on roll dates "
            f"(roll |gap| max {s['roll_abs_max']:.4f}, overall median {s['median_abs_gap']:.4f})."
        )
    return (
        f"BACK-ADJUSTED: roll-window |gap| x baseline = {ratio:.2f} (~no roll jump); "
        f"{s['top_in_roll']}/{s['top_k']} largest gaps fall on roll dates "
        f"(roll |gap| max {s['roll_abs_max']:.4f}, overall median {s['median_abs_gap']:.4f})."
    )


def report(day_bars: pd.DataFrame, top_k: int = 15) -> dict:
    """Print the full roll diagnostic and return the raw dict + verdict."""
    diag = diagnose(day_bars, top_k=top_k)
    s = diag["stats"]
    print("=" * 78)
    print("ROLL DIAGNOSTIC (day bars, IS only)")
    print("=" * 78)
    print(f"sessions          : {s['n_sessions']}")
    print(f"gap std           : {s['gap_std']:.6f}")
    print(f"mean |gap|        : {s['mean_abs_gap']:.6f}")
    print(f"median |gap|      : {s['median_abs_gap']:.6f}")
    print(f"roll dates matched: {s['n_roll_dates']}")
    print(f"roll |gap| mean   : {s['roll_abs_mean']:.6f}")
    print(f"roll |gap| max    : {s['roll_abs_max']:.6f}")
    print(f"roll/baseline     : {s['roll_to_baseline_ratio']:.2f}x")
    print(f"top-{s['top_k']} on roll   : {s['top_in_roll']}")
    print(f"\nTop {top_k} |overnight gap|:")
    print(f"  {'rank':>4}  {'date':<12}  {'gap':>10}  {'abs':>10}  on_roll")
    for r, (_, ts, absg, g) in enumerate(diag["top"], start=1):
        on_roll = "yes" if ts in {t for t, _ in diag["roll_rows"]} else ""
        print(f"  {r:>4}  {ts.strftime('%Y-%m-%d'):<12}  {g:>10.6f}  {absg:>10.6f}  {on_roll}")
    print("\nQuarterly roll windows (2nd Thu +/- 1 session):")
    print(f"  {'date':<12}  {'gap':>10}")
    for ts, g in diag["roll_rows"]:
        print(f"  {ts.strftime('%Y-%m-%d'):<12}  {g:>10.6f}")
    v = verdict(diag)
    print(f"\nVERDICT: {v}")
    print("=" * 78 + "\n")
    diag["verdict"] = v
    return diag
