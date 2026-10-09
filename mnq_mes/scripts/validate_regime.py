"""Validation report + plots for the market-regime indicator.

For every CSV in data/raw: runs compute_regime + signal_events and prints a
validation report (regime counts/%, persistence, signal counts, NaN check,
state transitions). Saves 3 PNGs to data/plots:
  1. mnq_last180_regime.png  — MNQ last 180 bars: candles + regime shading +
                               Donchian channel + EMAs + signal markers
  2. mnq_full_regime.png     — MNQ full series: state / span_atr / ADX
  3. mes_full_regime.png     — MES full series: state / span_atr / ADX

Usage:
    & "D:\\freqtrade\\.venv\\Scripts\\python.exe" mnq_mes/scripts/validate_regime.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mrind import (  # noqa: E402
    RegimeParams,
    compute_regime,
    signal_events,
    warmup_bars,
)

RAW_DIR = ROOT / "data" / "raw"
PLOTS_DIR = ROOT / "data" / "plots"

INDICATOR_COLS = [
    "atr", "adx", "ema_fast", "ema_slow", "don_hi", "don_lo",
    "bb_mid", "bb_hi", "bb_lo", "rsi", "span_atr",
]
ALL_TAGS = [
    "RANGE_LONG", "RANGE_SHORT", "TREND_LONG", "TREND_SHORT",
    "BRKOUT_LONG", "BRKOUT_SHORT",
]
REGIME_COLORS = {"TREND_UP": "#2ca02c", "TREND_DOWN": "#d62728", "RANGE": "#9e9e9e"}
TAG_COLORS = {
    "RANGE_LONG": "#2ca02c",
    "TREND_LONG": "#17becf",
    "BRKOUT_LONG": "#000000",
    "RANGE_SHORT": "#ff9800",
    "TREND_SHORT": "#e377c2",
    "BRKOUT_SHORT": "#8c564b",
}


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def effective_warmup(sig: pd.DataFrame) -> int:
    """First row index where all indicator columns are finite (data-driven)."""
    ind = sig[INDICATOR_COLS]
    ok = ind.notna().all(axis=1)
    if not ok.any():
        return len(sig)
    return int(ok.idxmax())


def build_report(df: pd.DataFrame, name: str) -> str:
    p = RegimeParams()
    sig = signal_events(compute_regime(df, p), p)
    n = len(sig)
    lines: list[str] = []
    add = lines.append

    add(f"=== {name} ===")
    add(f"bars total: {n}  (spec warmup = {warmup_bars(p)} bars)")

    # ---- regime distribution ---- #
    add("regime bars (incl. warmup; warmup rows are forced RANGE):")
    counts = sig["state_label"].value_counts()
    for label in ("TREND_UP", "TREND_DOWN", "RANGE"):
        cnt = int(counts.get(label, 0))
        add(f"  {label:11s}: {cnt:6d}  ({100.0 * cnt / n:5.1f}%)")

    # ---- persistence: mean consecutive-run length per regime ---- #
    state = sig["state"].to_numpy()
    boundaries = np.where(state[1:] != state[:-1])[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [n]])
    runs = pd.DataFrame({"state": state[starts], "len": ends - starts})
    means = runs.groupby("state")["len"].mean()
    add("persistence (mean consecutive-run length, bars):")
    for label, st in (("TREND_UP", 1), ("TREND_DOWN", -1), ("RANGE", 0)):
        add(f"  {label:11s}: {float(means.get(st, np.nan)):8.2f}")

    # ---- signals ---- #
    counts = sig["sig"].value_counts()
    add("signal counts per tag:")
    for tag in ALL_TAGS:
        add(f"  {tag:14s}: {int(counts.get(tag, 0))}")

    # ---- NaN check beyond effective warmup ---- #
    eff = effective_warmup(sig)
    tail = sig[INDICATOR_COLS].iloc[eff:]
    nan_tail = int(tail.isna().sum().sum())
    # don_hi_prev/don_lo_prev are NaN exactly one extra row (the shift boundary)
    prev_tail = sig[["don_hi_prev", "don_lo_prev"]].iloc[eff + 1:]
    nan_prev = int(prev_tail.isna().sum().sum())
    if nan_tail == 0 and nan_prev == 0:
        add(f"NaN check: PASS — effective warmup = {eff} bars; 0 NaN in indicators after it")
    else:
        add(
            f"NaN check: FAIL — effective warmup = {eff} bars; "
            f"{nan_tail} NaN in indicators, {nan_prev} NaN in prev-channel after warmup"
        )

    # ---- state transitions ---- #
    if eff > 0:
        transitions = int(np.sum(state[eff:] != state[eff - 1:-1]))
    else:
        transitions = int(np.sum(state[1:] != state[:-1]))
    add(f"state-transition count (after effective warmup): {transitions}")
    add("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# matplotlib availability
# --------------------------------------------------------------------------- #
def ensure_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401

        return True
    except Exception:
        pass
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--no-input", "matplotlib"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        import matplotlib  # noqa: F401

        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _compute(df: pd.DataFrame) -> pd.DataFrame:
    p = RegimeParams()
    return signal_events(compute_regime(df, p), p)


def plot_last180(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    sig = _compute(df).tail(180).reset_index(drop=True)
    dates = pd.to_datetime(sig["Date"])
    x = mdates.date2num(dates)
    o = sig["Open"].to_numpy(dtype=float)
    hi = sig["High"].to_numpy(dtype=float)
    lo = sig["Low"].to_numpy(dtype=float)
    cl = sig["Close"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(15, 8))
    up = cl >= o
    ax.vlines(x, lo, hi, color="#9e9e9e", linewidth=0.8, zorder=2)
    ax.bar(
        x,
        np.abs(cl - o),
        bottom=np.minimum(o, cl),
        width=0.65,
        color=np.where(up, "#26a69a", "#ef5350"),
        zorder=3,
    )

    # regime shading (contiguous runs)
    labels = sig["state_label"].to_numpy()
    i, m = 0, len(sig)
    while i < m:
        j = i
        while j + 1 < m and labels[j + 1] == labels[i]:
            j += 1
        ax.axvspan(x[i], x[j], color=REGIME_COLORS[labels[i]], alpha=0.12, zorder=1)
        i = j + 1

    # Donchian channel + EMAs
    ax.fill_between(
        x, sig["don_lo"], sig["don_hi"], alpha=0.10, color="#ff9800", zorder=1, label="Donchian 20"
    )
    ax.plot(x, sig["don_hi"], color="#ff9800", lw=0.8, ls="--", alpha=0.9)
    ax.plot(x, sig["don_lo"], color="#ff9800", lw=0.8, ls="--", alpha=0.9)
    ax.plot(x, sig["ema_fast"], color="#1565c0", lw=1.1, label="EMA 9")
    ax.plot(x, sig["ema_slow"], color="#6a1b9a", lw=1.3, label="EMA 21")

    # signal markers
    for tag, color in TAG_COLORS.items():
        m_ = (sig["sig"] == tag).to_numpy()
        if not m_.any():
            continue
        if tag.endswith("LONG"):
            ax.scatter(x[m_], lo[m_] * 0.996, marker="^", s=110, color=color,
                       edgecolors="black", linewidths=0.4, label=tag, zorder=5)
        else:
            ax.scatter(x[m_], hi[m_] * 1.004, marker="v", s=110, color=color,
                       edgecolors="black", linewidths=0.4, label=tag, zorder=5)

    ax.legend(loc="upper left", fontsize=8, ncols=2)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    ax.set_title("MNQ=F last 180 bars — regime shading + Donchian + EMAs + signals")
    ax.grid(alpha=0.25, zorder=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_full(df: pd.DataFrame, name: str, out_path: Path) -> None:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    sig = _compute(df)
    dates = pd.to_datetime(sig["Date"])

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    ax = axes[0]
    ax.step(dates, sig["state"], where="post", color="#1f77b4", lw=1.0)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["TREND_DOWN", "RANGE", "TREND_UP"])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("state")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(dates, sig["span_atr"], color="#ff9800", lw=0.9)
    ax.axhline(1.0, color="k", lw=0.6, ls="--")
    ax.set_ylabel("span_atr")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(dates, sig["adx"], color="#d62728", lw=0.9)
    ax.axhline(20.0, color="k", lw=0.6, ls="--")
    ax.set_ylabel("ADX")
    ax.grid(alpha=0.3)

    axes[0].set_title(f"{name} — full series: state / span_atr / ADX")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="Regime indicator validation report + plots.")
    parser.add_argument("--raw", default=str(RAW_DIR), help="directory with CSVs")
    parser.add_argument("--plots", default=str(PLOTS_DIR), help="output dir for PNGs")
    args = parser.parse_args()
    raw_dir = Path(args.raw)
    plots_dir = Path(args.plots)
    plots_dir.mkdir(parents=True, exist_ok=True)

    csvs = sorted(raw_dir.glob("*.csv"))
    if not csvs:
        print(f"no CSVs found in {raw_dir}", file=sys.stderr)
        return 1

    frames: dict[str, pd.DataFrame] = {}
    for path in csvs:
        df = pd.read_csv(path)
        frames[path.stem] = df
        print(build_report(df, path.stem))

    if not ensure_matplotlib():
        print("matplotlib unavailable — skipping plots", file=sys.stderr)
        return 0

    import matplotlib

    matplotlib.use("Agg")

    if "MNQ=F" in frames:
        plot_last180(frames["MNQ=F"], plots_dir / "mnq_last180_regime.png")
        plot_full(frames["MNQ=F"], "MNQ=F", plots_dir / "mnq_full_regime.png")
        print("saved: mnq_last180_regime.png, mnq_full_regime.png")
    else:
        print("MNQ=F.csv not found — skipping MNQ plots", file=sys.stderr)

    if "MES=F" in frames:
        plot_full(frames["MES=F"], "MES=F", plots_dir / "mes_full_regime.png")
        print("saved: mes_full_regime.png")
    else:
        print("MES=F.csv not found — skipping MES plot", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
