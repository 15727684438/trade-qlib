"""Download daily OHLCV for MES=F and MNQ=F (2019-05-06 -> today) to data/raw.

Strategy (per project brief):
  1. Try yfinance (pip install --no-input yfinance on first run). If the import
     or runtime breaks under pandas 3.0.5, fall back to:
  2. Direct Yahoo chart API:
        https://query1.finance.yahoo.com/v8/finance/chart/<SYM>?period1=<ms>&period2=<ms>&interval=1d
     with a browser User-Agent, parsing timestamp + indicators.quote[0].

Output CSVs: data/raw/<SYM>.csv with columns Date(ISO),Open,High,Low,Close,Volume.
Exit code non-zero only if BOTH methods fail.

Usage:
    & "D:\\freqtrade\\.venv\\Scripts\\python.exe" mnq_mes/scripts/download_data.py
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

DEFAULT_START = date(2019, 5, 6)
SYMBOLS = ["MES=F", "MNQ=F"]
COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)


def detect_proxy() -> str | None:
    """Return an HTTP(S) proxy URL to use, or None.

    Order: existing HTTPS_PROXY/HTTP_PROXY env vars, then common local proxy
    ports (Clash Verge/mihomo default 8888, Clash 7890/7897, v2ray 10808/10809).
    Yahoo throttles direct datacenter IPs, so a working local proxy is used
    when present.
    """
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        val = os.environ.get(var)
        if val:
            return val
    for port in (8888, 7890, 7897, 10808, 10809):
        s = socket.socket()
        s.settimeout(0.3)
        try:
            s.connect(("127.0.0.1", port))
            return f"http://127.0.0.1:{port}"
        except OSError:
            pass
        finally:
            s.close()
    return None


def apply_proxy() -> str | None:
    """Set proxy env vars (honoured by requests AND curl_cffi/yfinance)."""
    proxy = detect_proxy()
    if proxy:
        os.environ.setdefault("HTTPS_PROXY", proxy)
        os.environ.setdefault("HTTP_PROXY", proxy)
    return proxy


# --------------------------------------------------------------------------- #
# yfinance path
# --------------------------------------------------------------------------- #
def ensure_yfinance() -> bool:
    """Make sure yfinance is importable (installs it if missing)."""
    try:
        import yfinance  # noqa: F401

        return True
    except Exception:
        pass
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--no-input", "yfinance"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        import yfinance  # noqa: F401

        return True
    except Exception:
        return False


def download_yfinance(start: date, interval: str = "1d") -> dict[str, pd.DataFrame]:
    """Download all symbols via yfinance; raises on any problem.

    ``interval``: "1d" (default, returns date-only Date column) or any other
    yfinance interval ("1h", ...) which keeps tz-aware timestamps in Date.
    """
    import yfinance as yf

    frames: dict[str, pd.DataFrame] = {}
    for sym in SYMBOLS:
        df = yf.download(
            sym,
            start=start.isoformat(),
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if df is None or len(df) == 0:
            raise RuntimeError(f"yfinance returned no data for {sym}")
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            # yfinance single-ticker: level 0 = field name, level 1 = ticker
            df.columns = [col[0] for col in df.columns]
        df = df.rename(columns={df.columns[0]: "Date"})
        df = df[[c for c in COLUMNS if c in df.columns]].copy()
        df["Date"] = pd.to_datetime(df["Date"])
        if interval == "1d":
            df["Date"] = df["Date"].dt.date
            df = df[df["Date"] >= start].reset_index(drop=True)
        df = df.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)
        frames[sym] = df
    return frames


# --------------------------------------------------------------------------- #
# Direct chart API path
# --------------------------------------------------------------------------- #
def _yahoo_session() -> "requests.Session":
    """Session with a browser User-Agent and the A3 cookie Yahoo expects."""
    import requests

    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept": "application/json,text/plain,*/*"})
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("http_proxy")
    if proxy:
        s.proxies.update({"http": proxy, "https": proxy})
    try:
        s.get("https://fc.yahoo.com", timeout=15)  # sets A3 cookie (404 is normal)
    except Exception:
        pass
    return s


def _fetch_chart(session, sym: str, start: date, today: date, interval: str = "1d") -> pd.DataFrame:
    """One direct chart-API request; raises on any failure.

    ``interval``: "1d" (default) returns a date-only Date column; other
    intervals ("1h", ...) return tz-aware (exchange-local) timestamps.
    """
    from zoneinfo import ZoneInfo

    # NOTE: the v8 chart API takes period1/period2 in SECONDS (ms values are
    # rejected with HTTP 400 — verified empirically through a working proxy).
    period1 = int(datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp())
    period2 = int(
        datetime(today.year, today.month, today.day, 23, 59, 59, tzinfo=timezone.utc).timestamp()
    )
    last_err = None
    for host in ("query1", "query2"):
        url = f"https://{host}.finance.yahoo.com/v8/finance/chart/{sym}"
        params = {"period1": period1, "period2": period2, "interval": interval}
        try:
            resp = session.get(url, params=params, timeout=30)
            if resp.status_code in (401, 403) and "crumb" not in params:
                # some sessions require a crumb: fetch it and retry once
                try:
                    r = session.get("https://query1.finance.yahoo.com/v1/test/getcrumb", timeout=15)
                    if r.status_code == 200 and r.text.strip():
                        params["crumb"] = r.text.strip()
                        resp = session.get(url, params=params, timeout=30)
                except Exception:
                    pass
            if resp.status_code == 429 or resp.status_code == 403:
                raise RuntimeError(f"Yahoo rate-limited (HTTP {resp.status_code})")
            resp.raise_for_status()
            js = resp.json()
            if "chart" not in js or not js["chart"].get("result"):
                raise RuntimeError(
                    f"chart API returned no result for {sym}: {js.get('chart', {}).get('error')}"
                )
            res = js["chart"]["result"][0]
            timestamps = res.get("timestamp") or []
            quote = res["indicators"]["quote"][0]

            tzname = res.get("meta", {}).get("exchangeTimezoneName") or "UTC"
            try:
                tz = ZoneInfo(tzname)
            except Exception:
                tz = timezone.utc

            rows = []
            for i, ts in enumerate(timestamps):
                rows.append(
                    {
                        "Date": datetime.fromtimestamp(ts, tz=tz),  # tz-aware, exchange-local
                        "Open": quote["open"][i],
                        "High": quote["high"][i],
                        "Low": quote["low"][i],
                        "Close": quote["close"][i],
                        "Volume": quote["volume"][i],
                    }
                )
            df = pd.DataFrame(rows, columns=COLUMNS)
            df["Date"] = pd.to_datetime(df["Date"])
            if interval == "1d":
                df["Date"] = df["Date"].dt.date
                df = df[(df["Date"] >= start) & (df["Date"] <= today)].reset_index(drop=True)
            else:
                start_dt = datetime.combine(start, datetime.min.time(), tzinfo=tz)
                end_dt = datetime.combine(today, datetime.max.time(), tzinfo=tz)
                df = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)].reset_index(drop=True)
            df = df.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)
            if len(df) == 0:
                raise RuntimeError(f"no usable rows after filtering for {sym}")
            return df.sort_values("Date").reset_index(drop=True)
        except Exception as exc:  # noqa: BLE001 - try next host
            last_err = exc
    raise RuntimeError(f"direct chart API failed for {sym}: {last_err}")


def download_direct(
    sym: str,
    start: date,
    attempts: int = 4,
    base_sleep: int = 30,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download one symbol via the direct chart API with retry/backoff."""
    import time

    session = _yahoo_session()
    today = datetime.now(timezone.utc).date()
    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return _fetch_chart(session, sym, start, today, interval=interval)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < attempts:
                sleep = base_sleep * attempt  # 30, 60, 120, 240 s
                print(
                    f"[download] {sym}: direct API attempt {attempt}/{attempts} failed "
                    f"({exc}); retrying in {sleep}s",
                    file=sys.stderr,
                )
                time.sleep(sleep)
    raise RuntimeError(f"direct API gave up after {attempts} attempts: {last_err}")


# --------------------------------------------------------------------------- #
# Saving / main
# --------------------------------------------------------------------------- #
def save_csv(df: pd.DataFrame, sym: str) -> Path:
    out = df.copy()
    out["Date"] = out["Date"].astype(str)  # ISO date string
    out = out[COLUMNS]
    path = RAW_DIR / f"{sym}.csv"
    out.to_csv(path, index=False)
    return path


def _download_with_retries(fn, label: str, attempts: int = 2, sleep: int = 45):
    """Run ``fn`` up to ``attempts`` times with a fixed sleep between tries."""
    import time

    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < attempts:
                print(
                    f"[download] {label} attempt {attempt}/{attempts} failed ({exc}); "
                    f"retrying in {sleep}s",
                    file=sys.stderr,
                )
                time.sleep(sleep)
    raise RuntimeError(f"{label} gave up after {attempts} attempts: {last_err}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download daily OHLCV for MES=F / MNQ=F.")
    parser.add_argument("--start", default=DEFAULT_START.isoformat(), help="start date YYYY-MM-DD")
    args = parser.parse_args()
    start = date.fromisoformat(args.start)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    proxy = apply_proxy()
    if proxy:
        print(f"[download] using proxy: {proxy}", file=sys.stderr)

    method = None
    frames: dict[str, pd.DataFrame] = {}
    try:
        if ensure_yfinance():
            frames = _download_with_retries(lambda: download_yfinance(start), "yfinance")
            method = "yfinance"
        else:
            raise RuntimeError("yfinance import failed after install attempt")
    except Exception as exc:  # noqa: BLE001 - any yfinance breakage triggers fallback
        print(f"[download] yfinance failed ({exc}); falling back to direct chart API", file=sys.stderr)
        try:
            frames = {sym: download_direct(sym, start) for sym in SYMBOLS}
            method = "direct chart API"
        except Exception as exc2:  # noqa: BLE001
            print(f"[download] FATAL: both methods failed: {exc2}", file=sys.stderr)
            return 1

    for sym in SYMBOLS:
        df = frames[sym]
        path = save_csv(df, sym)
        first = df["Date"].min()
        last = df["Date"].max()
        print(f"{sym}: {len(df)} rows, first={first}, last={last} -> {path}")

    print(f"download method: {method}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
