"""Regime range/trend strategy for MNQ/MES daily bars (auto-trading port).

Canonical contract
------------------
* Spec: ``docs/regime-indicator-spec.md`` (params §2, computations §3, signals
  §4, API/parity §6).
* Reference implementation: ``mnq_mes/src/mrind.py`` — this strategy REUSES it
  verbatim (``compute_regime`` / ``signal_events``); it does not re-implement
  any formula.

Port notes (Oracle-approved P2)
------------------------------
* Warmup rows are dropped by freqtrade via ``startup_candle_count = 30``
  (mrind's own warmup is 21 bars for the default params; 30 covers it).
* The ``sig`` column is kept on the dataframe so ``enter_tag`` preserves the
  exact signal tag (RANGE_LONG/TREND_LONG/BRKOUT_LONG and the short mirrors).
* Same-bar entry+exit ordering: an entry is only allowed on a bar where there
  is no same-bar exit (``dir == 1 and not exit_long`` etc.), matching mrind's
  per-bar semantics.
* Fills: freqtrade's backtest SHIFTS the entry/exit signal columns by one bar
  (backtesting.py ``_get_ohlcv_as_lists``) to avoid lookahead, so a signal on
  bar *t* fills at bar *t+1* OPEN (long/short) and exits likewise. This is a
  documented, conservative deviation from spec §4's "signal-bar close".
* ``custom_stoploss`` implements the spec §4 ATR stop ("stop when close <=
  max_close_since_entry - 2.0*atr"). freqtrade's custom_stoploss cannot be
  purely close-based for fills — it triggers on intrabar high/low. We
  approximate ``max_close_since_entry`` with ``trade.max_rate`` (long) /
  ``trade.min_rate`` (short), i.e. a HIGH/LOW-extreme-based trailing. ATR is
  read from the analyzed dataframe at the last COMPLETED bar. This is the
  documented freqtrade approximation from the spec's "Freqtrade mapping" note.
* ``bot_start`` injects synthetic market entries for whitelisted pairs that do
  not exist on the configured exchange. BACKTEST-ONLY: live/dry-run on real
  MNQ/MES requires a real broker connector (CME micro futures are not listed
  on crypto exchanges like gate/okx). The injection no-ops for pairs that ARE
  real markets (verified on okx BTC/USDT, ETH/USDT).
"""

import math
import sys
from pathlib import Path

# Make the canonical reference importable when freqtrade loads this file.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pandas import DataFrame  # noqa: E402

from freqtrade.strategy import IStrategy, Trade, stoploss_from_open  # noqa: E402
from mrind import RegimeParams, compute_regime, signal_events  # noqa: E402


class RegimeRangeTrend(IStrategy):
    """Trade the market-regime state machine from the regime spec.

    Range regime: mean-reversion at the Donchian edges (RANGE_LONG/SHORT).
    Trend regime: pullback continuation to the fast EMA (TREND_LONG/SHORT)
    or Donchian-break breakout (BRKOUT_LONG/SHORT).
    Exits follow the per-bar guidance (range: back to BB mid; trend: anchor
    EMA broken) and a close-based ATR trailing (approximated, see class
    docstring) via custom_stoploss.
    """

    INTERFACE_VERSION = 3

    can_short: bool = True
    timeframe = "1d"

    # Number of candles needed before the first valid signal (mrind warmup is
    # 21 bars with the default params; 30 is the safe cover per P2 notes).
    startup_candle_count: int = 30

    # ROI exits disabled (the strategy exits via exit_signal + ATR stop only).
    minimal_roi = {"0": 100}

    # Backstop only: the real exit management is the exit signal and the
    # custom ATR trailing stop.
    stoploss = -0.05
    trailing_stop = False
    use_exit_signal = True
    use_custom_stoploss = True

    # ---- RegimeParams (spec §2 defaults) + ATR stop multiplier (spec §4) ----
    atr_len: int = 14
    adx_len: int = 14
    adx_threshold: float = 20.0
    ema_fast_len: int = 9
    ema_slow_len: int = 21
    donch_len: int = 20
    bb_len: int = 20
    bb_mult: float = 2.0
    rsi_len: int = 14
    chan_atr_min: float = 1.0
    atr_stop_mult: float = 2.0

    def bot_start(self) -> None:
        """Inject synthetic market entries for the configured pairs.

        MNQ/MES are CME instruments, not real markets on the configured ccxt
        exchange (gate is used only as an offline ccxt stub for the backtest).
        freqtrade needs the pair in ``exchange.markets`` to compute stake
        limits, so we register a minimal swap market per whitelisted pair here
        (called via ``ft_bot_start`` before the backtest loop starts).
        """
        ex = self.dp._exchange if self.dp is not None else None
        if ex is None:
            return
        for pair in self.config.get("exchange", {}).get("pair_whitelist", []):
            if pair in ex.markets:
                continue
            base = pair.split("/")[0]
            ex.markets[pair] = {
                "id": pair,
                "symbol": pair,
                "base": base,
                "quote": "USD",
                "settle": "USD",
                "type": "swap",
                "swap": True,
                "future": False,
                "linear": True,
                "active": True,
                "precision": {"amount": 0.00000001, "price": 0.01},
                "limits": {
                    "amount": {"min": None, "max": None},
                    "cost": {"min": None, "max": None},
                    "price": {"min": None, "max": None},
                },
                "taker": 0.0001,
                "maker": 0.0001,
            }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Add regime/signal columns via the canonical mrind reference."""
        p = RegimeParams(
            atr_len=self.atr_len,
            adx_len=self.adx_len,
            adx_threshold=self.adx_threshold,
            ema_fast_len=self.ema_fast_len,
            ema_slow_len=self.ema_slow_len,
            donch_len=self.donch_len,
            bb_len=self.bb_len,
            bb_mult=self.bb_mult,
            rsi_len=self.rsi_len,
            chan_atr_min=self.chan_atr_min,
        )
        dataframe = compute_regime(dataframe, p)
        dataframe = signal_events(dataframe, p)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Enter on mrind signal bars, only when there is no same-bar exit."""
        long_cond = (dataframe["dir"] == 1) & (~dataframe["exit_long"])
        short_cond = (dataframe["dir"] == -1) & (~dataframe["exit_short"])

        dataframe.loc[long_cond, "enter_long"] = 1
        dataframe.loc[short_cond, "enter_short"] = 1

        # Keep the exact signal tag (RANGE/TREND/BRKOUT) for analysis/reporting.
        dataframe.loc[long_cond, "enter_tag"] = dataframe.loc[long_cond, "sig"]
        dataframe.loc[short_cond, "enter_tag"] = dataframe.loc[short_cond, "sig"]

        # NOTE: sig/dir/exit_* columns are NOT mutated after this point.
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Exit on mrind per-bar exit guidance."""
        # exit_long/exit_short arrive as bool from signal_events; normalize to
        # integer signals (True -> 1, False -> 0) for freqtrade's exit-trend
        # contract (assigning int 1 into a bool column would raise
        # LossySetitemError on modern pandas).
        dataframe["exit_long"] = dataframe["exit_long"].astype(int)
        dataframe["exit_short"] = dataframe["exit_short"].astype(int)
        return dataframe

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time,
        current_rate: float,
        current_profit: float,
        after_fill: bool = False,
        **kwargs,
    ) -> float:
        """Close-based ATR trailing (spec §4), approximated via trade max/min rate.

        Spec §4: stop when close <= max_close_since_entry - 2.0*atr (atr
        re-evaluated each bar). freqtrade triggers stops intrabar (high/low), so
        this is the documented approximation: ``trade.max_rate`` /
        ``trade.min_rate`` stand in for the close-based extremes, matching the
        spec's "Freqtrade mapping" note.

        ATR is read from the ANALYZED dataframe (``get_analyzed_dataframe``,
        which is sliced to the current candle in backtest) at ``iloc[-2]`` — the
        last *completed* bar — so the stop never uses the current (possibly
        still forming) bar's ATR.

        Returns a stoploss ratio relative to the current rate (the value
        freqtrade's custom_stoploss expects); the open-relative stop is
        computed from the spec formula and converted with ``stoploss_from_open``.
        The ATR trailing only ever tightens: the open-relative stop is clamped
        so it never loosens beyond the initial ``self.stoploss`` (-0.05).
        """
        try:
            analyzed, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            atr = float(analyzed["atr"].iloc[-2])
        except Exception:
            # Analyzed dataframe unavailable (e.g. raw OHLCV source without
            # indicators) — fall back to the plain stoploss.
            return self.stoploss

        if not math.isfinite(atr) or atr <= 0:
            return self.stoploss

        # ATR trailing anchor: max close since entry (long) / min close since
        # entry (short), approximated by the trade's rate extremes. On the first
        # custom_stoploss call after entry, trade.max_rate/min_rate may still be
        # None (freqtrade initializes them lazily) — fall back to the open rate.
        if trade.is_short:
            extreme = trade.min_rate if trade.min_rate is not None else trade.open_rate
            stop_price = extreme + self.atr_stop_mult * atr
            # Mirror of the long case: relative stop from open_rate.
            open_relative_stop = (stop_price - trade.open_rate) / trade.open_rate
            # Only tighten: never loosen beyond the initial -0.05 (i.e. the
            # stop may move closer to the open, but not further above it).
            open_relative_stop = min(open_relative_stop, -self.stoploss)
        else:
            extreme = trade.max_rate if trade.max_rate is not None else trade.open_rate
            stop_price = extreme - self.atr_stop_mult * atr
            open_relative_stop = (stop_price - trade.open_rate) / trade.open_rate
            # Only tighten: never loosen beyond the initial -0.05 (i.e. the
            # stop may move closer to the open, but not further below it).
            open_relative_stop = max(open_relative_stop, self.stoploss)

        # Convert the open-relative stop to the current-rate-relative value
        # freqtrade's custom_stoploss must return.
        return stoploss_from_open(open_relative_stop, current_profit, trade.is_short)


class RegimeRangeTrendLong(RegimeRangeTrend):
    """Long-only variant of :class:`RegimeRangeTrend`.

    Same regime logic and signals; short entries are disabled via
    ``can_short = False`` (freqtrade ignores ``enter_short`` in that case).
    Rationale (P2 backtest, fixed-stop v2): on MNQ/MES daily 2020-2026 the
    symmetric strategy loses (MNQ -86.2%, MES -73.5%) and this long-only
    variant also loses but less (MNQ -43.1%, MES -25.4%) — the default regime
    signals lack positive expectation on daily bars. This variant exposes the
    long-only behaviour for regimes/markets where shorting is unwanted.
    (The earlier "+9% longs" figure was an artifact of the pre-fix stop; see
    docs/regime-quant-guide.md for the honest v2 results.)
    """

    can_short: bool = False
