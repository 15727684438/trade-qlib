# MNQ/MES 震荡区间 & 趋势识别器 — 设计规格

Date: 2026-08-25 · Applies to: MNQ (Micro E-mini Nasdaq-100), MES (Micro E-mini S&P 500), also NQ/ES
Deliverables: TradingView Pine v6 indicator + Freqtrade strategy + Python reference (`mnq_mes/src/mrind.py`)

> This document is the canonical contract. `mrind.py` is the reference implementation;
> `tradingview/mnq_mes_range_trend.pine` is a byte-level port of its formulas; the Freqtrade
> strategy (P2) reuses `mrind.py`. Any divergence is a bug.

## 1. Idea
Markets alternate between **range (震荡)** and **trend (趋势)** regimes. A single composite
regime state is derived per bar from three well-known building blocks:

- **Directionality**: Wilder ADX (trend strength) — high ADX ⇒ trending candidate.
- **Expansion/compression**: 20-bar Donchian span relative to ATR (span_atr = (don_hi - don_lo)/atr).
  A trend has price expanding away from noise; a range compresses it.
- **Direction**: price vs slow EMA (+ fast EMA alignment for entries).

Range mode trades **mean-reversion** at the Donchian channel edges; trend mode trades
**pullback continuation** to the fast EMA; a range→trend transition with a Donchian break
is a **breakout** entry.

## 2. Parameters (defaults)
| name | default | meaning |
|---|---|---|
| atr_len | 14 | ATR period (Wilder) |
| adx_len | 14 | ADX period (Wilder) |
| adx_threshold | 20.0 | ADX >= threshold → trending candidate |
| ema_fast_len | 9 | fast EMA |
| ema_slow_len | 21 | slow EMA (direction anchor) |
| donch_len | 20 | Donchian channel length (range bounds) |
| bb_len | 20 | Bollinger length (exit reference) |
| bb_mult | 2.0 | BB std multiplier |
| rsi_len | 14 | RSI length (Wilder) |
| chan_atr_min | 1.0 | min span_atr for a trending regime |

## 3. Computations (per bar, after warmup)
```
atr      = ATR(high, low, close, atr_len)            # Wilder
adx      = ADX(high, low, close, adx_len)            # Wilder
ema_fast = EMA(close, ema_fast_len)
ema_slow = EMA(close, ema_slow_len)
don_hi   = highest(high, donch_len)                  # rolling, current bar inclusive
don_lo   = lowest(low, donch_len)
bb_mid   = SMA(close, bb_len)
bb_std   = stdev(close, bb_len)                      # POPULATION stdev (ddof=0)
bb_hi    = bb_mid + bb_mult * bb_std
bb_lo    = bb_mid - bb_mult * bb_std
rsi      = RSI(close, rsi_len)                       # Wilder
span_atr = (don_hi - don_lo) / atr
trending = (adx >= adx_threshold) and (span_atr >= chan_atr_min)
up       = close > ema_slow
down     = close < ema_slow
state:   TREND_UP (=1)   if trending and up
         TREND_DOWN (=-1) if trending and down
         RANGE (=0)      otherwise
```
EMA smoothing = 2/(n+1). Donchian/BB/EMA are inclusive-of-current-bar rolling windows (TA-Lib/Pine convention).

## 4. Signals
Long signals (dir=+1): 
- **RANGE_LONG**:   state==RANGE and low <= don_lo and rsi < 45
- **TREND_LONG**:   state==TREND_UP and low <= ema_fast and close > ema_slow
- **BRKOUT_LONG**:  state==TREND_UP and crossover(close, don_hi_prev)   (don_hi_prev = prior bar don_hi)

Short signals (dir=-1):
- **RANGE_SHORT**:  state==RANGE and high >= don_hi and rsi > 55
- **TREND_SHORT**:  state==TREND_DOWN and high >= ema_fast and close < ema_slow
- **BRKOUT_SHORT**: state==TREND_DOWN and crossunder(close, don_lo_prev)

Exit guidance (long; symmetric short) — per-bar, NO position tracking. mrind.py is canonical;
Pine visual and P2 strategy must match this exactly:
- in RANGE:      exit when close >= bb_mid
- in TREND_UP:   exit when close < ema_slow
- in TREND_DOWN: always exit (no long should be held)

ATR stop (both): entry-dependent, therefore NOT in the per-bar reference. Delegated to the P2
Freqtrade strategy state machine: stop when close <= max_close_since_entry - 2.0 * atr
(atr re-evaluated each bar; max_close_since_entry updates on each bar close). Freqtrade mapping:
built-in trailing_stop is HIGH-based (not close-based) — use custom_stoploss for close-based
fidelity, else document the approximation. Fills are assumed at signal-bar CLOSE everywhere
(touch prices are not executable).

Tag precedence if multiple conditions fire on one bar: RANGE_LONG < TREND_LONG < BRKOUT_LONG
(breakout most meaningful); same for shorts.

Empirical notes (accepted, P1 gate review):
- RANGE_LONG is structurally rare on MNQ/MES daily (~25 in 7.5y per symbol): a fresh 20-day low
  almost always co-occurs with ADX>=20, classifying TREND_DOWN rather than RANGE (positive drift).
  Accepted by design; do NOT relax the RSI<45 filter to compensate. With ~26 samples, RANGE_LONG
  is not statistically validatable in P2.
- chan_atr_min is a cross-market safety net, not an MNQ/MES discriminator: on these symbols every
  ADX>=20 bar already has span_atr>=1.0 (min ~2.2, median ~4.5).
- bb_hi/bb_lo are display-only outputs; signals/exits use only bb_mid (== SMA(close,20)).

## 5. Display (Pine)
- overlay: don_hi/don_lo channel (fill), ema_fast/ema_slow, optional BB
- bgcolor: TREND_UP green, TREND_DOWN red, RANGE gray-blue
- markers: ▲/▼ at signals with tag labels
- sub-pane: ADX + threshold line, RSI
- table: state, ADX, span_atr, RSI
- alertcondition(): regime change, RANGE_LONG/SHORT, TREND_LONG/SHORT, BRKOUT_LONG/SHORT

## 6. Parity contract
- TA-Lib ATR/ADX/RSI (Wilder) == Pine ta.atr/adx/rsi (Wilder); both use population stdev for BB;
  EMA smoothing 2/(n+1); rolling windows inclusive of current bar.
- mrind.py API:
  - `RegimeParams` dataclass (fields/names = section 2 table, defaults as listed)
  - `compute_regime(df, p=RegimeParams()) -> df` — input needs columns open,high,low,close;
    adds columns: atr, adx, ema_fast, ema_slow, don_hi, don_lo, bb_mid, bb_hi, bb_lo, rsi,
    span_atr, trending, up, down, state (1/-1/0), state_label.
  - `signal_events(df, p=RegimeParams()) -> df` — adds: sig (tag or ""), dir (1/-1/0),
    exit_long, exit_short (bool), plus don_hi_prev/don_lo_prev used for breakout.
  - Warmup rows (first max(atr_len, adx_len, donch_len, bb_len, rsi_len, ema lengths) bars)
    get state=0 and no signals.
- Validation script must compare regime stats and be cross-checkable by hand.

## 7. Data (P1 validation / P2 backtest)
- Primary: Yahoo `MES=F`, `MNQ=F` daily 2019-05-06 → present (yfinance or direct chart API).
- Caveats: front-month splice with unadjusted roll gaps; acceptable for regime-stat validation
  and demo backtest; document Databento (`GLBX.MDP3`, `stype_in="continuous"`) as upgrade path.
