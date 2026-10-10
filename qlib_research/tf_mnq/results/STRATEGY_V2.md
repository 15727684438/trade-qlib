# STRATEGY V2 — EMA20/144 regime + structural trail (IS-only verdict)

- Core regime EMA20>EMA144 & close>EMA144 (fixed); structural stop n=2/buf0.25/floor2.0/act0; base gates {VR<1, ATRpct<=90, ext<=2.5}; 1 contract, $50k, 1.5t (stress 3t).

- Cost model corrected 2026-10-10: base ('1.5t') = $0.50/side comm + $0.20/side slip; stress ('3.0t') = $0.50/side + $1.50/side slip.

- Pre-registered gates (both 1.5t & 3t): MaxDD_usd@1c<=$1200; net profit>=$5000; win in [0.42,0.55]; Sharpe>=1.0; expectancy>=$4; n>=250; worst trade>=-$400; boot p<=0.10; neighborhood MaxDD<=$1500.

- **Verdict: ZERO PASSERS**

## Grid (IS)

| tag | cost | n | PF | win | exp$ | net$ | Sharpe$ | MaxDD$ | worst trade$ | halted d | foregone | g1 | g2 | g3 | g4 | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_T2 | 1.5t | 774 | 1.202 | 0.366 | $6.83 | $5283 | 0.78 | $1328 | $-230 | 0 | 0 | False | True | False | False | False |
| B0_1.5t | 1.5t | 1146 | 1.242 | 0.397 | $7.73 | $8854 | 1.13 | $1819 | $-259 | 0 | 0 | False | True | False | True | False |
| B0_3t | 3.0t | 1146 | 1.153 | 0.382 | $5.13 | $5875 | 0.75 | $2439 | $-262 | 0 | 0 | True | True | True | True | True |
| Bridge | 1.5t | 774 | 1.347 | 0.412 | $10.03 | $7764 | 1.31 | $1854 | $-259 | 0 | 0 | False | True | False | True | False |
| DD_daily100 | 1.5t | 772 | 1.183 | 0.364 | $6.20 | $4788 | 0.72 | $1942 | $-230 | 15 | 13 | False | False | False | False | False |
| DD_daily150 | 1.5t | 773 | 1.204 | 0.366 | $6.87 | $5312 | 0.79 | $1382 | $-230 | 3 | 3 | False | True | False | False | False |
| DD_halt300 | 1.5t | 38 | 0.396 | 0.211 | $-13.67 | $-519 | -0.86 | $636 | $-55 | 1350 | 1644 | True | False | False | False | False |
| DD_halt300_rth | 1.5t | 43 | 1.337 | 0.372 | $7.20 | $310 | 0.19 | $605 | $-216 | 1172 | 466 | True | False | False | False | False |
| E_confirm2 | 1.5t | 774 | 1.201 | 0.366 | $6.80 | $5260 | 0.78 | $1349 | $-230 | 0 | 0 | False | True | False | False | False |
| G_atrp80 | 1.5t | 730 | 1.177 | 0.356 | $5.84 | $4260 | 0.66 | $1170 | $-222 | 0 | 0 | True | False | False | False | False |
| G_er25 | 1.5t | 551 | 1.280 | 0.388 | $9.26 | $5101 | 0.90 | $1311 | $-350 | 0 | 0 | False | True | False | False | False |
| G_rth | 1.5t | 272 | 1.413 | 0.452 | $15.75 | $4283 | 0.96 | $1229 | $-222 | 0 | 0 | False | False | True | False | False |
| SP_act1.0 | 1.5t | 706 | 1.188 | 0.405 | $7.82 | $5520 | 0.77 | $2181 | $-280 | 0 | 0 | False | True | False | False | False |
| SP_buf0.5 | 1.5t | 724 | 1.283 | 0.370 | $10.00 | $7241 | 1.03 | $1507 | $-230 | 0 | 0 | False | True | False | True | False |
| SP_floor1.5 | 1.5t | 801 | 1.246 | 0.346 | $7.47 | $5987 | 0.92 | $1887 | $-268 | 0 | 0 | False | True | False | False | False |
| SP_pivotn3 | 1.5t | 719 | 1.184 | 0.360 | $6.95 | $4995 | 0.70 | $1698 | $-241 | 0 | 0 | False | False | False | False | False |
| T1 | 1.5t | 1566 | 1.173 | 0.324 | $5.81 | $9105 | 0.96 | $2848 | $-289 | 0 | 0 | False | True | False | False | False |
| T3 | 1.5t | 1001 | 1.222 | 0.356 | $7.38 | $7389 | 0.97 | $2407 | $-239 | 0 | 0 | False | True | False | False | False |
| W_be1.0 | 1.5t | 807 | 1.286 | 0.316 | $7.76 | $6261 | 1.00 | $1548 | $-350 | 0 | 0 | False | True | False | True | False |
| X_best_dd300 | 1.5t | 63 | 0.543 | 0.238 | $-9.36 | $-589 | -0.72 | $610 | $-90 | 1358 | 6490 | True | False | False | False | False |
| X_best_dd300_rth | 1.5t | 99 | 0.946 | 0.333 | $-1.40 | $-138 | -0.06 | $694 | $-258 | 1172 | 1497 | True | False | False | False | False |

## Diagnostics

- DSR(A_T2) = 0.0709 (p=0.9291); E[max SR] N=100 = 1.470, observed = 0.793; Sidak p = 0.9731.
- Placebo (500 random-entry, n-matched, same structural stop + breaker): real PF=1.202 at 87.8th pct (p95=1.270, max=1.429); median placebo n=768 vs real 774.
- Per-year net $ (A_T2): [(2019, -41), (2020, 1097), (2021, 2251), (2022, 508), (2023, 1468)] -> 4/5 positive.
- OOS NOT touched.

---

## OOS verdict — A_T2 (single pre-registered touch, 2024-01-01..2026-10-02)

> OOS re-evaluated 2026-10-10 under corrected cost (commission $0.50/side + slippage $0.20/side); same frozen config, no re-selection.

- IS reproduction: **OK** (n=774, PF 1.202, win 0.366, net $5282.93, MaxDD $1327.89, Sharpe 0.782) — matches frozen grid row A_T2.

| period | n | PF | win | net$ | expectancy$ | Sharpe$ | MaxDD@1c$ | MaxDD@2c$ | worst$ | CAGR$ | MAR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| IS 2019-05..2023-12 | 774 | 1.202 | 0.366 | $5283 | $6.83 | 0.78 | $1328 | $2656 | $-230 | 2.18% | 0.82 |
| OOS 2024-01..2026-10 | 514 | 1.033 | 0.356 | $946 | $1.84 | 0.17 | $2808 | $5617 | $-646 | 0.68% | 0.12 |

- 3t for the record: net $946, PF 1.033, MaxDD@1c $2808 (identical to 1.5t: `run_ema_struct` uses fixed 1/2/1-tick fills, so cost.slip_ticks does not bind here).

### Per calendar year (OOS)

| year | n | PF | net$ |
|---|---|---|---|
| 2024 | 150 | 1.194 | $1306 |
| 2025 | 219 | 0.867 | $-1594 |
| 2026 | 145 | 1.122 | $1234 |

### Pre-registered gates

| # | gate | actual | verdict |
|---|---|---|---|
| 1 | OOS MaxDD@1c <= $1,500 | $2808 | FAIL |
| 2 | OOS expectancy > 0 | $1.84 | PASS |
| 3 | OOS PF >= 1.05 | 1.033 | FAIL |
| 4 | every year PF >= 1.0 | 2024=1.194, 2025=0.867, 2026=1.122 | FAIL |
| 5 | OOS MaxDD@1c <= 1.6x IS ($2125) | $2808 | FAIL |

**OVERALL OOS: FAIL** — at least one gate failed; the V2 family is closed (no second touch, no nudging, no re-tuning).

### IS -> OOS degradation

- PF 1.202 -> 1.033; win 0.366 -> 0.356; MaxDD@1c $1328 -> $2808; expectancy $6.83 -> $1.84.
