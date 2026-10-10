# STRATEGY V2 — EMA20/144 regime + structural trail (IS-only verdict)

- Core regime EMA20>EMA144 & close>EMA144 (fixed); structural stop n=2/buf0.25/floor2.0/act0; base gates {VR<1, ATRpct<=90, ext<=2.5}; 1 contract, $50k, 1.5t (stress 3t).

- Pre-registered gates (both 1.5t & 3t): MaxDD_usd@1c<=$1200; net profit>=$5000; win in [0.42,0.55]; Sharpe>=1.0; expectancy>=$4; n>=250; worst trade>=-$400; boot p<=0.10; neighborhood MaxDD<=$1500.

- **Verdict: ZERO PASSERS**

## Grid (IS)

| tag | cost | n | PF | win | exp$ | net$ | Sharpe$ | MaxDD$ | worst trade$ | halted d | foregone | g1 | g2 | g3 | g4 | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_T2 | 1.5t | 774 | 1.185 | 0.362 | $6.33 | $4896 | 0.72 | $1362 | $-230 | 0 | 0 | False | False | False | False | False |
| B0_1.5t | 1.5t | 1146 | 1.186 | 0.387 | $6.13 | $7021 | 0.90 | $2158 | $-261 | 0 | 0 | False | True | False | False | False |
| B0_3t | 3.0t | 1146 | 1.137 | 0.377 | $4.63 | $5302 | 0.68 | $2586 | $-262 | 0 | 0 | True | True | True | True | True |
| Bridge | 1.5t | 774 | 1.282 | 0.404 | $8.43 | $6526 | 1.10 | $1938 | $-261 | 0 | 0 | False | True | False | True | False |
| DD_daily100 | 1.5t | 772 | 1.166 | 0.360 | $5.70 | $4402 | 0.66 | $1976 | $-230 | 15 | 13 | False | False | False | False | False |
| DD_daily150 | 1.5t | 773 | 1.187 | 0.362 | $6.37 | $4926 | 0.73 | $1416 | $-230 | 3 | 3 | False | False | False | False | False |
| DD_halt300 | 1.5t | 38 | 0.384 | 0.211 | $-14.17 | $-538 | -0.89 | $647 | $-55 | 1350 | 1644 | True | False | False | False | False |
| DD_halt300_rth | 1.5t | 43 | 1.309 | 0.372 | $6.70 | $288 | 0.18 | $606 | $-217 | 1172 | 466 | True | False | False | False | False |
| E_confirm2 | 1.5t | 774 | 1.184 | 0.362 | $6.30 | $4873 | 0.72 | $1383 | $-230 | 0 | 0 | False | False | False | False | False |
| G_atrp80 | 1.5t | 730 | 1.160 | 0.352 | $5.34 | $3895 | 0.61 | $1231 | $-222 | 0 | 0 | False | False | False | False | False |
| G_er25 | 1.5t | 551 | 1.263 | 0.381 | $8.76 | $4826 | 0.85 | $1325 | $-351 | 0 | 0 | False | False | False | False | False |
| G_rth | 1.5t | 272 | 1.397 | 0.452 | $15.25 | $4147 | 0.93 | $1247 | $-222 | 0 | 0 | False | False | True | False | False |
| SP_act1.0 | 1.5t | 706 | 1.175 | 0.404 | $7.32 | $5167 | 0.72 | $2213 | $-280 | 0 | 0 | False | True | False | False | False |
| SP_buf0.5 | 1.5t | 724 | 1.266 | 0.370 | $9.50 | $6879 | 0.98 | $1523 | $-230 | 0 | 0 | False | True | False | False | False |
| SP_floor1.5 | 1.5t | 801 | 1.227 | 0.341 | $6.97 | $5586 | 0.86 | $1901 | $-268 | 0 | 0 | False | True | False | False | False |
| SP_pivotn3 | 1.5t | 719 | 1.169 | 0.359 | $6.45 | $4635 | 0.65 | $1730 | $-242 | 0 | 0 | False | False | False | False | False |
| T1 | 1.5t | 1566 | 1.157 | 0.321 | $5.31 | $8322 | 0.88 | $2917 | $-290 | 0 | 0 | False | True | False | False | False |
| T3 | 1.5t | 1001 | 1.205 | 0.354 | $6.88 | $6889 | 0.90 | $2456 | $-240 | 0 | 0 | False | True | False | False | False |
| W_be1.0 | 1.5t | 811 | 1.258 | 0.308 | $7.06 | $5730 | 0.92 | $1559 | $-351 | 0 | 0 | False | True | False | False | False |
| X_best_dd300 | 1.5t | 62 | 0.543 | 0.242 | $-9.43 | $-584 | -0.72 | $605 | $-91 | 1358 | 6495 | True | False | False | False | False |
| X_best_dd300_rth | 1.5t | 99 | 0.927 | 0.333 | $-1.90 | $-188 | -0.08 | $697 | $-259 | 1172 | 1497 | True | False | False | False | False |

## Diagnostics

- DSR(A_T2) = 0.0776 (p=0.9224); E[max SR] N=100 = 1.390, observed = 0.735; Sidak p = 0.9919.
- Placebo (500 random-entry, n-matched, same structural stop + breaker): real PF=1.185 at 88.0th pct (p95=1.250, max=1.408); median placebo n=768 vs real 774.
- Per-year net $ (A_T2): [(2019, -94), (2020, 989), (2021, 2161), (2022, 443), (2023, 1397)] -> 4/5 positive.
- OOS NOT touched.

---

## OOS verdict — A_T2 (single pre-registered touch, 2024-01-01..2026-10-02)

- IS reproduction: **OK** (n=774, PF 1.185, win 0.362, net $4895.93, MaxDD $1361.89, Sharpe 0.724) — matches frozen grid row A_T2.

| period | n | PF | win | net$ | expectancy$ | Sharpe$ | MaxDD@1c$ | MaxDD@2c$ | worst$ | CAGR$ | MAR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| IS 2019-05..2023-12 | 774 | 1.185 | 0.362 | $4896 | $6.33 | 0.72 | $1362 | $2724 | $-230 | 2.03% | 0.75 |
| OOS 2024-01..2026-10 | 514 | 1.024 | 0.352 | $689 | $1.34 | 0.13 | $2872 | $5745 | $-647 | 0.50% | 0.09 |

- 3t for the record: net $689, PF 1.024, MaxDD@1c $2872 (identical to 1.5t: `run_ema_struct` uses fixed 1/2/1-tick fills, so cost.slip_ticks does not bind here).

### Per calendar year (OOS)

| year | n | PF | net$ |
|---|---|---|---|
| 2024 | 150 | 1.182 | $1231 |
| 2025 | 219 | 0.858 | $-1704 |
| 2026 | 145 | 1.115 | $1162 |

### Pre-registered gates

| # | gate | actual | verdict |
|---|---|---|---|
| 1 | OOS MaxDD@1c <= $1,500 | $2872 | FAIL |
| 2 | OOS expectancy > 0 | $1.34 | PASS |
| 3 | OOS PF >= 1.05 | 1.024 | FAIL |
| 4 | every year PF >= 1.0 | 2024=1.182, 2025=0.858, 2026=1.115 | FAIL |
| 5 | OOS MaxDD@1c <= 1.6x IS ($2179) | $2872 | FAIL |

**OVERALL OOS: FAIL** — at least one gate failed; the V2 family is closed (no second touch, no nudging, no re-tuning).

### IS -> OOS degradation

- PF 1.185 -> 1.024; win 0.362 -> 0.352; MaxDD@1c $1362 -> $2872; expectancy $6.33 -> $1.34.
