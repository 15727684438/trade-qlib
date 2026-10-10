Cost model corrected 2026-10-10: base ('1.5t' label retained) = commission $0.50/side + slippage $0.20/side ($1.40 round trip); stress ('3.0t') = $0.50/side + $1.50/side slippage.

# RISK-FIRST REDESIGN (15min) — IS-only verdict

- Base = Donchian-20, trail 2.5xATR, vetoes {VR>=1, ATRpct<=90}, market, 1.5t, 1 contract, $50k.

- FREQ=15min; IS bars=111302; ppy=23941.7; ATR ratio 15min/15min = 1.0000.
- Pre-registered rescale: fixed-pt C1/C2/C3 thresholds x1.0000 (C1 [-100.0, -150.0], C2 [300.0, 500.0], C3 -250.0 pt); all k*ATR multipliers unchanged; USD DD gates unscaled ($1200 / $1500).

- Pre-registered gate: MaxDD_usd@1c <= $1200 (hard), PF>=1.10, boot p<=0.10, expectancy>=$2, n>=300, worst trade <= $500, neighborhood MaxDD<=$1500.

- **Verdict: ZERO PASSERS** (IS gates at BOTH 1.5t and 3t).

## Why nothing passed

The pre-registered gate required MaxDD_usd@1c <= $1200 at BOTH 1.5t and 3t while keeping PF>=1.10 (boot p<=0.10), expectancy>=$2, n>=300 and worst trade>=-$500. Two disjoint failure modes emerged:

1. **Good-PF variants blow the DD gate.** Best non-breaker improvements (Donchian-32, ratchet-only, time-stop-16) reach PF 1.24-1.29 but only cut MaxDD to ~$1,730-$2,100, still far above the $1,200 gate (and $1,500 hard cap).
2. **The DD-halt breaker meets the DD gate but destroys the sample/edge.** C2 with T=300/500 pt halts new entries for ~1,185-1,190 days, cuts n from 1,302 to ~220, and drops PF to 0.97-1.05 with expectancy ~$0-$1 — failing gate 2 (PF/bootstrap) and gate 3 (n>=300, expectancy>=$2). The breaker does **not** manufacture PF; it lowers it.

### Key competitors (IS, 1.5t)

| config | n | PF | win | exp$ | CAGR$ | Sharpe$ | MaxDD$ | halted d | foregone | g1 (DD<=1200) | g2 (PF/boot) | g3 (exp/n) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0_base_1.5t | 1302 | 1.290 | 0.396 | $9.56 | 4.90% | 1.46 | $1772 | 0 | 0 | False | True | True |
| F1_donch32 | 1081 | 1.348 | 0.418 | $11.26 | 4.80% | 1.57 | $1532 | 0 | 0 | False | True | True |
| E3_ratchet | 1319 | 1.300 | 0.393 | $9.63 | 4.99% | 1.50 | $1843 | 0 | 0 | False | True | True |
| E5_time16 | 1325 | 1.298 | 0.380 | $9.49 | 4.94% | 1.48 | $1638 | 0 | 0 | False | True | True |
| C2_dd300 | 218 | 1.150 | 0.362 | $2.49 | 0.23% | 0.32 | $637 | 1190 | 3283 | True | False | False |
| C2_dd500 | 226 | 1.067 | 0.363 | $1.24 | 0.12% | 0.14 | $1015 | 1185 | 3256 | True | False | False |
| X_Ebest_C2best | 220 | 1.183 | 0.373 | $2.90 | 0.27% | 0.38 | $611 | 1190 | 3283 | True | False | False |
| X_Fbest_C2best | 177 | 1.120 | 0.390 | $2.15 | 0.16% | 0.25 | $645 | 1190 | 2734 | True | False | False |

**Honest statement: ZERO variants pass all IS gates.** The $1,500 risk budget (let alone the $1,200 buffer) is not attainable at 1 contract on a $50k account with this breakout+trail engine: the per-contract risk (base MaxDD $2,304 = 1,152 pt) is ~2x the budget, and the only mechanism that fits the budget (equity-DD halting) removes so much participation that the statistical edge collapses.

## Sample table (all variants, 1.5t)

| tag | n | PF | win | exp$ | CAGR$ | Sharpe$ | MaxDD$ | worst trade$ | worst day$ | halted d | foregone | MAR | g1 | g2 | g3 | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0_base_1.5t | 1302 | 1.290 | 0.396 | $9.56 | 4.90% | 1.46 | $1772 | $-287 | $-439 | 0 | 0 | 1.382 | False | True | True | False |
| E1_trail1.0 | 2677 | 0.514 | 0.168 | $-7.15 | -9.86% | -4.65 | $19336 | $-163 | $-406 | 0 | 0 | -0.255 | False | False | False | False |
| E1_trail1.5 | 1963 | 0.858 | 0.288 | $-3.02 | -2.68% | -1.00 | $6298 | $-258 | $-756 | 0 | 0 | -0.212 | False | False | False | False |
| E1_trail2.0 | 1562 | 1.126 | 0.361 | $3.43 | 2.21% | 0.74 | $2005 | $-342 | $-731 | 0 | 0 | 0.552 | False | True | True | False |
| E2_init1.0 | 1521 | 1.206 | 0.300 | $5.41 | 3.33% | 1.10 | $2487 | $-183 | $-704 | 0 | 0 | 0.670 | False | True | True | False |
| E2_init1.5 | 1358 | 1.274 | 0.370 | $8.45 | 4.55% | 1.42 | $2957 | $-273 | $-1053 | 0 | 0 | 0.769 | False | True | True | False |
| E2_init2.0 | 1316 | 1.274 | 0.390 | $8.89 | 4.62% | 1.39 | $2712 | $-258 | $-399 | 0 | 0 | 0.853 | False | True | True | False |
| E3_ratchet | 1319 | 1.300 | 0.393 | $9.63 | 4.99% | 1.50 | $1843 | $-283 | $-433 | 0 | 0 | 1.353 | False | True | True | False |
| E4_be1.0 | 1658 | 0.978 | 0.472 | $-0.56 | -0.41% | -0.11 | $6430 | $-436 | $-443 | 0 | 0 | -0.031 | False | False | False | False |
| E4_be1.5 | 1398 | 1.196 | 0.496 | $5.84 | 3.31% | 1.02 | $3801 | $-287 | $-439 | 0 | 0 | 0.435 | False | True | True | False |
| E5_time16 | 1325 | 1.298 | 0.380 | $9.49 | 4.94% | 1.48 | $1638 | $-287 | $-439 | 0 | 0 | 1.509 | False | True | True | False |
| E5_time48 | 1302 | 1.290 | 0.396 | $9.56 | 4.90% | 1.46 | $1772 | $-287 | $-439 | 0 | 0 | 1.382 | False | True | True | False |
| F1_donch32 | 1081 | 1.348 | 0.418 | $11.26 | 4.80% | 1.57 | $1532 | $-287 | $-378 | 0 | 0 | 1.566 | False | True | True | False |
| F1_donch48 | 928 | 1.290 | 0.390 | $9.84 | 3.67% | 1.27 | $2701 | $-300 | $-378 | 0 | 0 | 0.680 | False | True | True | False |
| F2_cap75 | 1138 | 1.180 | 0.388 | $5.51 | 2.57% | 0.88 | $2498 | $-436 | $-650 | 0 | 0 | 0.515 | False | True | True | False |
| F2_cap85 | 1258 | 1.269 | 0.392 | $8.65 | 4.33% | 1.34 | $2419 | $-259 | $-414 | 0 | 0 | 0.894 | False | True | True | False |
| C1_daily100 | 1290 | 1.279 | 0.396 | $9.13 | 4.65% | 1.42 | $2789 | $-287 | $-439 | 40 | 38 | 0.835 | False | True | True | False |
| C1_daily150 | 1301 | 1.291 | 0.397 | $9.58 | 4.91% | 1.47 | $1826 | $-287 | $-439 | 10 | 3 | 1.343 | False | True | True | False |
| C2_dd300 | 218 | 1.150 | 0.362 | $2.49 | 0.23% | 0.32 | $637 | $-150 | $-180 | 1190 | 3283 | 0.182 | True | False | False | False |
| C2_dd500 | 226 | 1.067 | 0.363 | $1.24 | 0.12% | 0.14 | $1015 | $-258 | $-248 | 1185 | 3256 | 0.059 | True | False | False | False |
| C3_weekly250 | 1292 | 1.265 | 0.396 | $8.73 | 4.47% | 1.34 | $2938 | $-287 | $-439 | 15 | 26 | 0.761 | False | True | True | False |
| X_Ebest_C2best | 220 | 1.183 | 0.373 | $2.90 | 0.27% | 0.38 | $611 | $-150 | $-174 | 1190 | 3283 | 0.223 | True | False | False | False |
| X_Fbest_C2best | 177 | 1.120 | 0.390 | $2.15 | 0.16% | 0.25 | $645 | $-239 | $-355 | 1190 | 2734 | 0.126 | True | False | False | False |

## Multiple-testing / placebo

- DSR(base) = 0.0084 (p=0.9916); E[max SR] under N=100 = 2.579, observed base SR = 1.475.
- Random-entry placebo WITH chosen breaker (C2_dd500): real PF=1.067 at percentile 75.8% (p95=1.333, max=1.721); median placebo n=92 vs real 226.
- 2020 contribution — base: n=300, PF=1.367, PnL=$3756; ref: n=51, PF=0.833, PnL=$-281.
- OOS NOT touched this round.

## 15min vs 5min comparison (IS, 1.5t)

| config | 15min n | 15min PF | 15min MaxDD$ | 5min n | 5min PF | 5min MaxDD$ |
|---|---|---|---|---|---|---|
| R0_base_1.5t | 1302 | 1.290 | $1772 | 3855 | 0.919 | $7480 |
| F1_donch32 | 1081 | 1.348 | $1532 | 3145 | 0.908 | $6866 |
| E3_ratchet | 1319 | 1.300 | $1843 | 3895 | 0.919 | $7220 |
| E5_time16 | 1325 | 1.298 | $1638 | 3928 | 0.912 | $7722 |
| C2_dd300 | 218 | 1.150 | $637 | 22 | 0.270 | $339 |
| C2_dd500 | 226 | 1.067 | $1015 | 82 | 0.642 | $557 |
