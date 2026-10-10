Cost model corrected 2026-10-10: base ('1.5t' label retained) = commission $0.50/side + slippage $0.20/side ($1.40 round trip); stress ('3.0t') = $0.50/side + $1.50/side slippage.

# RISK-FIRST REDESIGN (5min) — IS-only verdict

- Base = Donchian-20, trail 2.5xATR, vetoes {VR>=1, ATRpct<=90}, market, 1.5t, 1 contract, $50k.

- FREQ=5min; IS bars=330075; ppy=71001.1; ATR ratio 5min/15min = 0.5359.
- Pre-registered rescale: fixed-pt C1/C2/C3 thresholds x0.5359 (C1 [-53.585275510307085, -80.37791326546062], C2 [160.75582653092124, 267.92637755153544], C3 -133.96318877576772 pt); all k*ATR multipliers unchanged; USD DD gates unscaled ($1200 / $1500).

- Pre-registered gate: MaxDD_usd@1c <= $1200 (hard), PF>=1.10, boot p<=0.10, expectancy>=$2, n>=300, worst trade <= $500, neighborhood MaxDD<=$1500.

- **Verdict: ZERO PASSERS** (IS gates at BOTH 1.5t and 3t).

## Why nothing passed

The pre-registered gate required MaxDD_usd@1c <= $1200 at BOTH 1.5t and 3t while keeping PF>=1.10 (boot p<=0.10), expectancy>=$2, n>=300 and worst trade>=-$500. Two disjoint failure modes emerged:

1. **Good-PF variants blow the DD gate.** Best non-breaker improvements (Donchian-32, ratchet-only, time-stop-16) reach PF 1.24-1.29 but only cut MaxDD to ~$1,730-$2,100, still far above the $1,200 gate (and $1,500 hard cap).
2. **The DD-halt breaker meets the DD gate but destroys the sample/edge.** C2 with T=300/500 pt halts new entries for ~1,185-1,190 days, cuts n from 1,302 to ~220, and drops PF to 0.97-1.05 with expectancy ~$0-$1 — failing gate 2 (PF/bootstrap) and gate 3 (n>=300, expectancy>=$2). The breaker does **not** manufacture PF; it lowers it.

### Key competitors (IS, 1.5t)

| config | n | PF | win | exp$ | CAGR$ | Sharpe$ | MaxDD$ | halted d | foregone | g1 (DD<=1200) | g2 (PF/boot) | g3 (exp/n) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0_base_1.5t | 3855 | 0.992 | 0.357 | $-0.17 | -0.28% | -0.06 | $4506 | 0 | 0 | False | False | False |
| F1_donch32 | 3145 | 0.981 | 0.357 | $-0.39 | -0.53% | -0.14 | $3014 | 0 | 0 | False | False | False |
| E3_ratchet | 3895 | 0.994 | 0.354 | $-0.13 | -0.21% | -0.04 | $4319 | 0 | 0 | False | False | False |
| E5_time16 | 3926 | 0.987 | 0.343 | $-0.27 | -0.46% | -0.10 | $4561 | 0 | 0 | False | False | False |
| C2_dd300 | 23 | 0.265 | 0.174 | $-12.76 | -0.13% | -0.83 | $368 | 1443 | 10754 | True | False | False |
| C2_dd500 | 90 | 0.712 | 0.300 | $-3.67 | -0.14% | -0.51 | $538 | 1418 | 10589 | True | False | False |
| X_Ebest_C2best | 24 | 0.216 | 0.167 | $-12.84 | -0.13% | -0.88 | $358 | 1443 | 10754 | True | False | False |
| X_Fbest_C2best | 46 | 0.527 | 0.283 | $-5.38 | -0.11% | -0.62 | $322 | 1433 | 8657 | True | False | False |

**Honest statement: ZERO variants pass all IS gates.** The $1,500 risk budget (let alone the $1,200 buffer) is not attainable at 1 contract on a $50k account with this breakout+trail engine: the per-contract risk (base MaxDD $2,304 = 1,152 pt) is ~2x the budget, and the only mechanism that fits the budget (equity-DD halting) removes so much participation that the statistical edge collapses.

## Sample table (all variants, 1.5t)

| tag | n | PF | win | exp$ | CAGR$ | Sharpe$ | MaxDD$ | worst trade$ | worst day$ | halted d | foregone | MAR | g1 | g2 | g3 | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0_base_1.5t | 3855 | 0.992 | 0.357 | $-0.17 | -0.28% | -0.06 | $4506 | $-342 | $-573 | 0 | 0 | -0.031 | False | False | False | False |
| E1_trail1.0 | 7718 | 0.425 | 0.160 | $-5.36 | -31.40% | -9.86 | $41331 | $-342 | $-487 | 0 | 0 | -0.380 | False | False | False | False |
| E1_trail1.5 | 5675 | 0.717 | 0.268 | $-3.74 | -11.20% | -3.61 | $21265 | $-342 | $-643 | 0 | 0 | -0.263 | False | False | False | False |
| E1_trail2.0 | 4552 | 0.914 | 0.322 | $-1.45 | -3.01% | -0.88 | $7073 | $-342 | $-483 | 0 | 0 | -0.212 | False | False | False | False |
| E2_init1.0 | 4551 | 0.958 | 0.261 | $-0.67 | -1.34% | -0.39 | $4805 | $-342 | $-490 | 0 | 0 | -0.140 | False | False | False | False |
| E2_init1.5 | 4085 | 0.975 | 0.321 | $-0.47 | -0.84% | -0.22 | $4554 | $-342 | $-532 | 0 | 0 | -0.093 | False | False | False | False |
| E2_init2.0 | 3906 | 0.977 | 0.348 | $-0.47 | -0.81% | -0.20 | $4915 | $-342 | $-514 | 0 | 0 | -0.082 | False | False | False | False |
| E3_ratchet | 3895 | 0.994 | 0.354 | $-0.13 | -0.21% | -0.04 | $4319 | $-342 | $-544 | 0 | 0 | -0.025 | False | False | False | False |
| E4_be1.0 | 4862 | 0.793 | 0.469 | $-3.18 | -7.65% | -1.94 | $15465 | $-342 | $-767 | 0 | 0 | -0.247 | False | False | False | False |
| E4_be1.5 | 4145 | 0.912 | 0.465 | $-1.64 | -3.10% | -0.80 | $7568 | $-342 | $-535 | 0 | 0 | -0.205 | False | False | False | False |
| E5_time16 | 3926 | 0.987 | 0.343 | $-0.27 | -0.46% | -0.10 | $4561 | $-342 | $-573 | 0 | 0 | -0.050 | False | False | False | False |
| E5_time48 | 3855 | 0.992 | 0.357 | $-0.17 | -0.28% | -0.06 | $4506 | $-342 | $-573 | 0 | 0 | -0.031 | False | False | False | False |
| F1_donch32 | 3145 | 0.981 | 0.357 | $-0.39 | -0.53% | -0.14 | $3014 | $-342 | $-419 | 0 | 0 | -0.088 | False | False | False | False |
| F1_donch48 | 2573 | 0.980 | 0.358 | $-0.41 | -0.46% | -0.13 | $3144 | $-273 | $-496 | 0 | 0 | -0.073 | False | False | False | False |
| F2_cap75 | 3163 | 1.037 | 0.361 | $0.65 | 0.87% | 0.31 | $3319 | $-342 | $-531 | 0 | 0 | 0.130 | False | False | False | False |
| F2_cap85 | 3632 | 1.027 | 0.359 | $0.53 | 0.82% | 0.25 | $3609 | $-342 | $-632 | 0 | 0 | 0.113 | False | False | False | False |
| C1_daily100 | 3569 | 0.993 | 0.355 | $-0.15 | -0.23% | -0.05 | $3301 | $-342 | $-373 | 226 | 912 | -0.034 | False | False | False | False |
| C1_daily150 | 3742 | 0.997 | 0.356 | $-0.05 | -0.08% | -0.00 | $3274 | $-342 | $-377 | 91 | 371 | -0.013 | False | False | False | False |
| C2_dd300 | 23 | 0.265 | 0.174 | $-12.76 | -0.13% | -0.83 | $368 | $-61 | $-88 | 1443 | 10754 | -0.172 | True | False | False | False |
| C2_dd500 | 90 | 0.712 | 0.300 | $-3.67 | -0.14% | -0.51 | $538 | $-61 | $-93 | 1418 | 10589 | -0.132 | True | False | False | False |
| C3_weekly250 | 3475 | 0.990 | 0.357 | $-0.19 | -0.29% | -0.07 | $4005 | $-342 | $-536 | 200 | 1129 | -0.036 | False | False | False | False |
| X_Ebest_C2best | 24 | 0.216 | 0.167 | $-12.84 | -0.13% | -0.88 | $358 | $-61 | $-86 | 1443 | 10754 | -0.185 | True | False | False | False |
| X_Fbest_C2best | 46 | 0.527 | 0.283 | $-5.38 | -0.11% | -0.62 | $322 | $-57 | $-88 | 1433 | 8657 | -0.166 | True | False | False | False |

## Multiple-testing / placebo

- DSR(base) = 0.0000 (p=1.0000); E[max SR] under N=100 = 3.334, observed base SR = -0.035.
- Random-entry placebo WITH chosen breaker (C2_dd500): real PF=0.712 at percentile 40.5% (p95=1.290, max=1.878); median placebo n=60 vs real 90.
- 2020 contribution — base: n=900, PF=1.037, PnL=$716; ref: n=0, PF=nan, PnL=$0.
- OOS NOT touched this round.

## 5min vs 15min comparison (IS, 1.5t)

| config | 5min n | 5min PF | 5min MaxDD$ | 15min n | 15min PF | 15min MaxDD$ |
|---|---|---|---|---|---|---|
| R0_base_1.5t | 3855 | 0.992 | $4506 | 1302 | 1.290 | $1772 |
| F1_donch32 | 3145 | 0.981 | $3014 | 1081 | 1.348 | $1532 |
| E3_ratchet | 3895 | 0.994 | $4319 | 1319 | 1.300 | $1843 |
| E5_time16 | 3926 | 0.987 | $4561 | 1325 | 1.298 | $1638 |
| C2_dd300 | 23 | 0.265 | $368 | 218 | 1.150 | $637 |
| C2_dd500 | 90 | 0.712 | $538 | 226 | 1.067 | $1015 |
