# RISK-FIRST REDESIGN — IS-only verdict

- Base = Donchian-20, trail 2.5xATR, vetoes {VR>=1, ATRpct<=90}, market, 1.5t, 1 contract, $50k.

- Pre-registered gate: MaxDD_usd@1c <= $1200 (hard), PF>=1.10, boot p<=0.10, expectancy>=$2, n>=300, worst trade <= $500, neighborhood MaxDD<=$1500.

- **Verdict: ZERO PASSERS** (IS gates at BOTH 1.5t and 3t).

## Why nothing passed

The pre-registered gate required MaxDD_usd@1c <= $1200 at BOTH 1.5t and 3t while keeping PF>=1.10 (boot p<=0.10), expectancy>=$2, n>=300 and worst trade>=-$500. Two disjoint failure modes emerged:

1. **Good-PF variants blow the DD gate.** Best non-breaker improvements (Donchian-32, ratchet-only, time-stop-16) reach PF 1.24-1.29 but only cut MaxDD to ~$1,730-$2,100, still far above the $1,200 gate (and $1,500 hard cap).
2. **The DD-halt breaker meets the DD gate but destroys the sample/edge.** C2 with T=300/500 pt halts new entries for ~1,185-1,190 days, cuts n from 1,302 to ~220, and drops PF to 0.97-1.05 with expectancy ~$0-$1 — failing gate 2 (PF/bootstrap) and gate 3 (n>=300, expectancy>=$2). The breaker does **not** manufacture PF; it lowers it.

### Key competitors (IS, 1.5t)

| config | n | PF | win | exp$ | CAGR$ | Sharpe$ | MaxDD$ | halted d | foregone | g1 (DD<=1200) | g2 (PF/boot) | g3 (exp/n) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0_base_1.5t | 1302 | 1.235 | 0.387 | $7.96 | 4.13% | 1.22 | $2304 | 0 | 0 | False | True | True |
| F1_donch32 | 1081 | 1.290 | 0.408 | $9.66 | 4.16% | 1.34 | $1731 | 0 | 0 | False | True | True |
| E3_ratchet | 1319 | 1.243 | 0.382 | $8.03 | 4.22% | 1.25 | $2097 | 0 | 0 | False | True | True |
| E5_time16 | 1328 | 1.238 | 0.370 | $7.79 | 4.13% | 1.22 | $1800 | 0 | 0 | False | True | True |
| C2_dd300 | 218 | 1.050 | 0.335 | $0.89 | 0.08% | 0.12 | $649 | 1190 | 3283 | True | False | False |
| C2_dd500 | 225 | 0.968 | 0.333 | $-0.64 | -0.06% | -0.07 | $1018 | 1185 | 3262 | True | False | False |
| X_Ebest_C2best | 220 | 1.077 | 0.336 | $1.30 | 0.12% | 0.17 | $623 | 1190 | 3283 | True | False | False |
| X_Fbest_C2best | 20 | 0.098 | 0.150 | $-25.96 | -0.22% | -1.15 | $617 | 1425 | 3145 | True | False | False |

**Honest statement: ZERO variants pass all IS gates.** The $1,500 risk budget (let alone the $1,200 buffer) is not attainable at 1 contract on a $50k account with this breakout+trail engine: the per-contract risk (base MaxDD $2,304 = 1,152 pt) is ~2x the budget, and the only mechanism that fits the budget (equity-DD halting) removes so much participation that the statistical edge collapses.

## Sample table (all variants, 1.5t)

| tag | n | PF | win | exp$ | CAGR$ | Sharpe$ | MaxDD$ | worst trade$ | worst day$ | halted d | foregone | MAR | g1 | g2 | g3 | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0_base_1.5t | 1302 | 1.235 | 0.387 | $7.96 | 4.13% | 1.22 | $2304 | $-289 | $-442 | 0 | 0 | 0.897 | False | True | True | False |
| E1_trail1.0 | 2677 | 0.455 | 0.161 | $-8.75 | -12.71% | -5.58 | $23573 | $-165 | $-412 | 0 | 0 | -0.270 | False | False | False | False |
| E1_trail1.5 | 1963 | 0.794 | 0.276 | $-4.62 | -4.21% | -1.53 | $9318 | $-260 | $-762 | 0 | 0 | -0.226 | False | False | False | False |
| E1_trail2.0 | 1562 | 1.065 | 0.349 | $1.83 | 1.20% | 0.40 | $2347 | $-343 | $-736 | 0 | 0 | 0.256 | False | False | False | False |
| E2_init1.0 | 1527 | 1.140 | 0.293 | $3.79 | 2.39% | 0.78 | $3061 | $-184 | $-708 | 0 | 0 | 0.390 | False | True | True | False |
| E2_init1.5 | 1361 | 1.217 | 0.361 | $6.90 | 3.77% | 1.16 | $3014 | $-274 | $-1057 | 0 | 0 | 0.626 | False | True | True | False |
| E2_init2.0 | 1317 | 1.217 | 0.380 | $7.27 | 3.84% | 1.14 | $3224 | $-260 | $-403 | 0 | 0 | 0.596 | False | True | True | False |
| E3_ratchet | 1319 | 1.243 | 0.382 | $8.03 | 4.22% | 1.25 | $2097 | $-285 | $-436 | 0 | 0 | 1.006 | False | True | True | False |
| E4_be1.0 | 1664 | 0.910 | 0.234 | $-2.36 | -1.75% | -0.50 | $6974 | $-437 | $-452 | 0 | 0 | -0.125 | False | False | False | False |
| E4_be1.5 | 1401 | 1.132 | 0.335 | $4.05 | 2.34% | 0.71 | $4336 | $-289 | $-442 | 0 | 0 | 0.270 | False | True | True | False |
| E5_time16 | 1328 | 1.238 | 0.370 | $7.79 | 4.13% | 1.22 | $1800 | $-289 | $-442 | 0 | 0 | 1.146 | False | True | True | False |
| E5_time48 | 1302 | 1.235 | 0.387 | $7.96 | 4.13% | 1.22 | $2304 | $-289 | $-442 | 0 | 0 | 0.897 | False | True | True | False |
| F1_donch32 | 1081 | 1.290 | 0.408 | $9.66 | 4.16% | 1.34 | $1731 | $-289 | $-382 | 0 | 0 | 1.203 | False | True | True | False |
| F1_donch48 | 928 | 1.236 | 0.381 | $8.24 | 3.11% | 1.06 | $2949 | $-302 | $-382 | 0 | 0 | 0.527 | False | True | True | False |
| F2_cap75 | 1138 | 1.123 | 0.378 | $3.91 | 1.85% | 0.63 | $2937 | $-437 | $-653 | 0 | 0 | 0.315 | False | True | True | False |
| F2_cap85 | 1258 | 1.213 | 0.382 | $7.05 | 3.57% | 1.09 | $2702 | $-261 | $-419 | 0 | 0 | 0.661 | False | True | True | False |
| C1_daily100 | 1287 | 1.229 | 0.388 | $7.69 | 3.96% | 1.19 | $3195 | $-289 | $-442 | 43 | 41 | 0.620 | False | True | True | False |
| C1_daily150 | 1301 | 1.236 | 0.387 | $7.98 | 4.14% | 1.22 | $2358 | $-289 | $-442 | 11 | 3 | 0.879 | False | True | True | False |
| C2_dd300 | 218 | 1.050 | 0.335 | $0.89 | 0.08% | 0.12 | $649 | $-152 | $-183 | 1190 | 3283 | 0.064 | True | False | False | False |
| C2_dd500 | 225 | 0.968 | 0.333 | $-0.64 | -0.06% | -0.07 | $1018 | $-260 | $-251 | 1185 | 3262 | -0.030 | True | False | False | False |
| C3_weekly250 | 1292 | 1.210 | 0.386 | $7.13 | 3.70% | 1.10 | $3454 | $-289 | $-442 | 15 | 26 | 0.536 | False | True | True | False |
| X_Ebest_C2best | 220 | 1.077 | 0.336 | $1.30 | 0.12% | 0.17 | $623 | $-152 | $-177 | 1190 | 3283 | 0.099 | True | False | False | False |
| X_Fbest_C2best | 20 | 0.098 | 0.150 | $-25.96 | -0.22% | -1.15 | $617 | $-72 | $-77 | 1425 | 3145 | -0.182 | True | False | False | False |

## Multiple-testing / placebo

- DSR(base) = 0.0002 (p=0.9998); E[max SR] under N=100 = 2.868, observed base SR = 1.229.
- Random-entry placebo WITH chosen breaker (C2_dd500): real PF=0.968 at percentile 71.4% (p95=1.299, max=1.870); median placebo n=91 vs real 225.
- 2020 contribution — base: n=300, PF=1.312, PnL=$3276; ref: n=50, PF=0.754, PnL=$-424.
- OOS NOT touched this round.
