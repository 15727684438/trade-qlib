# STRATEGY V2B — EMA20/144 regime + Donchian breakout, long+short, 2-leg partial (IS-only verdict)

- 5min decisions / 1min execution; regime LONG (EMA20>EMA144 & close>EMA144) / SHORT (EMA20<EMA144 & close<EMA144); anchor T2 Donchian-20; gates {VR<1, ATRpct<=90, ext<=2.5}; size 2c; partial 0.5R (1 of 2), runner trails structurally (pivot n=2, buf 0.25 ATR, ratchet-only, floor=initial stop) + regime-break exit.

- Fill model: canonical adverse slip = slip_ticks*tick on entry/stop/regime/partial; partial carries >=1 tick. Costs: $0.75/side, $2/pt, tick $0.50.

- Acceptance: rank by net_profit_usd@1.5t among configs with MaxDD_usd@2c <= $1500 at BOTH 1.5t and 3.0t.

- **Verdict: NONE meets MaxDD gate at both cost levels**

## Grid (IS) — all configs run at 1.5t and 3.0t

| tag | cost | n | PF | win | exp$ | net$ | Sharpe$ | MaxDD$ | worst$ | L n/PF/net$ | S n/PF/net$ | TP1 | win w/partial | win w/o partial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| anchor | 1.5t | 973 | 0.961 | 0.490 | $-2.04 | $-1985 | -0.20 | $5158 | $-513 | 628/0.905/$-2984 | 345/1.052/$999 | 600 | 0.793 | 0.003 |
| anchor | 3.0t | 973 | 0.912 | 0.477 | $-4.70 | $-4573 | -0.48 | $6387 | $-514 | 628/0.853/$-4787 | 345/1.011/$214 | 597 | 0.776 | 0.003 |
| anchor_long | 1.5t | 628 | 0.905 | 0.479 | $-4.75 | $-2984 | -0.42 | $4202 | $-513 | 628/0.905/$-2984 | 0/nan/$0 | 374 | 0.802 | 0.004 |
| anchor_long | 3.0t | 628 | 0.853 | 0.465 | $-7.62 | $-4787 | -0.68 | $5418 | $-514 | 628/0.853/$-4787 | 0/nan/$0 | 372 | 0.782 | 0.004 |
| T1 | 1.5t | 2123 | 0.917 | 0.445 | $-4.11 | $-8733 | -0.68 | $12879 | $-414 | 1233/0.966/$-1922 | 890/0.861/$-6810 | 1201 | 0.787 | 0.000 |
| T1 | 3.0t | 2124 | 0.860 | 0.433 | $-7.19 | $-15267 | -1.20 | $16291 | $-416 | 1233/0.900/$-5813 | 891/0.814/$-9454 | 1186 | 0.776 | 0.000 |
| T3 | 1.5t | 1222 | 0.981 | 0.491 | $-0.97 | $-1191 | -0.10 | $5483 | $-513 | 745/1.021/$722 | 477/0.930/$-1913 | 752 | 0.797 | 0.002 |
| T3 | 3.0t | 1222 | 0.929 | 0.476 | $-3.64 | $-4452 | -0.43 | $6666 | $-514 | 745/0.959/$-1432 | 477/0.891/$-3020 | 747 | 0.779 | 0.000 |
| partial_1.0R | 1.5t | 973 | 1.013 | 0.427 | $0.82 | $799 | 0.10 | $5217 | $-513 | 628/0.953/$-1731 | 345/1.110/$2530 | 408 | 0.941 | 0.055 |
| partial_1.0R | 3.0t | 973 | 0.961 | 0.415 | $-2.45 | $-2381 | -0.20 | $6445 | $-514 | 628/0.899/$-3855 | 345/1.062/$1474 | 399 | 0.940 | 0.051 |
| floor_1.5 | 1.5t | 1009 | 0.923 | 0.421 | $-3.47 | $-3500 | -0.41 | $5793 | $-402 | 651/0.867/$-3798 | 358/1.017/$298 | 636 | 0.668 | 0.000 |
| floor_1.5 | 3.0t | 1011 | 0.874 | 0.411 | $-5.89 | $-5951 | -0.71 | $6854 | $-403 | 652/0.821/$-5268 | 359/0.962/$-682 | 630 | 0.660 | 0.000 |
| rth_off | 1.5t | 3664 | 0.902 | 0.422 | $-3.55 | $-12990 | -0.98 | $14849 | $-687 | 2266/0.884/$-8869 | 1398/0.926/$-4121 | 2257 | 0.684 | 0.002 |
| rth_off | 3.0t | 3682 | 0.830 | 0.403 | $-6.44 | $-23715 | -1.82 | $23848 | $-690 | 2279/0.804/$-15924 | 1403/0.866/$-7791 | 2213 | 0.670 | 0.002 |
| one_contract | 1.5t | 973 | 1.021 | 0.362 | $0.72 | $705 | 0.12 | $2696 | $-256 | 628/0.981/$-379 | 345/1.083/$1084 | 0 | — | 0.362 |
| one_contract | 3.0t | 973 | 0.977 | 0.350 | $-0.81 | $-793 | -0.10 | $3275 | $-257 | 628/0.930/$-1433 | 345/1.048/$640 | 0 | — | 0.350 |

## Acceptance (MaxDD@2c <= $1500 at BOTH costs)

| tag | size | net@1.5t$ | MaxDD@1.5t$ | MaxDD@3t$ | eligible |
|---|---|---|---|---|---|
| anchor | 2 | $-1985 | $5158 | $6387 | no |
| anchor_long | 2 | $-2984 | $4202 | $5418 | no |
| T1 | 2 | $-8733 | $12879 | $16291 | no |
| T3 | 2 | $-1191 | $5483 | $6666 | no |
| partial_1.0R | 2 | $799 | $5217 | $6445 | no |
| floor_1.5 | 2 | $-3500 | $5793 | $6854 | no |
| rth_off | 2 | $-12990 | $14849 | $23848 | no |
| one_contract | 1 | $705 | $2696 | $3275 | no |

**No config satisfies the MaxDD@2c<=$1500 gate at both cost levels.**

## Anchor check (STEP 1)

- `run_ema2_struct` with shorts+partial disabled and legacy fills reproduces frozen A_T2 trade-for-trade (n=774, PF 1.185, net $4895.93, MaxDD $1361.89, max per-trade diff $0.00) — see `results/ema2_anchor_check.csv`.

- OOS NOT touched.
