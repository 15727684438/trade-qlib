# STRATEGY V2B — EMA20/144 regime + Donchian breakout, long+short, 2-leg partial (IS-only verdict)

- 5min decisions / 1min execution; regime LONG (EMA20>EMA144 & close>EMA144) / SHORT (EMA20<EMA144 & close<EMA144); anchor T2 Donchian-20; gates {VR<1, ATRpct<=90, ext<=2.5}; size 2c; partial 0.5R (1 of 2), runner trails structurally (pivot n=2, buf 0.25 ATR, ratchet-only, floor=initial stop) + regime-break exit.

- Fill model: canonical adverse slip = slip_ticks*tick on entry/stop/regime/partial; partial carries >=1 tick. Costs: $0.50/side + $0.20/side slip, $2/pt, tick $0.50.

- Cost model corrected 2026-10-10: base ('1.5t') = $0.50/side comm + $0.20/side slip; stress ('3.0t') = $0.50/side + $1.50/side slip.

- Acceptance: rank by net_profit_usd@1.5t among configs with MaxDD_usd@2c <= $1500 at BOTH 1.5t and 3.0t.

- **Verdict: NONE meets MaxDD gate at both cost levels**

## Grid (IS) — all configs run at 1.5t and 3.0t

| tag | cost | n | PF | win | exp$ | net$ | Sharpe$ | MaxDD$ | worst$ | L n/PF/net$ | S n/PF/net$ | TP1 | win w/partial | win w/o partial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| anchor | 1.5t | 972 | 1.016 | 0.500 | $0.79 | $770 | 0.10 | $3996 | $-511 | 628/0.961/$-1205 | 344/1.105/$1975 | 603 | 0.804 | 0.003 |
| anchor | 3.0t | 973 | 0.930 | 0.482 | $-3.70 | $-3600 | -0.37 | $5987 | $-513 | 628/0.871/$-4159 | 345/1.029/$559 | 597 | 0.784 | 0.003 |
| anchor_long | 1.5t | 628 | 0.961 | 0.489 | $-1.92 | $-1205 | -0.16 | $3100 | $-511 | 628/0.961/$-1205 | 0/nan/$0 | 375 | 0.816 | 0.004 |
| anchor_long | 3.0t | 628 | 0.871 | 0.471 | $-6.62 | $-4159 | -0.59 | $4907 | $-513 | 628/0.871/$-4159 | 0/nan/$0 | 372 | 0.793 | 0.004 |
| T1 | 1.5t | 2123 | 0.974 | 0.459 | $-1.25 | $-2648 | -0.19 | $9449 | $-410 | 1233/1.030/$1594 | 890/0.911/$-4242 | 1212 | 0.804 | 0.000 |
| T1 | 3.0t | 2124 | 0.878 | 0.438 | $-6.19 | $-13143 | -1.03 | $14970 | $-415 | 1233/0.920/$-4580 | 891/0.830/$-8563 | 1186 | 0.785 | 0.000 |
| T3 | 1.5t | 1221 | 1.039 | 0.502 | $1.92 | $2343 | 0.26 | $4078 | $-511 | 745/1.091/$2989 | 476/0.976/$-646 | 757 | 0.807 | 0.004 |
| T3 | 3.0t | 1222 | 0.948 | 0.482 | $-2.64 | $-3230 | -0.31 | $6181 | $-513 | 745/0.980/$-687 | 477/0.908/$-2543 | 747 | 0.787 | 0.002 |
| partial_1.0R | 1.5t | 972 | 1.066 | 0.441 | $3.96 | $3848 | 0.39 | $4152 | $-511 | 628/1.011/$382 | 344/1.154/$3466 | 414 | 0.949 | 0.065 |
| partial_1.0R | 3.0t | 973 | 0.977 | 0.417 | $-1.45 | $-1408 | -0.11 | $6044 | $-513 | 628/0.914/$-3227 | 345/1.078/$1819 | 399 | 0.940 | 0.054 |
| floor_1.5 | 1.5t | 1009 | 0.986 | 0.431 | $-0.59 | $-592 | -0.06 | $4587 | $-400 | 651/0.924/$-2080 | 358/1.091/$1488 | 647 | 0.672 | 0.000 |
| floor_1.5 | 3.0t | 1011 | 0.894 | 0.414 | $-4.89 | $-4940 | -0.59 | $6436 | $-402 | 652/0.841/$-4616 | 359/0.982/$-323 | 630 | 0.665 | 0.000 |
| rth_off | 1.5t | 3656 | 0.980 | 0.446 | $-0.68 | $-2478 | -0.16 | $8905 | $-684 | 2262/0.967/$-2378 | 1394/0.998/$-99 | 2291 | 0.710 | 0.003 |
| rth_off | 3.0t | 3682 | 0.854 | 0.409 | $-5.44 | $-20033 | -1.53 | $20178 | $-689 | 2279/0.829/$-13645 | 1403/0.889/$-6388 | 2213 | 0.680 | 0.002 |
| one_contract | 1.5t | 972 | 1.072 | 0.377 | $2.36 | $2298 | 0.35 | $2431 | $-255 | 628/1.034/$667 | 344/1.129/$1630 | 0 | — | 0.377 |
| one_contract | 3.0t | 973 | 0.991 | 0.355 | $-0.31 | $-306 | -0.03 | $3075 | $-257 | 628/0.945/$-1119 | 345/1.062/$813 | 0 | — | 0.355 |

## Acceptance (MaxDD@2c <= $1500 at BOTH costs)

| tag | size | net@1.5t$ | MaxDD@1.5t$ | MaxDD@3t$ | eligible |
|---|---|---|---|---|---|
| anchor | 2 | $770 | $3996 | $5987 | no |
| anchor_long | 2 | $-1205 | $3100 | $4907 | no |
| T1 | 2 | $-2648 | $9449 | $14970 | no |
| T3 | 2 | $2343 | $4078 | $6181 | no |
| partial_1.0R | 2 | $3848 | $4152 | $6044 | no |
| floor_1.5 | 2 | $-592 | $4587 | $6436 | no |
| rth_off | 2 | $-2478 | $8905 | $20178 | no |
| one_contract | 1 | $2298 | $2431 | $3075 | no |

**No config satisfies the MaxDD@2c<=$1500 gate at both cost levels.**

## Anchor check (STEP 1)

- `run_ema2_struct` with shorts+partial disabled and legacy fills reproduces frozen A_T2 trade-for-trade (n=774, PF 1.202, net $5282.93, MaxDD $1327.89, max per-trade diff $0.00) — see `results/ema2_anchor_check.csv`.

- OOS NOT touched.
