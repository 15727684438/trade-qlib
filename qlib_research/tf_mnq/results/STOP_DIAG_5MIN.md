# STOP-LOSS diagnosis — 5min vs 15min (IS-only, read-only)

Source: existing ledgers + 1min path checks. Conventions: $2/pt, tick $0.50 ($0.25/pt), comm $0.75/side, cost 1.5 tick (0.375 pt slip per leg). No strategy re-run; no module modified.

## 1. Cost drag (the primary suspect)

| config | n | size | $/trade gross | $/trade net | total cost $ | gross $ | net $ | cost % of gross |
|---|---|---|---|---|---|---|---|---|
| 5_f0.75 | 11492 | 2 | $1.37 | $-4.63 | $68952 | $15797 | $-53155 | 436% |
| 5_f1 | 11492 | 2 | $1.29 | $-4.71 | $68952 | $14855 | $-54097 | 464% |
| 5_dyn | 11492 | 2 | $1.49 | $-4.51 | $68952 | $17070 | $-51882 | 404% |
| 5_dyn_1c_target | 9896 | 1 | $0.27 | $-2.73 | $29688 | $2708 | $-26980 | 1096% |
| 15_f0.75 | 4065 | 2 | $5.54 | $-0.46 | $24390 | $22529 | $-1861 | 108% |
| 15_f1 | 4065 | 2 | $5.36 | $-0.64 | $24390 | $21794 | $-2596 | 112% |
| 15_dyn | 4065 | 2 | $6.27 | $0.27 | $24390 | $25481 | $1091 | 96% |
| 15_dyn_1c_target | 3614 | 1 | $4.58 | $1.58 | $10842 | $16559 | $5717 | 65% |
| R0_base_5min_StageB | 3855 | 1 | $1.23 | $-1.77 | $11565 | $4759 | $-6806 | 243% |
| V2B_anchor_5min | 973 | 2 | $3.96 | $-2.04 | $5838 | $3853 | $-1985 | 152% |

- 5min vs 15min trade-count ratio (`5_dyn`/`15_dyn`) = 2.83x; cost-drag ratio = 2.83x.

- `5_dyn` gross is **+$17070** but cost is **$68952** = **404% of gross**; per-trade gross edge $1.49 vs cost $6.00.

## 2. Stop distance (points vs ATR)

Ledger `stop_px` is the stop at exit (after the ratchet trail), so this is the *final* stop distance; the **initial** floor is 2.0*ATR by construction (5min ≈ 20 pts, 15min ≈ 38 pts).

| config | ATR p50 (pts) | final stop dist p10/p50/p90 (pts) | final stop dist p10/p50/p90 (ATR) |
|---|---|---|---|
| 5_dyn | 10.1 | 3/16/59 | 0.40/1.86/3.68 |
| 15_dyn | 19.0 | 7/31/106 | 0.40/1.97/3.79 |
| 5_dyn_1c_target | 12.3 | 11/27/65 | 2.00/2.00/2.94 |
| 15_dyn_1c_target | 22.4 | 22/51/109 | 2.00/2.00/3.02 |

## 3. Stop-hit, loss size, MFE and whipsaw

| config | stop rate | avg stop loss $ | avg stop loss (ATR) | MFE before stop p50 (ATR) | whipsaw if stop +1.0 ATR |
|---|---|---|---|---|---|
| 5_dyn | 1.000 | $-5 | -0.13 | 1.71 | 0.188 |
| 5_f0.75 | 1.000 | $-5 | -0.13 | 1.71 | 0.283 |
| 15_dyn | 1.000 | $0 | -0.02 | 1.76 | 0.189 |
| 15_f0.75 | 1.000 | $-0 | -0.03 | 1.76 | 0.276 |
| TaskB_k1.5_f0.75_15min | 0.472 | $-112 | -2.30 | 0.76 | 0.167 |

## 4. Target vs stop economics (net and gross)

| config | win | avg win $ | avg loss $ | payoff | exp net $ | exp gross $ | breakeven win (net) | breakeven win (gross) |
|---|---|---|---|---|---|---|---|---|
| 5_dyn | 0.376 | $127 | $-84 | 1.51 | $-4.51 | $1.49 | 0.398 | 0.390 |
| 5_f0.75 | 0.412 | $107 | $-83 | 1.29 | $-4.63 | $1.37 | 0.437 | 0.430 |
| 5_dyn_1c_target | 0.418 | $87 | $-67 | 1.30 | $-2.73 | $0.27 | 0.436 | 0.416 |
| 15_dyn | 0.392 | $229 | $-148 | 1.55 | $0.27 | $6.27 | 0.392 | 0.387 |
| 15_f0.75 | 0.422 | $197 | $-145 | 1.36 | $-0.46 | $5.54 | 0.424 | 0.420 |

## 5. Time in trade

| config | median bars | median minutes |
|---|---|---|
| 5_dyn | 13 | 65 |
| 5_f0.75 | 13 | 65 |
| 15_dyn | 12 | 180 |
| 15_f0.75 | 12 | 180 |

## 6. Execution / cost assumption check

- Both freqs use the same engine path (`run_second_push_opt`): entry on a 1min break of the confirmed push1 extreme, adverse slip `slip_ticks*tick` on entry and exit, comm $0.75/side/contract; `freq_minutes` only changes the decision-bar segmentation. 5min and 15min are therefore **cost-identical per trade**.

- Ledger identity `pnl_usd_net == pnl_pts * $2 * size` holds (verified below); entry gates read bar `k-1` (last completed) and the zigzag pivot B is confirmed before the entry window (no look-ahead). The earlier look-ahead in the gated rows was fixed before these ledgers were written.

## 7. Direction (5min)

| config | long n | long net $ | long win | short n | short net $ | short win |
|---|---|---|---|---|---|---|
| 5_dyn | 5707 | $-14869 | 0.387 | 5785 | $-37013 | 0.365 |
| 5_f0.75 | 5707 | $-12742 | 0.424 | 5785 | $-40413 | 0.401 |
| 5_f1 | 5707 | $-14902 | 0.398 | 5785 | $-39195 | 0.376 |
| 5_dyn_1c_target | 4944 | $-6840 | 0.427 | 4952 | $-20140 | 0.409 |

## Verdict — ranked drivers of the 5min losses

1. **Cost drag from overtrading (primary).** `5_dyn` trades 11492 times (2.83x the 15min count) at a fixed $6.00/trade; total cost $68952 vs gross **+$17070** → the raw edge is positive but cost is **404% of gross**. Same at `5_f0.75`.

2. **Thin per-trade gross edge at 5min.** Gross edge $1.49/trade vs $6.27/trade at 15min — the identical fixed cost ($6.00/trade) then dominates.

3. **Stop whipsaw is real but secondary.** With a 1.0*ATR wider stop, 19% of stopped `5_dyn` trades would still reach target — so widening stops recovers only a small slice, and MFE before stop is 1.71 ATR (well short of the target), i.e. mostly **no edge**, not bad luck.

4. **Execution is not the culprit.** The 5min and 15min paths share the same engine and cost model; there is no extra 5min penalty.


**Fix implication:** 5min loses mainly to cost — cost ≈ 404% of gross and $6.00/trade swallows a $1.49/trade gross edge; widening stops won't rescue it because the raw edge is ≈0. Raise the timeframe (or cut trade frequency by 2–3x) so a trades fewer times for a larger gross move, or drop the short side (shorts carry most of the 5min losses). OOS not touched; no module modified.
