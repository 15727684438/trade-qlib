# STOP-LOSS diagnosis — 5min vs 15min (IS-only, read-only)

Source: existing ledgers + 1min path checks. Conventions: $2/pt, tick $0.50 ($0.25/pt), comm $0.50/side, slip 0.4 tick (0.10 pt = $0.20 per leg). No strategy re-run; no module modified.

## 1. Cost drag (the primary suspect)

| config | n | size | $/trade gross | $/trade net | total cost $ | gross $ | net $ | cost % of gross |
|---|---|---|---|---|---|---|---|---|
| 5_f0.75 | 11487 | 2 | $1.23 | $-1.57 | $32164 | $14089 | $-18075 | 228% |
| 5_f1 | 11487 | 2 | $1.16 | $-1.64 | $32164 | $13359 | $-18805 | 241% |
| 5_dyn | 11487 | 2 | $1.33 | $-1.47 | $32164 | $15308 | $-16855 | 210% |
| 5_dyn_1c_target | 9961 | 1 | $0.35 | $-1.05 | $13945 | $3464 | $-10482 | 403% |
| 15_f0.75 | 4065 | 2 | $5.25 | $2.45 | $11382 | $21341 | $9959 | 53% |
| 15_f1 | 4065 | 2 | $5.10 | $2.30 | $11382 | $20726 | $9344 | 55% |
| 15_dyn | 4065 | 2 | $5.98 | $3.18 | $11382 | $24320 | $12938 | 47% |
| 15_dyn_1c_target | 3620 | 1 | $4.34 | $2.94 | $5068 | $15710 | $10642 | 32% |
| R0_base_5min_StageB | 3855 | 1 | $1.23 | $-0.17 | $5397 | $4759 | $-638 | 113% |
| V2B_anchor_5min | 972 | 2 | $3.59 | $0.79 | $2722 | $3492 | $770 | 78% |

- 5min vs 15min trade-count ratio (`5_dyn`/`15_dyn`) = 2.83x; cost-drag ratio = 2.83x.

- `5_dyn` gross is **+$15308** but cost is **$32164** = **210% of gross**; per-trade gross edge $1.33 vs cost $2.80.

## 2. Stop distance (points vs ATR)

Ledger `stop_px` is the stop at exit (after the ratchet trail), so this is the *final* stop distance; the **initial** floor is 2.0*ATR by construction (5min ≈ 20 pts, 15min ≈ 38 pts).

| config | ATR p50 (pts) | final stop dist p10/p50/p90 (pts) | final stop dist p10/p50/p90 (ATR) |
|---|---|---|---|
| 5_dyn | 10.1 | 3/16/59 | 0.38/1.83/3.69 |
| 15_dyn | 19.0 | 7/31/106 | 0.40/1.95/3.80 |
| 5_dyn_1c_target | 12.2 | 11/27/65 | 2.00/2.00/2.91 |
| 15_dyn_1c_target | 22.4 | 22/51/109 | 2.00/2.00/3.01 |

## 3. Stop-hit, loss size, MFE and whipsaw

| config | stop rate | avg stop loss $ | avg stop loss (ATR) | MFE before stop p50 (ATR) | whipsaw if stop +1.0 ATR |
|---|---|---|---|---|---|
| 5_dyn | 1.000 | $-1 | -0.03 | 1.79 | 0.186 |
| 5_f0.75 | 1.000 | $-2 | -0.04 | 1.79 | 0.280 |
| 15_dyn | 1.000 | $3 | 0.03 | 1.77 | 0.190 |
| 15_f0.75 | 1.000 | $2 | 0.02 | 1.77 | 0.278 |
| TaskB_k1.5_f0.75_15min | 0.468 | $-111 | -2.27 | 0.77 | 0.166 |

## 4. Target vs stop economics (net and gross)

| config | win | avg win $ | avg loss $ | payoff | exp net $ | exp gross $ | breakeven win (net) | breakeven win (gross) |
|---|---|---|---|---|---|---|---|---|
| 5_dyn | 0.391 | $124 | $-82 | 1.51 | $-1.47 | $1.33 | 0.398 | 0.392 |
| 5_f0.75 | 0.428 | $106 | $-82 | 1.29 | $-1.57 | $1.23 | 0.436 | 0.432 |
| 5_dyn_1c_target | 0.422 | $87 | $-66 | 1.33 | $-1.05 | $0.35 | 0.429 | 0.420 |
| 15_dyn | 0.401 | $226 | $-146 | 1.55 | $3.18 | $5.98 | 0.392 | 0.389 |
| 15_f0.75 | 0.430 | $196 | $-143 | 1.37 | $2.45 | $5.25 | 0.422 | 0.421 |

## 5. Time in trade

| config | median bars | median minutes |
|---|---|---|
| 5_dyn | 13 | 65 |
| 5_f0.75 | 13 | 65 |
| 15_dyn | 12 | 180 |
| 15_f0.75 | 12 | 180 |

## 6. Execution / cost assumption check

- Both freqs use the same engine path (`run_second_push_opt`): entry on a 1min break of the confirmed push1 extreme, adverse slip `slip_ticks*tick` on entry and exit, comm $0.50/side/contract; `freq_minutes` only changes the decision-bar segmentation. 5min and 15min are therefore **cost-identical per trade**.

- Ledger identity `pnl_usd_net == pnl_pts * $2 * size` holds (verified below); entry gates read bar `k-1` (last completed) and the zigzag pivot B is confirmed before the entry window (no look-ahead). The earlier look-ahead in the gated rows was fixed before these ledgers were written.

## 7. Direction (5min)

| config | long n | long net $ | long win | short n | short net $ | short win |
|---|---|---|---|---|---|---|
| 5_dyn | 5706 | $2801 | 0.404 | 5781 | $-19656 | 0.378 |
| 5_f0.75 | 5706 | $4975 | 0.441 | 5781 | $-23050 | 0.415 |
| 5_f1 | 5706 | $2737 | 0.416 | 5781 | $-21542 | 0.389 |
| 5_dyn_1c_target | 4971 | $1872 | 0.432 | 4990 | $-12354 | 0.412 |

## Verdict — ranked drivers of the 5min losses

1. **Cost drag from overtrading (primary).** `5_dyn` trades 11487 times (2.83x the 15min count) at a fixed $2.80/trade; total cost $32164 vs gross **+$15308** → the raw edge is positive but cost is **210% of gross**. Same at `5_f0.75`.

2. **Thin per-trade gross edge at 5min.** Gross edge $1.33/trade vs $5.98/trade at 15min — the identical fixed cost ($2.80/trade) then dominates.

3. **Stop whipsaw is real but secondary.** With a 1.0*ATR wider stop, 19% of stopped `5_dyn` trades would still reach target — so widening stops recovers only a small slice, and MFE before stop is 1.79 ATR (well short of the target), i.e. mostly **no edge**, not bad luck.

4. **Execution is not the culprit.** The 5min and 15min paths share the same engine and cost model; there is no extra 5min penalty.


**Fix implication:** 5min loses mainly to cost — cost ≈ 210% of gross and $2.80/trade swallows a $1.33/trade gross edge; widening stops won't rescue it because the raw edge is ≈0. Raise the timeframe (or cut trade frequency by 2–3x) so a trades fewer times for a larger gross move, or drop the short side (shorts carry most of the 5min losses). OOS not touched; no module modified.
