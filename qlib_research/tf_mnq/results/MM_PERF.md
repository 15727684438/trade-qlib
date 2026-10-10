# MM PERF — what happens after a Measured-Move setup? (IS only, 2019-05-06..2023-12-31)

Reuses the causal MM events (`measured_move_events.csv`, tol=0.40, n_candles>=3). Forward measurement starts on the **next bar** after the combination's last candle (ref = next open). Costs: $2/point/ct + $0.50/side commission; the project $0.20/side slippage is shown as a separate `net$full`. `net$comm` = gross - $1.00 round trip. OOS (2024+) untouched.

Cost model corrected 2026-10-10: commission $0.50/side ($1.00 open+close) + slippage $0.20/side; '1.5t' label retained, stress '3.0t' = $1.50/side slip.

## Direct answer

**No potential for profit was found at any horizon or condition that is robust across both timeframes.** The MM setup is followed by a **flat-to-negative** forward return; continuation and reversal are a coin flip; and at longer horizons the setup under-performs a session/regime/direction-matched random entry. Details:

- **15min** — H20: MM -1.65pt (net$-4.29) vs random +0.10pt vs long-drift +1.57pt; H100: MM -2.12pt (net$-5.23) vs random -2.25pt vs long-drift +7.62pt; H400: MM -2.29pt (net$-5.58) vs random +8.06pt vs long-drift +30.31pt. P(reach 1xD within 20)=0.592; P(cont)/P(rev) @100=0.497/0.492.
- **5min** — H20: MM -2.18pt (net$-5.36) vs random -0.92pt vs long-drift +0.47pt; H100: MM -1.64pt (net$-4.28) vs random +0.55pt vs long-drift +2.47pt; H400: MM +2.19pt (net$+3.39) vs random +0.26pt vs long-drift +9.41pt. P(reach 1xD within 20)=0.561; P(cont)/P(rev) @100=0.460/0.521.

The **short side is consistently negative** in both timeframes (a drift artefact of the 2019-2023 bull market, not a pattern effect). Positive cells are almost all long-side and simply track the drift; only one narrow cell beats its matched random at every horizon (see robustness scan).

## Robustness scan — cells that beat the matched random baseline at H=20 AND H=100 AND H=400 (and net$comm>0)

### 15min

| dimension | stratum | n | MM-rnd H20 | MM-rnd H100 | MM-rnd H400 | net$ H20/H100/H400 |
|---|---|---|---|---|---|---|
| n_candles | 4 | 211 | +4.43 | +29.38 | +25.55 | +6.6/+19.0/+11.9 |

### 5min

_none_

Cross-timeframe note: the only 15min candidate (n_candles=4, n=212) is **not** confirmed on 5min, so it is treated as isolated/overfit.

## 15min

### Overall: MM forward return vs matched random vs drift

| horizon | MM n | MM ret mean | MM ret med | random ret mean | drift(long) | MM net$comm | MM net$full |
|---|---|---|---|---|---|---|---|
| 5 | 1615 | -1.72 | -1.25 | 0.78 | 0.41 | -4.43 | -4.83 |
| 10 | 1615 | -1.85 | -1.75 | 2.42 | 0.82 | -4.71 | -5.11 |
| 20 | 1614 | -1.65 | -1.00 | 0.10 | 1.57 | -4.29 | -4.69 |
| 50 | 1614 | -2.23 | -2.00 | -1.74 | 3.86 | -5.46 | -5.86 |
| 100 | 1612 | -2.12 | -3.12 | -2.25 | 7.62 | -5.23 | -5.63 |
| 200 | 1609 | 0.73 | 2.00 | 0.18 | 15.18 | 0.45 | 0.05 |
| 400 | 1606 | -2.29 | 3.25 | 8.06 | 30.31 | -5.58 | -5.98 |
| rev | 1615 | -1.13 | -5.50 | 0.53 | — | -3.27 | — |
| end | 1615 | 267.97 | 475.75 | 211.52 | — | 534.93 | — |

### Condition scan, horizon=20 bars (15min)

| dimension | stratum | n | MM ret mean | random ret mean | MM-rnd | net$comm | win | p_cont | p_rev |
|---|---|---|---|---|---|---|---|---|---|
| direction | long | 862 | -1.50 | 0.22 | -1.73 | -4.00 | 0.52 | 0.44 | 0.42 |
| direction | short | 752 | -1.81 | -0.04 | -1.77 | -4.63 | 0.45 | 0.44 | 0.45 |
| session | Asia | 509 | 1.94 | 0.85 | 1.09 | 2.88 | 0.49 | 0.45 | 0.46 |
| session | Europe | 499 | -8.23 | 8.38 | -16.61 | -17.45 | 0.45 | 0.49 | 0.47 |
| session | US | 551 | 3.10 | -7.62 | 10.72 | 5.20 | 0.53 | 0.40 | 0.38 |
| session | Post | 55 | -22.72 | -4.49 | -18.24 | -46.45 | 0.35 | 0.27 | 0.49 |
| regime | low | 541 | -6.24 | 5.64 | -11.88 | -13.48 | 0.47 | 0.47 | 0.47 |
| regime | mid | 537 | 0.38 | -0.66 | 1.04 | -0.24 | 0.50 | 0.44 | 0.45 |
| regime | high | 534 | 0.93 | -4.97 | 5.90 | 0.85 | 0.50 | 0.41 | 0.38 |
| size_pts_band | low | 547 | -2.59 | 3.07 | -5.66 | -6.18 | 0.48 | 0.52 | 0.47 |
| size_pts_band | mid | 532 | 2.15 | 2.49 | -0.34 | 3.30 | 0.49 | 0.45 | 0.47 |
| size_pts_band | high | 535 | -4.46 | -5.31 | 0.85 | -9.92 | 0.49 | 0.34 | 0.37 |
| size_terc | low | 537 | 1.02 | 1.81 | -0.79 | 1.03 | 0.50 | 0.53 | 0.46 |
| size_terc | mid | 539 | -5.03 | 1.13 | -6.17 | -11.06 | 0.46 | 0.43 | 0.48 |
| size_terc | high | 538 | -0.92 | -2.63 | 1.72 | -2.83 | 0.51 | 0.36 | 0.37 |
| n_candles | 3 | 1378 | -2.54 | 0.78 | -3.32 | -6.09 | 0.48 | 0.44 | 0.45 |
| n_candles | 4 | 212 | 3.81 | -0.62 | 4.43 | 6.62 | 0.51 | 0.42 | 0.38 |
| n_candles | 5+ | 24 | 1.68 | -32.54 | 34.22 | 2.35 | 0.58 | 0.46 | 0.29 |

### Condition scan, horizon=100 bars (15min)

| dimension | stratum | n | MM ret mean | random ret mean | MM-rnd | net$comm | win | p_cont | p_rev |
|---|---|---|---|---|---|---|---|---|---|
| direction | long | 860 | -0.70 | -0.15 | -0.55 | -2.41 | 0.52 | 0.50 | 0.48 |
| direction | short | 752 | -3.73 | -4.66 | 0.94 | -8.46 | 0.46 | 0.49 | 0.50 |
| session | Asia | 508 | 8.51 | -8.37 | 16.88 | 16.01 | 0.50 | 0.49 | 0.51 |
| session | Europe | 498 | -14.19 | -5.98 | -8.21 | -29.38 | 0.46 | 0.50 | 0.50 |
| session | US | 551 | 2.38 | 6.32 | -3.94 | 3.75 | 0.52 | 0.51 | 0.46 |
| session | Post | 55 | -35.88 | 2.10 | -37.99 | -72.76 | 0.45 | 0.42 | 0.58 |
| regime | low | 539 | 2.58 | -6.23 | 8.81 | 4.16 | 0.50 | 0.50 | 0.50 |
| regime | mid | 537 | -7.40 | -6.95 | -0.45 | -15.79 | 0.46 | 0.49 | 0.50 |
| regime | high | 534 | -1.18 | 6.76 | -7.94 | -3.35 | 0.52 | 0.50 | 0.48 |
| size_pts_band | low | 545 | 1.94 | -16.24 | 18.18 | 2.87 | 0.50 | 0.53 | 0.47 |
| size_pts_band | mid | 532 | -4.75 | 4.62 | -9.37 | -10.49 | 0.47 | 0.48 | 0.52 |
| size_pts_band | high | 535 | -3.63 | 5.17 | -8.80 | -8.25 | 0.50 | 0.48 | 0.49 |
| size_terc | low | 537 | -0.62 | -7.91 | 7.29 | -2.24 | 0.49 | 0.54 | 0.46 |
| size_terc | mid | 537 | -8.00 | 2.03 | -10.02 | -16.99 | 0.46 | 0.47 | 0.53 |
| size_terc | high | 538 | 2.26 | -0.89 | 3.16 | 3.53 | 0.52 | 0.49 | 0.48 |
| n_candles | 3 | 1376 | -3.68 | 0.90 | -4.57 | -8.35 | 0.49 | 0.49 | 0.50 |
| n_candles | 4 | 212 | 10.01 | -19.37 | 29.38 | 19.03 | 0.54 | 0.51 | 0.47 |
| n_candles | 5+ | 24 | -19.82 | -32.07 | 12.25 | -40.65 | 0.33 | 0.67 | 0.29 |

### Condition scan, horizon=400 bars (15min)

| dimension | stratum | n | MM ret mean | random ret mean | MM-rnd | net$comm | win | p_cont | p_rev |
|---|---|---|---|---|---|---|---|---|---|
| direction | long | 857 | 18.19 | 23.31 | -5.13 | 35.37 | 0.58 | 0.51 | 0.49 |
| direction | short | 749 | -25.71 | -9.44 | -16.27 | -52.43 | 0.43 | 0.49 | 0.51 |
| session | Asia | 506 | -1.93 | -4.93 | 3.00 | -4.85 | 0.49 | 0.49 | 0.51 |
| session | Europe | 495 | -9.34 | 10.48 | -19.82 | -19.68 | 0.49 | 0.51 | 0.49 |
| session | US | 550 | 10.62 | 15.90 | -5.28 | 20.23 | 0.54 | 0.52 | 0.48 |
| session | Post | 55 | -71.18 | 27.41 | -98.59 | -143.36 | 0.49 | 0.42 | 0.58 |
| regime | low | 536 | 0.65 | 8.37 | -7.72 | 0.30 | 0.50 | 0.50 | 0.50 |
| regime | mid | 534 | -3.27 | -2.34 | -0.92 | -7.53 | 0.51 | 0.49 | 0.51 |
| regime | high | 534 | -3.37 | 18.70 | -22.07 | -7.73 | 0.51 | 0.51 | 0.49 |
| size_pts_band | low | 541 | -0.93 | -12.91 | 11.98 | -2.86 | 0.50 | 0.53 | 0.47 |
| size_pts_band | mid | 530 | 2.71 | 13.86 | -11.15 | 4.42 | 0.51 | 0.48 | 0.52 |
| size_pts_band | high | 535 | -8.61 | 23.66 | -32.27 | -18.23 | 0.50 | 0.50 | 0.50 |
| size_terc | low | 535 | -8.50 | -19.54 | 11.04 | -18.00 | 0.51 | 0.53 | 0.47 |
| size_terc | mid | 535 | -6.49 | 36.01 | -42.50 | -13.97 | 0.49 | 0.47 | 0.53 |
| size_terc | high | 536 | 8.10 | 7.66 | 0.44 | 15.20 | 0.52 | 0.50 | 0.50 |
| n_candles | 3 | 1371 | -3.23 | 14.03 | -17.26 | -7.46 | 0.50 | 0.50 | 0.50 |
| n_candles | 4 | 211 | 6.44 | -19.10 | 25.55 | 11.88 | 0.53 | 0.52 | 0.48 |
| n_candles | 5+ | 24 | -25.31 | -94.11 | 68.80 | -51.62 | 0.54 | 0.71 | 0.29 |

### Continuation and projection completion (15min)

| horizon | P(reach 1xD) | P(reach 2xD) | P(reach 3xD) | p_cont | p_rev | completion median (MFE/D) |
|---|---|---|---|---|---|---|
| 5 | 0.327 | 0.134 | 0.070 | 0.299 | 0.276 | 0.58 |
| 10 | 0.449 | 0.240 | 0.137 | 0.377 | 0.362 | 0.85 |
| 20 | 0.592 | 0.369 | 0.245 | 0.439 | 0.435 | 1.37 |
| 50 | 0.729 | 0.544 | 0.419 | 0.478 | 0.470 | 2.28 |
| 100 | 0.822 | 0.676 | 0.559 | 0.497 | 0.492 | 3.57 |
| 200 | 0.874 | 0.763 | 0.665 | 0.503 | 0.495 | 5.40 |
| 400 | 0.918 | 0.832 | 0.754 | 0.503 | 0.497 | 7.50 |

Full-path MFE / MAE (to data end, all events; note this is dominated by drift/era length, not by the setup):

| metric | P25 | P50 | P75 | P90 |
|---|---|---|---|---|
| mfe (pts) | 1198.1 | 2760.2 | 4690.4 | 7964.8 |
| mae (pts) | 1138.0 | 2452.5 | 4495.5 | 7475.1 |

Median time-to-MFE (within the 400-bar window) = 213 bars. `to data end` returns are era/drift-dominated and are shown only for completeness.

## 5min

### Overall: MM forward return vs matched random vs drift

| horizon | MM n | MM ret mean | MM ret med | random ret mean | drift(long) | MM net$comm | MM net$full |
|---|---|---|---|---|---|---|---|
| 5 | 4780 | -1.32 | -1.25 | -0.24 | -0.02 | -3.63 | -4.03 |
| 10 | 4780 | -2.02 | -1.12 | -0.11 | 0.09 | -5.03 | -5.43 |
| 20 | 4780 | -2.18 | -1.50 | -0.92 | 0.47 | -5.36 | -5.76 |
| 50 | 4780 | -3.00 | -1.00 | -0.08 | 1.15 | -7.01 | -7.41 |
| 100 | 4779 | -1.64 | -0.75 | 0.55 | 2.47 | -4.28 | -4.68 |
| 200 | 4778 | -2.24 | -3.88 | 1.38 | 5.05 | -5.48 | -5.88 |
| 400 | 4771 | 2.19 | -0.75 | 0.26 | 9.41 | 3.39 | 2.99 |
| rev | 4780 | -1.44 | -3.00 | 0.37 | — | -3.88 | — |
| end | 4780 | 197.21 | 355.25 | 183.65 | — | 393.42 | — |

### Condition scan, horizon=20 bars (5min)

| dimension | stratum | n | MM ret mean | random ret mean | MM-rnd | net$comm | win | p_cont | p_rev |
|---|---|---|---|---|---|---|---|---|---|
| direction | long | 2502 | -0.80 | -0.02 | -0.77 | -2.60 | 0.50 | 0.41 | 0.46 |
| direction | short | 2278 | -3.69 | -1.91 | -1.79 | -8.39 | 0.45 | 0.41 | 0.47 |
| session | Asia | 1616 | -0.68 | 0.55 | -1.23 | -2.35 | 0.48 | 0.40 | 0.50 |
| session | Europe | 1423 | -1.72 | -0.62 | -1.10 | -4.45 | 0.48 | 0.43 | 0.47 |
| session | US | 1509 | -3.37 | -3.78 | 0.41 | -7.73 | 0.47 | 0.40 | 0.44 |
| session | Post | 232 | -7.69 | 5.52 | -13.22 | -16.38 | 0.38 | 0.39 | 0.38 |
| regime | low | 1592 | -1.30 | 0.05 | -1.35 | -3.61 | 0.47 | 0.41 | 0.51 |
| regime | mid | 1592 | -0.88 | -0.37 | -0.51 | -2.76 | 0.48 | 0.43 | 0.46 |
| regime | high | 1594 | -4.36 | -2.45 | -1.91 | -9.72 | 0.47 | 0.39 | 0.43 |
| size_pts_band | low | 1623 | -1.51 | -0.05 | -1.46 | -4.02 | 0.47 | 0.43 | 0.54 |
| size_pts_band | mid | 1579 | -2.03 | -0.12 | -1.91 | -5.06 | 0.47 | 0.44 | 0.47 |
| size_pts_band | high | 1578 | -3.01 | -2.62 | -0.39 | -7.03 | 0.48 | 0.35 | 0.38 |
| size_terc | low | 1589 | -1.27 | -0.31 | -0.96 | -3.54 | 0.48 | 0.47 | 0.53 |
| size_terc | mid | 1595 | -3.23 | -1.41 | -1.82 | -7.46 | 0.46 | 0.42 | 0.51 |
| size_terc | high | 1596 | -2.03 | -1.04 | -0.99 | -5.06 | 0.48 | 0.35 | 0.36 |
| n_candles | 3 | 4091 | -2.35 | -0.95 | -1.39 | -5.70 | 0.47 | 0.41 | 0.48 |
| n_candles | 4 | 568 | -2.39 | -0.08 | -2.31 | -5.78 | 0.46 | 0.38 | 0.42 |
| n_candles | 5+ | 121 | 4.56 | -3.75 | 8.31 | 8.12 | 0.57 | 0.34 | 0.29 |

### Condition scan, horizon=100 bars (5min)

| dimension | stratum | n | MM ret mean | random ret mean | MM-rnd | net$comm | win | p_cont | p_rev |
|---|---|---|---|---|---|---|---|---|---|
| direction | long | 2501 | -0.02 | 4.76 | -4.78 | -1.04 | 0.54 | 0.46 | 0.52 |
| direction | short | 2278 | -3.42 | -4.08 | 0.66 | -7.85 | 0.45 | 0.46 | 0.53 |
| session | Asia | 1616 | 0.05 | 2.34 | -2.28 | -0.89 | 0.50 | 0.45 | 0.55 |
| session | Europe | 1423 | 1.27 | 2.19 | -0.92 | 1.55 | 0.51 | 0.47 | 0.52 |
| session | US | 1508 | -4.22 | -3.06 | -1.16 | -9.44 | 0.49 | 0.46 | 0.49 |
| session | Post | 232 | -14.56 | 1.38 | -15.94 | -30.13 | 0.41 | 0.45 | 0.48 |
| regime | low | 1592 | 3.49 | 4.38 | -0.89 | 5.98 | 0.52 | 0.45 | 0.55 |
| regime | mid | 1591 | -0.46 | 2.46 | -2.91 | -1.91 | 0.49 | 0.47 | 0.51 |
| regime | high | 1594 | -7.93 | -5.21 | -2.73 | -16.86 | 0.47 | 0.45 | 0.50 |
| size_pts_band | low | 1622 | -2.91 | 3.62 | -6.53 | -6.82 | 0.48 | 0.45 | 0.55 |
| size_pts_band | mid | 1579 | 2.21 | 1.26 | 0.95 | 3.41 | 0.51 | 0.48 | 0.52 |
| size_pts_band | high | 1578 | -4.19 | -3.34 | -0.86 | -9.38 | 0.49 | 0.46 | 0.49 |
| size_terc | low | 1588 | -0.59 | -1.07 | 0.48 | -2.18 | 0.49 | 0.47 | 0.53 |
| size_terc | mid | 1595 | -5.53 | 1.24 | -6.77 | -12.05 | 0.48 | 0.45 | 0.54 |
| size_terc | high | 1596 | 1.19 | 1.47 | -0.27 | 1.39 | 0.51 | 0.46 | 0.49 |
| n_candles | 3 | 4090 | -2.34 | 1.07 | -3.42 | -5.69 | 0.50 | 0.46 | 0.53 |
| n_candles | 4 | 568 | -0.11 | -1.65 | 1.54 | -1.22 | 0.47 | 0.47 | 0.49 |
| n_candles | 5+ | 121 | 14.88 | -6.95 | 21.83 | 28.77 | 0.52 | 0.50 | 0.43 |

### Condition scan, horizon=400 bars (5min)

| dimension | stratum | n | MM ret mean | random ret mean | MM-rnd | net$comm | win | p_cont | p_rev |
|---|---|---|---|---|---|---|---|---|---|
| direction | long | 2495 | 9.10 | 11.52 | -2.42 | 17.21 | 0.54 | 0.47 | 0.53 |
| direction | short | 2276 | -5.38 | -12.13 | 6.75 | -11.76 | 0.45 | 0.46 | 0.53 |
| session | Asia | 1614 | -1.60 | 0.09 | -1.69 | -4.20 | 0.49 | 0.45 | 0.55 |
| session | Europe | 1420 | 10.90 | 4.52 | 6.37 | 20.79 | 0.50 | 0.47 | 0.53 |
| session | US | 1505 | 1.50 | -2.91 | 4.41 | 1.99 | 0.50 | 0.48 | 0.52 |
| session | Post | 232 | -20.17 | -4.23 | -15.94 | -41.34 | 0.47 | 0.48 | 0.52 |
| regime | low | 1590 | 5.83 | 4.18 | 1.65 | 10.67 | 0.51 | 0.45 | 0.55 |
| regime | mid | 1588 | -1.03 | 1.99 | -3.02 | -3.07 | 0.48 | 0.48 | 0.52 |
| regime | high | 1591 | 1.77 | -5.42 | 7.19 | 2.55 | 0.50 | 0.48 | 0.52 |
| size_pts_band | low | 1619 | -2.33 | 6.16 | -8.49 | -5.66 | 0.49 | 0.45 | 0.55 |
| size_pts_band | mid | 1574 | 2.39 | 5.82 | -3.43 | 3.78 | 0.50 | 0.48 | 0.52 |
| size_pts_band | high | 1578 | 6.64 | -11.45 | 18.09 | 12.27 | 0.50 | 0.48 | 0.52 |
| size_terc | low | 1587 | 3.38 | 4.58 | -1.20 | 5.76 | 0.50 | 0.47 | 0.53 |
| size_terc | mid | 1590 | -1.70 | 4.93 | -6.63 | -4.41 | 0.49 | 0.46 | 0.54 |
| size_terc | high | 1594 | 4.90 | -8.74 | 13.64 | 8.80 | 0.51 | 0.48 | 0.52 |
| n_candles | 3 | 4082 | 1.97 | 0.86 | 1.11 | 2.94 | 0.50 | 0.47 | 0.53 |
| n_candles | 4 | 568 | 4.77 | 3.94 | 0.84 | 8.55 | 0.52 | 0.49 | 0.51 |
| n_candles | 5+ | 121 | -2.48 | -37.90 | 35.41 | -5.97 | 0.47 | 0.53 | 0.47 |

### Continuation and projection completion (5min)

| horizon | P(reach 1xD) | P(reach 2xD) | P(reach 3xD) | p_cont | p_rev | completion median (MFE/D) |
|---|---|---|---|---|---|---|
| 5 | 0.297 | 0.118 | 0.062 | 0.256 | 0.290 | 0.54 |
| 10 | 0.434 | 0.199 | 0.112 | 0.349 | 0.391 | 0.81 |
| 20 | 0.561 | 0.326 | 0.200 | 0.409 | 0.465 | 1.21 |
| 50 | 0.703 | 0.511 | 0.373 | 0.448 | 0.511 | 2.06 |
| 100 | 0.789 | 0.636 | 0.515 | 0.460 | 0.521 | 3.14 |
| 200 | 0.851 | 0.735 | 0.638 | 0.467 | 0.528 | 4.81 |
| 400 | 0.904 | 0.820 | 0.741 | 0.470 | 0.530 | 7.20 |

Full-path MFE / MAE (to data end, all events; note this is dominated by drift/era length, not by the setup):

| metric | P25 | P50 | P75 | P90 |
|---|---|---|---|---|
| mfe (pts) | 1190.3 | 2637.8 | 4623.9 | 7739.1 |
| mae (pts) | 1125.0 | 2364.6 | 4507.2 | 7416.6 |

Median time-to-MFE (within the 400-bar window) = 200 bars. `to data end` returns are era/drift-dominated and are shown only for completeness.

## Validation

1. **No look-ahead**: the forward path uses bars `end+1 .. end+H` only; the combination is known at the close of `end`. Random anchors use `open[t+1]`. Manual spot-check of the first 15min tol=0.40 event (2019-05-06 09:45-10:15, dir=+1, D=12.5): combo open=11446.0, close[end]=11458.5, next-bar open (ref)=11458.25 (matches the stored `ref`), MM target=11471.0, hit within 20 bars=True, ret(H=20)=close[end+20]-ref=+51.50 pts = $103.00 gross. All recomputed by hand from raw bars; no future bar used.

2. **Pipeline consistency**: at horizon=20 the overall `p_reach1` reproduces the old `measured_move_stats.csv` `p_hit20_body`: 15min 0.592 (old 0.592) and 5min 0.561 (old 0.561).

3. **Cost example (1 contract, 1 round trip)**: gross +10 pts = $20.00; net$comm = $20.00 - 2x$0.50 = $19.00; net$full = $19.00 - 0.4-tick/side slippage ($0.20/side = 0.20 pt round trip = $0.40) = $18.60. At the measured mean returns (<=2 pts) the round-trip cost alone erases any edge.

## Caveats

- Events are auto-detected on every bar; neighbouring detection windows share candles and the sample is not independent.

- In-sample only; OOS (2024+) never touched. Any 'positive' cell is in-sample and vulnerable to multiple testing (many cells x horizons).

- The matched random baseline controls session+regime+direction but not trend context; results are still descriptive, not causal.

- `to data end` metrics mix different eras and the long-run drift; they are not comparable across events.
