# MEASURED MOVE — equal-length TREND candle combinations (IS only, 2019-05-06..2023-12-31)

**Pattern**: >=3 consecutive same-direction candles with approximately equal bodies (each body within ``tol`` of the running segment mean). **D** = dir*(close[last]-open[first]); **MM target** = close[last] + dir*D (D_sum variant: close[last] + dir*D_sum). A combination is known only at the close of its last candle; the forward path is measured from the **next bar** (reference = next open). OOS (2024+) never touched.

Cost model corrected 2026-10-10: commission $0.50/side ($1.00 open+close) + slippage $0.20/side; '1.5t' label retained, stress '3.0t' = $1.50/side slip. (This report is cost-free probabilities; the cost note applies to the downstream momentum/MM-performance reports.)

## Base-rate comparison (body target)

Random-bar base rate = P(a random bar's forward move within N bars reaches the same required distance from the next-bar reference), estimated on a 200k-bar random sample.

| freq | tol | N combos | P(hit≤5) | P(hit≤10) | P(hit≤20) | P(hit≤50) | base20 (req) | base20 (2*D) | edge20 |
|---|---|---|---|---|---|---|---|---|---|
| 15min | 0.25 | 641 | 0.351 | 0.468 | 0.610 | 0.752 | 0.573 | 0.393 | 0.037 |
| 15min | 0.4 | 1617 | 0.328 | 0.450 | 0.592 | 0.729 | 0.564 | 0.382 | 0.028 |
| 15min | 0.6 | 3593 | 0.299 | 0.423 | 0.570 | 0.717 | 0.558 | 0.379 | 0.012 |
| 5min | 0.25 | 2052 | 0.313 | 0.459 | 0.580 | 0.710 | 0.560 | 0.386 | 0.020 |
| 5min | 0.4 | 4787 | 0.298 | 0.435 | 0.561 | 0.703 | 0.544 | 0.368 | 0.018 |
| 5min | 0.6 | 10593 | 0.289 | 0.422 | 0.551 | 0.699 | 0.543 | 0.368 | 0.009 |
| 1min | 0.25 | 11263 | 0.334 | 0.462 | 0.591 | 0.727 | 0.566 | 0.399 | 0.026 |
| 1min | 0.4 | 25244 | 0.293 | 0.421 | 0.556 | 0.704 | 0.531 | 0.356 | 0.026 |
| 1min | 0.6 | 55915 | 0.278 | 0.404 | 0.540 | 0.694 | 0.532 | 0.359 | 0.007 |

## Characteristics table — 15min (which conditions RAISE MM hit probability)

### tol = 0.25  (N=641, P(hit≤20)=0.610, base20=0.573, P(touch start first)=0.456)

| strat | group | N | P(hit≤5) | P(hit≤10) | P(hit≤20) | P(hit≤50) | base20 | edge20 | touch≤20 | touch-first | MFE/D | MAE/D |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| direction | long | 356 | 0.326 | 0.441 | 0.590 | 0.753 | 0.575 | 0.015 | 0.598 | 0.444 | 1.35 | 1.41 |
| direction | short | 285 | 0.382 | 0.502 | 0.635 | 0.751 | 0.570 | 0.065 | 0.670 | 0.470 | 1.55 | 1.65 |
| n_candles | 3 | 590 | 0.359 | 0.475 | 0.610 | 0.753 | 0.576 | 0.034 | 0.632 | 0.453 | 1.46 | 1.53 |
| n_candles | 4 | 45 | 0.244 | 0.400 | 0.644 | 0.756 | 0.555 | 0.089 | 0.644 | 0.511 | 1.22 | 1.67 |
| n_candles | 5+ | 6 | 0.333 | 0.333 | 0.333 | 0.667 | 0.407 | -0.073 | 0.333 | 0.333 | 0.71 | 0.69 |
| size_tercile | low | 214 | 0.514 | 0.645 | 0.762 | 0.879 | 0.770 | -0.008 | 0.799 | 0.514 | 2.31 | 2.76 |
| size_tercile | mid | 213 | 0.333 | 0.455 | 0.624 | 0.765 | 0.590 | 0.034 | 0.634 | 0.484 | 1.54 | 1.68 |
| size_tercile | high | 214 | 0.206 | 0.304 | 0.444 | 0.612 | 0.360 | 0.084 | 0.458 | 0.369 | 0.84 | 0.92 |
| regime | low | 213 | 0.385 | 0.535 | 0.690 | 0.850 | 0.673 | 0.017 | 0.742 | 0.507 | 1.93 | 2.11 |
| regime | mid | 214 | 0.383 | 0.486 | 0.631 | 0.766 | 0.591 | 0.039 | 0.636 | 0.463 | 1.62 | 1.70 |
| regime | high | 212 | 0.278 | 0.377 | 0.505 | 0.637 | 0.452 | 0.053 | 0.509 | 0.401 | 1.03 | 1.01 |
| session | Asia | 199 | 0.327 | 0.462 | 0.593 | 0.824 | 0.685 | -0.092 | 0.608 | 0.467 | 1.35 | 1.59 |
| session | Europe | 203 | 0.443 | 0.571 | 0.764 | 0.852 | 0.623 | 0.140 | 0.813 | 0.512 | 2.06 | 2.42 |
| session | US | 217 | 0.281 | 0.378 | 0.484 | 0.590 | 0.420 | 0.064 | 0.479 | 0.392 | 0.96 | 0.96 |
| session | Post | 22 | 0.409 | 0.455 | 0.591 | 0.773 | 0.605 | -0.014 | 0.636 | 0.455 | 1.17 | 1.07 |

### tol = 0.4  (N=1617, P(hit≤20)=0.592, base20=0.564, P(touch start first)=0.455)

| strat | group | N | P(hit≤5) | P(hit≤10) | P(hit≤20) | P(hit≤50) | base20 | edge20 | touch≤20 | touch-first | MFE/D | MAE/D |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| direction | long | 863 | 0.314 | 0.445 | 0.592 | 0.739 | 0.568 | 0.024 | 0.582 | 0.440 | 1.32 | 1.33 |
| direction | short | 754 | 0.344 | 0.455 | 0.592 | 0.718 | 0.558 | 0.033 | 0.641 | 0.471 | 1.39 | 1.47 |
| n_candles | 3 | 1381 | 0.339 | 0.461 | 0.602 | 0.736 | 0.575 | 0.027 | 0.626 | 0.468 | 1.42 | 1.45 |
| n_candles | 4 | 212 | 0.255 | 0.382 | 0.538 | 0.708 | 0.513 | 0.025 | 0.533 | 0.387 | 1.15 | 1.10 |
| n_candles | 5+ | 24 | 0.333 | 0.375 | 0.458 | 0.542 | 0.361 | 0.097 | 0.333 | 0.292 | 0.81 | 0.70 |
| size_tercile | low | 539 | 0.506 | 0.644 | 0.774 | 0.881 | 0.758 | 0.015 | 0.761 | 0.501 | 2.50 | 2.53 |
| size_tercile | mid | 539 | 0.295 | 0.417 | 0.571 | 0.724 | 0.574 | -0.002 | 0.635 | 0.492 | 1.27 | 1.49 |
| size_tercile | high | 539 | 0.182 | 0.288 | 0.430 | 0.583 | 0.359 | 0.072 | 0.432 | 0.371 | 0.82 | 0.83 |
| regime | low | 541 | 0.379 | 0.518 | 0.660 | 0.815 | 0.675 | -0.015 | 0.713 | 0.494 | 1.76 | 2.04 |
| regime | mid | 537 | 0.343 | 0.460 | 0.615 | 0.764 | 0.579 | 0.036 | 0.626 | 0.475 | 1.51 | 1.41 |
| regime | high | 537 | 0.259 | 0.369 | 0.499 | 0.607 | 0.435 | 0.064 | 0.486 | 0.397 | 0.98 | 0.97 |
| session | Asia | 511 | 0.331 | 0.470 | 0.616 | 0.820 | 0.690 | -0.074 | 0.632 | 0.474 | 1.49 | 1.53 |
| session | Europe | 499 | 0.395 | 0.529 | 0.715 | 0.824 | 0.608 | 0.107 | 0.749 | 0.503 | 1.95 | 2.09 |
| session | US | 552 | 0.274 | 0.375 | 0.475 | 0.567 | 0.404 | 0.071 | 0.458 | 0.386 | 0.92 | 0.90 |
| session | Post | 55 | 0.236 | 0.291 | 0.418 | 0.655 | 0.591 | -0.173 | 0.636 | 0.527 | 0.82 | 1.45 |

### tol = 0.6  (N=3593, P(hit≤20)=0.570, base20=0.558, P(touch start first)=0.473)

| strat | group | N | P(hit≤5) | P(hit≤10) | P(hit≤20) | P(hit≤50) | base20 | edge20 | touch≤20 | touch-first | MFE/D | MAE/D |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| direction | long | 1951 | 0.290 | 0.423 | 0.578 | 0.729 | 0.566 | 0.011 | 0.579 | 0.454 | 1.27 | 1.27 |
| direction | short | 1642 | 0.310 | 0.423 | 0.561 | 0.702 | 0.547 | 0.014 | 0.649 | 0.495 | 1.27 | 1.48 |
| n_candles | 3 | 2784 | 0.327 | 0.449 | 0.591 | 0.728 | 0.580 | 0.011 | 0.636 | 0.489 | 1.42 | 1.45 |
| n_candles | 4 | 647 | 0.213 | 0.345 | 0.512 | 0.692 | 0.502 | 0.010 | 0.563 | 0.439 | 1.03 | 1.18 |
| n_candles | 5+ | 162 | 0.167 | 0.284 | 0.438 | 0.617 | 0.388 | 0.050 | 0.370 | 0.333 | 0.78 | 0.74 |
| size_tercile | low | 1198 | 0.514 | 0.644 | 0.764 | 0.883 | 0.766 | -0.002 | 0.777 | 0.530 | 2.46 | 2.33 |
| size_tercile | mid | 1197 | 0.236 | 0.383 | 0.550 | 0.707 | 0.564 | -0.014 | 0.632 | 0.520 | 1.18 | 1.43 |
| size_tercile | high | 1198 | 0.147 | 0.242 | 0.396 | 0.561 | 0.343 | 0.053 | 0.424 | 0.369 | 0.75 | 0.83 |
| regime | low | 1198 | 0.358 | 0.517 | 0.668 | 0.826 | 0.673 | -0.005 | 0.714 | 0.520 | 1.76 | 1.91 |
| regime | mid | 1198 | 0.306 | 0.429 | 0.591 | 0.748 | 0.570 | 0.021 | 0.619 | 0.472 | 1.37 | 1.38 |
| regime | high | 1195 | 0.233 | 0.321 | 0.450 | 0.576 | 0.429 | 0.021 | 0.500 | 0.428 | 0.84 | 1.00 |
| session | Asia | 1140 | 0.318 | 0.461 | 0.609 | 0.820 | 0.686 | -0.077 | 0.648 | 0.495 | 1.46 | 1.52 |
| session | Europe | 1069 | 0.342 | 0.492 | 0.686 | 0.813 | 0.595 | 0.091 | 0.746 | 0.527 | 1.81 | 1.95 |
| session | US | 1214 | 0.248 | 0.337 | 0.439 | 0.540 | 0.396 | 0.043 | 0.461 | 0.405 | 0.83 | 0.91 |
| session | Post | 170 | 0.271 | 0.341 | 0.518 | 0.682 | 0.613 | -0.095 | 0.582 | 0.471 | 1.02 | 1.15 |

## D_sum target variant (15min, overall)

| tol | N | P(hit≤20) body | P(hit≤20) D_sum | base20 body | base20 D_sum |
|---|---|---|---|---|---|
| 0.25 | 641 | 0.610 | 0.608 | 0.573 | 0.573 |
| 0.4 | 1617 | 0.592 | 0.591 | 0.564 | 0.563 |
| 0.6 | 3593 | 0.570 | 0.571 | 0.558 | 0.557 |

## Size filter switch (mean body >= k*ATR, 15min, overall)

| tol | k | N | P(hit≤20) | base20 | P(touch start first) |
|---|---|---|---|---|---|
| 0.25 | 0 | 641 | 0.610 | 0.573 | 0.456 |
| 0.25 | 0.3 | 459 | 0.549 | 0.491 | 0.429 |
| 0.25 | 0.5 | 268 | 0.463 | 0.387 | 0.399 |
| 0.4 | 0 | 1617 | 0.592 | 0.564 | 0.455 |
| 0.4 | 0.3 | 1160 | 0.523 | 0.483 | 0.434 |
| 0.4 | 0.5 | 686 | 0.449 | 0.391 | 0.407 |
| 0.6 | 0 | 3593 | 0.570 | 0.558 | 0.473 |
| 0.6 | 0.3 | 2561 | 0.493 | 0.470 | 0.446 |
| 0.6 | 0.5 | 1494 | 0.423 | 0.383 | 0.412 |

## Timeframe comparison (tol=0.40, k=0, overall)

| freq | N | P(hit≤20) | base20 | edge20 | P(touch start first) | median bars-to-target |
|---|---|---|---|---|---|---|
| 15min | 1617 | 0.592 | 0.564 | 0.028 | 0.455 | 5.0 |
| 5min | 4787 | 0.561 | 0.544 | 0.018 | 0.486 | 5.0 |
| 1min | 25244 | 0.556 | 0.531 | 0.026 | 0.492 | 5.0 |

## Caveats

- **Overlapping / selection**: combinations come from every bar and neighbouring combinations share candles; hit probabilities are descriptive, not independent tests.

- **Base rate**: a random bar is required to travel the *same* distance from the next-bar reference; the full move from the combination start is 2*D, but measurement begins one bar later, so the required forward move is target minus next open.

- **Definition-dependent**: looser `tol` admits noisier combinations; results move with the chosen tolerance.

- **IS only**: 2024-01..2026-10 (OOS) NOT touched. Any edge here is in-sample and likely overstated.
