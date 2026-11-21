# Live scoreboard

Forecasts issued at 12:00 (Europe/Berlin) the day before; each day's forecast is committed to
`forecasts/<day>.csv` before the outcome is known. nMAE = mean absolute error / mean actual load.

15 day(s) scored, 2026-11-01 → 2026-11-20.

| region | this model | SMARD | error vs SMARD |
|---|---|---|---|
| DE | 1.98% | 3.57% | -45% |
| 50Hertz | 4.16% | 7.20% | -42% |
| Amprion | 2.73% | 4.46% | -39% |
| TenneT | 3.30% | 5.30% | -38% |
| TransnetBW | 3.19% | 3.76% | -15% |

Uncertainty bands, share of quarter-hours inside (target 50% / 90%):

| region | days | 50% band | 90% band |
|---|---|---|---|
| DE | 4 | 55% | 94% |
| 50Hertz | 4 | 34% | 77% |
| Amprion | 4 | 65% | 98% |
| TenneT | 4 | 42% | 79% |
| TransnetBW | 4 | 39% | 77% |
