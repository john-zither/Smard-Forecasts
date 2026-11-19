# Live scoreboard

Forecasts issued at 12:00 (Europe/Berlin) the day before; each day's forecast is committed to
`forecasts/<day>.csv` before the outcome is known. nMAE = mean absolute error / mean actual load.

13 day(s) scored, 2026-11-01 → 2026-11-18.

| region | this model | SMARD | error vs SMARD |
|---|---|---|---|
| DE | 2.22% | 3.98% | -44% |
| 50Hertz | 4.13% | 7.47% | -45% |
| Amprion | 2.87% | 4.82% | -40% |
| TenneT | 3.52% | 5.76% | -39% |
| TransnetBW | 3.07% | 3.80% | -19% |

Uncertainty bands, share of quarter-hours inside (target 50% / 90%):

| region | days | 50% band | 90% band |
|---|---|---|---|
| DE | 2 | 38% | 79% |
| 50Hertz | 2 | 30% | 69% |
| Amprion | 2 | 62% | 98% |
| TenneT | 2 | 29% | 64% |
| TransnetBW | 2 | 61% | 96% |
