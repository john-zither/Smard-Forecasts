# Live scoreboard

Forecasts issued at 12:00 (Europe/Berlin) the day before; each day's forecast is committed to
`forecasts/<day>.csv` before the outcome is known. nMAE = mean absolute error / mean actual load.

17 day(s) scored, 2026-11-01 → 2026-11-22.

| region | this model | SMARD | error vs SMARD |
|---|---|---|---|
| DE | 2.07% | 3.45% | -40% |
| 50Hertz | 4.34% | 7.32% | -41% |
| Amprion | 2.65% | 4.43% | -40% |
| TenneT | 3.33% | 5.37% | -38% |
| TransnetBW | 3.11% | 3.66% | -15% |

Uncertainty bands, share of quarter-hours inside (target 50% / 90%):

| region | days | 50% band | 90% band |
|---|---|---|---|
| DE | 6 | 49% | 90% |
| 50Hertz | 6 | 33% | 81% |
| Amprion | 6 | 68% | 99% |
| TenneT | 6 | 42% | 82% |
| TransnetBW | 6 | 41% | 84% |
