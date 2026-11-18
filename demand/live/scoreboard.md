# Live scoreboard

Forecasts issued at 12:00 (Europe/Berlin) the day before; each day's forecast is committed to
`forecasts/<day>.csv` before the outcome is known. nMAE = mean absolute error / mean actual load.

12 day(s) scored, 2026-11-01 → 2026-11-17.

| region | this model | SMARD | error vs SMARD |
|---|---|---|---|
| DE | 2.03% | 4.09% | -50% |
| 50Hertz | 3.69% | 7.52% | -51% |
| Amprion | 2.97% | 5.01% | -41% |
| TenneT | 3.09% | 5.62% | -45% |
| TransnetBW | 3.18% | 3.96% | -20% |

Uncertainty bands, share of quarter-hours inside (target 50% / 90%):

| region | days | 50% band | 90% band |
|---|---|---|---|
| DE | 1 | 59% | 96% |
| 50Hertz | 1 | 51% | 89% |
| Amprion | 1 | 50% | 96% |
| TenneT | 1 | 36% | 86% |
| TransnetBW | 1 | 71% | 96% |
