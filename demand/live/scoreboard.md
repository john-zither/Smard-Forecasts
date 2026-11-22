# Live scoreboard

Forecasts issued at 12:00 (Europe/Berlin) the day before; each day's forecast is committed to
`forecasts/<day>.csv` before the outcome is known. nMAE = mean absolute error / mean actual load.

16 day(s) scored, 2026-11-01 → 2026-11-21.

| region | this model | SMARD | error vs SMARD |
|---|---|---|---|
| DE | 2.05% | 3.50% | -41% |
| 50Hertz | 4.30% | 7.05% | -39% |
| Amprion | 2.69% | 4.41% | -39% |
| TenneT | 3.34% | 5.30% | -37% |
| TransnetBW | 3.11% | 3.70% | -16% |

Uncertainty bands, share of quarter-hours inside (target 50% / 90%):

| region | days | 50% band | 90% band |
|---|---|---|---|
| DE | 5 | 50% | 91% |
| 50Hertz | 5 | 31% | 81% |
| Amprion | 5 | 65% | 99% |
| TenneT | 5 | 43% | 79% |
| TransnetBW | 5 | 44% | 82% |
