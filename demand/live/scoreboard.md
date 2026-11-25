# Live scoreboard

Forecasts issued at 12:00 (Europe/Berlin) the day before; each day's forecast is committed to
`forecasts/<day>.csv` before the outcome is known. nMAE = mean absolute error / mean actual load.

19 day(s) scored, 2026-11-01 → 2026-11-24.

| region | this model | SMARD | error vs SMARD |
|---|---|---|---|
| DE | 2.00% | 3.59% | -44% |
| 50Hertz | 4.23% | 7.05% | -40% |
| Amprion | 2.55% | 4.57% | -44% |
| TenneT | 3.43% | 5.29% | -35% |
| TransnetBW | 3.01% | 3.78% | -21% |

Uncertainty bands, share of quarter-hours inside (target 50% / 90%):

| region | days | 50% band | 90% band |
|---|---|---|---|
| DE | 8 | 52% | 92% |
| 50Hertz | 7 | 37% | 84% |
| Amprion | 8 | 69% | 99% |
| TenneT | 7 | 43% | 82% |
| TransnetBW | 8 | 43% | 88% |
