# Live scoreboard

Forecasts issued at 12:00 (Europe/Berlin) the day before; each day's forecast is committed to
`forecasts/<day>.csv` before the outcome is known. nMAE = mean absolute error / mean actual load.

11 day(s) scored, 2026-11-01 → 2026-11-15.

| region | this model | SMARD | error vs SMARD |
|---|---|---|---|
| DE | 2.05% | 4.14% | -50% |
| 50Hertz | 3.63% | 7.27% | -50% |
| Amprion | 2.94% | 4.99% | -41% |
| TenneT | 3.05% | 5.72% | -47% |
| TransnetBW | 3.33% | 4.19% | -20% |
