# Live scoreboard

Forecasts issued at 12:00 (Europe/Berlin) the day before; each day's forecast is committed to
`forecasts/<day>.csv` before the outcome is known. nMAE = mean absolute error / mean actual load.

11 day(s) scored, 2026-11-01 → 2026-11-15.

| region | this model | SMARD | error vs SMARD |
|---|---|---|---|
| DE | 2.24% | 4.35% | -48% |
| 50Hertz | 3.63% | 7.27% | -50% |
| Amprion | 2.94% | 4.99% | -41% |
| TenneT | 3.77% | 6.44% | -41% |
| TransnetBW | 3.35% | 4.20% | -20% |
