# Live scoreboard

Forecasts issued at 12:00 (Europe/Berlin) the day before; each day's forecast is committed to
`forecasts/<day>.csv` before the outcome is known. nMAE = mean absolute error / mean actual load.

8 day(s) scored, 2026-11-01 → 2026-11-10.

| region | this model | SMARD | error vs SMARD |
|---|---|---|---|
| DE | 2.19% | 4.29% | -49% |
| 50Hertz | 3.86% | 6.64% | -42% |
| Amprion | 3.16% | 5.16% | -39% |
| TenneT | 2.81% | 5.66% | -50% |
| TransnetBW | 3.27% | 4.07% | -20% |
