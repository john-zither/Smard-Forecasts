# Live scoreboard

Forecasts issued at 12:00 (Europe/Berlin) the day before; each day's forecast is committed to
`forecasts/<day>.csv` before the outcome is known. nMAE = mean absolute error / mean actual load.

4 day(s) scored, 2026-11-01 → 2026-11-04.

| region | this model | SMARD | error vs SMARD |
|---|---|---|---|
| DE | 1.99% | 4.04% | -51% |
| 50Hertz | 3.22% | 6.33% | -49% |
| Amprion | 3.39% | 5.75% | -41% |
| TenneT | 2.89% | 5.65% | -49% |
| TransnetBW | 4.61% | 4.42% | +4% |
