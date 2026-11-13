# Live scoreboard

Forecasts issued at 12:00 (Europe/Berlin) the day before; each day's forecast is committed to
`forecasts/<day>.csv` before the outcome is known. nMAE = mean absolute error / mean actual load.

9 day(s) scored, 2026-11-01 → 2026-11-12.

| region | this model | SMARD | error vs SMARD |
|---|---|---|---|
| DE | 2.16% | 4.56% | -53% |
| 50Hertz | 3.59% | 7.36% | -51% |
| Amprion | 3.07% | 5.10% | -40% |
| TenneT | 2.95% | 5.84% | -49% |
| TransnetBW | 3.28% | 4.18% | -21% |
