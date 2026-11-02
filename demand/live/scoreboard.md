# Live scoreboard

Forecasts issued at 12:00 (Europe/Berlin) the day before; each day's forecast is committed to
`forecasts/<day>.csv` before the outcome is known. nMAE = mean absolute error / mean actual load.

1 day(s) scored, 2026-11-01 → 2026-11-01.

| region | this model | SMARD | error vs SMARD |
|---|---|---|---|
| DE | 4.36% | 4.58% | -5% |
| 50Hertz | 1.47% | 8.02% | -82% |
| Amprion | 2.99% | 9.70% | -69% |
| TenneT | 15.82% | 15.72% | +1% |
| TransnetBW | 2.72% | 5.55% | -51% |
