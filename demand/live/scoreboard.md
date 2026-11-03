# Live scoreboard

Forecasts issued at 12:00 (Europe/Berlin) the day before; each day's forecast is committed to
`forecasts/<day>.csv` before the outcome is known. nMAE = mean absolute error / mean actual load.

2 day(s) scored, 2026-11-01 → 2026-11-02.

| region | this model | SMARD | error vs SMARD |
|---|---|---|---|
| DE | 2.10% | 2.65% | -21% |
| 50Hertz | 2.30% | 7.48% | -69% |
| Amprion | 4.08% | 8.38% | -51% |
| TenneT | 5.30% | 5.36% | -1% |
| TransnetBW | 2.42% | 4.48% | -46% |
